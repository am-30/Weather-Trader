"""
Historical replay engine for backtesting the Kalshi weather MC model.

Reconstructs the exact MCParams that would have been used at a given historical
hour on a given date, using only data that was available at that moment (no
lookahead / future leakage). Runs the MC simulation and compares the resulting
probability distribution to the actual NWS settlement outcome.

Usage::

    from datetime import date
    from kalshi_weather_trader.backtesting.replay_engine import ReplayEngine

    engine = ReplayEngine()
    df = engine.replay_all(
        start_date=date(2026, 3, 10),
        end_date=date(2026, 3, 22),
    )
    print(df[["target_date", "eval_hour", "mean_max", "actual_high"]].head(10))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone
from typing import Optional

import numpy as np
import pandas as pd
import pytz
import structlog

from kalshi_weather_trader.db.db_manager import (
    get_asos_readings_for_date,
    get_market,
    get_nwp_forecasts_before_utc,
    get_system_state,
)
from kalshi_weather_trader.db.schemas import (
    ASOSReadingDocument,
    NWPForecastDocument,
    SystemStateDocument,
)
from kalshi_weather_trader.quant.mc_params_builder import build_mc_params_historical
from kalshi_weather_trader.quant.monte_carlo import price_full_distribution

logger = structlog.get_logger(__name__)

_EASTERN = pytz.timezone("America/New_York")
_DEFAULT_EVAL_HOURS = [8, 10, 12, 14, 16]
_DEFAULT_SEED = 42


@dataclass
class ReplayResult:
    """All inputs and outputs from one (date, eval_hour) MC replay.

    Attributes:
        target_date:       Trading date replayed.
        eval_hour:         Eastern Time hour at which the replay was anchored.
        T0:                Starting temperature used (°F).
        hard_floor:        Maximum ASOS observation up to eval_hour (°F).
        sigma:             OU diffusion coefficient used (°F/√hr).
        theta:             OU mean-reversion speed used (per hr).
        bias:              Kalman bias correction used (°F).
        nwp_predicted_high: Maximum of the blended NWP hourly curve (°F).
        actual_high:       NWS official daily high (°F) — the settlement value.
        strike_probs:      Dict of strike → P(daily_max >= strike).
        brier_scores:      Dict of strike → (predicted_prob - outcome)^2.
        mean_max:          Mean of the paths_max distribution (°F).
        std_max:           Std of the paths_max distribution (°F).
    """

    target_date: date
    eval_hour: int
    T0: float
    hard_floor: float
    sigma: float
    theta: float
    bias: float
    nwp_predicted_high: float
    actual_high: float
    strike_probs: dict[float, float] = field(default_factory=dict)
    brier_scores: dict[float, float] = field(default_factory=dict)
    mean_max: float = 0.0
    std_max: float = 0.0


def _eval_utc(target_date: date, hour_et: int) -> datetime:
    """Return UTC datetime for a given ET hour on target_date.

    Args:
        target_date: The calendar date (ET).
        hour_et:     Eastern Time hour (0–23).

    Returns:
        Timezone-aware UTC datetime.
    """
    et_dt = _EASTERN.localize(
        datetime.combine(target_date, time(hour_et, 0, 0))
    )
    return et_dt.astimezone(timezone.utc)


def _closest_reading(
    readings: list[ASOSReadingDocument], eval_utc: datetime
) -> Optional[ASOSReadingDocument]:
    """Return the reading with observation_time_utc closest to eval_utc.

    Args:
        readings:  List of ASOSReadingDocuments, all <= eval_utc.
        eval_utc:  Reference UTC timestamp.

    Returns:
        Closest reading, or None if list is empty.
    """
    if not readings:
        return None
    return min(
        readings,
        key=lambda r: abs((r.observation_time_utc.replace(tzinfo=timezone.utc)
                           if r.observation_time_utc.tzinfo is None
                           else r.observation_time_utc) - eval_utc),
    )


def _blend_nwp(
    forecasts: dict[str, NWPForecastDocument],
    model_weights: Optional[dict[str, float]] = None,
) -> list[float]:
    """Blend NWP model hourly curves into a single weighted-average curve.

    Uses model_weights if provided (normalised to available models). Falls back
    to equal weights when model_weights is None or missing keys.

    Args:
        forecasts:     Dict of model_name → NWPForecastDocument.
        model_weights: Optional dict of model_name → weight (need not sum to 1).

    Returns:
        Blended hourly temperature curve (list of floats, ET-indexed, same
        length as the shortest available model curve). Empty list if no models.
    """
    if not forecasts:
        return []

    models = list(forecasts.keys())
    min_len = min(len(f.hourly_temps) for f in forecasts.values())

    # Determine per-model weights
    if model_weights:
        raw = {m: model_weights.get(m, 1.0) for m in models}
    else:
        raw = {m: 1.0 for m in models}

    total = sum(raw.values())
    if total <= 0:
        total = len(models)
        raw = {m: 1.0 for m in models}

    weights = {m: raw[m] / total for m in models}

    blended = [0.0] * min_len
    for m, doc in forecasts.items():
        w = weights[m]
        for i in range(min_len):
            blended[i] += w * doc.hourly_temps[i]

    return blended


def _default_strikes(nwp_high: float) -> list[float]:
    """Generate a default strike list centered on the NWP predicted high.

    Args:
        nwp_high: NWP-predicted daily maximum temperature (°F).

    Returns:
        List of integer strike values in [nwp_high - 8, nwp_high + 8] step 1°F.
    """
    center = int(round(nwp_high))
    return [float(s) for s in range(center - 8, center + 9)]


class ReplayEngine:
    """Historical MC replay engine.

    Reconstructs the probability estimates that the MC model would have produced
    at specific hours on historical dates, using only data available at those
    moments. Compares the estimates to actual NWS settlement outcomes.

    Example::

        engine = ReplayEngine()
        results = engine.replay_date(date(2026, 3, 15), eval_hours=[10, 14])
        df = engine.replay_all(date(2026, 3, 10), date(2026, 3, 22))
    """

    def replay_date(
        self,
        target_date: date,
        eval_hours: list[int] = _DEFAULT_EVAL_HOURS,
        seed: int = _DEFAULT_SEED,
    ) -> list[ReplayResult]:
        """Replay one historical date at multiple intraday evaluation hours.

        For each eval_hour, reconstructs MCParams using only data that existed
        at that moment (no lookahead), runs the MC simulation, and computes
        probability estimates and Brier scores against the actual outcome.

        Args:
            target_date: Historical trading date to replay.
            eval_hours:  List of ET hours at which to anchor the simulation.
                         Defaults to [8, 10, 12, 14, 16].
            seed:        Random seed for reproducible MC draws. Defaults to 42.

        Returns:
            List of ReplayResult, one per eval_hour where sufficient data exists.
            Hours with no ASOS readings or no settled outcome are skipped.

        Raises:
            Nothing — errors are logged and the affected hour is skipped.
        """
        results: list[ReplayResult] = []

        # Fetch all ASOS for the date once (sorted oldest-first)
        all_asos = get_asos_readings_for_date(target_date)

        # Fetch market for outcome
        market = get_market(target_date)
        if market is None or market.final_official_high is None:
            logger.info(
                "replay.skip_date.no_settlement",
                target_date=str(target_date),
            )
            return results

        actual_high = float(market.final_official_high)

        # Fetch system_state for calibrated params (set once per day at start-of-day)
        state: Optional[SystemStateDocument] = get_system_state(target_date)

        for hour_et in eval_hours:
            try:
                result = self._replay_hour(
                    target_date=target_date,
                    hour_et=hour_et,
                    all_asos=all_asos,
                    state=state,
                    actual_high=actual_high,
                    seed=seed,
                )
                if result is not None:
                    results.append(result)
            except Exception as exc:
                logger.warning(
                    "replay.hour_failed",
                    target_date=str(target_date),
                    hour_et=hour_et,
                    error=str(exc),
                )

        return results

    def _replay_hour(
        self,
        target_date: date,
        hour_et: int,
        all_asos: list[ASOSReadingDocument],
        state: Optional[SystemStateDocument],
        actual_high: float,
        seed: int,
    ) -> Optional[ReplayResult]:
        """Internal: replay one (date, hour) pair.

        Args:
            target_date: Trading date.
            hour_et:     ET hour anchor.
            all_asos:    All ASOS readings for the date.
            state:       SystemStateDocument for the date, or None.
            actual_high: NWS official daily high (settlement).
            seed:        MC random seed.

        Returns:
            ReplayResult if successful, None if insufficient data.

        Raises:
            Exception propagated to caller for logging.
        """
        cutoff_utc = _eval_utc(target_date, hour_et)

        # Filter ASOS to readings that existed at eval_hour (no future leakage)
        past_asos = [
            r for r in all_asos
            if (r.observation_time_utc.replace(tzinfo=timezone.utc)
                if r.observation_time_utc.tzinfo is None
                else r.observation_time_utc) <= cutoff_utc
        ]

        if not past_asos:
            logger.debug(
                "replay.skip_hour.no_asos",
                target_date=str(target_date),
                hour_et=hour_et,
            )
            return None

        # T0: reading closest to eval_hour (from the past-only set)
        asos_at_hour = _closest_reading(past_asos, cutoff_utc)
        assert asos_at_hour is not None  # guaranteed since past_asos non-empty

        # Hard floor: max observed temperature up to eval_hour (no future leakage)
        hard_floor = max(r.temperature_f for r in past_asos)

        # NWP: only fetches that existed at eval_hour
        nwp_forecasts = get_nwp_forecasts_before_utc(target_date, cutoff_utc)
        model_weights = state.model_weights if state is not None else None
        blended_curve = _blend_nwp(nwp_forecasts, model_weights)
        nwp_predicted_high = max(blended_curve) if blended_curve else asos_at_hour.temperature_f

        # Build MCParams using the historical factory
        params = build_mc_params_historical(
            past_date=target_date,
            hour_et=hour_et,
            state=state,
            asos_at_hour=asos_at_hour,
            hard_floor=hard_floor,
            nwp_curve=blended_curve,
        )

        # Determine strikes: from market cap/floor if stored, else NWP-centered default
        market = get_market(target_date)
        if (
            market is not None
            and hasattr(market, "floor_strike")
            and market.floor_strike is not None  # type: ignore[attr-defined]
            and hasattr(market, "cap_strike")
            and market.cap_strike is not None  # type: ignore[attr-defined]
        ):
            strikes = [
                float(s)
                for s in range(
                    int(market.floor_strike),  # type: ignore[attr-defined]
                    int(market.cap_strike) + 1,  # type: ignore[attr-defined]
                )
            ]
        else:
            strikes = _default_strikes(nwp_predicted_high)

        # Run MC simulation
        mc_result = price_full_distribution(
            params=params,
            strikes=strikes,
            target_date=target_date,
            seed=seed,
        )

        # Compute Brier scores: (predicted_prob - outcome)^2 per strike
        brier: dict[float, float] = {}
        for strike, prob in mc_result.probabilities.items():
            outcome = 1.0 if actual_high >= strike else 0.0
            brier[strike] = (prob - outcome) ** 2

        sigma = state.sigma_volatility if state is not None else params.sigma
        theta = state.theta_decay if state is not None else params.theta
        bias = state.kalman_bias_estimate if state is not None else params.bias

        return ReplayResult(
            target_date=target_date,
            eval_hour=hour_et,
            T0=params.T0,
            hard_floor=hard_floor,
            sigma=sigma,
            theta=theta,
            bias=bias,
            nwp_predicted_high=nwp_predicted_high,
            actual_high=actual_high,
            strike_probs=dict(mc_result.probabilities),
            brier_scores=brier,
            mean_max=mc_result.mean_max,
            std_max=mc_result.std_max,
        )

    def replay_all(
        self,
        start_date: date,
        end_date: date,
        eval_hours: list[int] = _DEFAULT_EVAL_HOURS,
        seed: int = _DEFAULT_SEED,
    ) -> pd.DataFrame:
        """Run replay_date for every calendar date in [start_date, end_date].

        Only settled dates (final_official_high is not None) produce rows.
        Dates with no ASOS data are silently skipped.

        Args:
            start_date: First date to replay (inclusive).
            end_date:   Last date to replay (inclusive).
            eval_hours: ET hours to evaluate at each date.
            seed:       Random seed for all MC draws (same seed per date/hour
                        so results are reproducible across runs).

        Returns:
            DataFrame with one row per (target_date, eval_hour). Columns:
            target_date, eval_hour, T0, hard_floor, sigma, theta, bias,
            nwp_predicted_high, actual_high, mean_max, std_max,
            plus one ``prob_{strike}`` column and one ``brier_{strike}`` column
            per strike value encountered.

        Raises:
            Nothing — failed dates are skipped and logged.
        """
        from datetime import timedelta

        rows: list[dict] = []

        current = start_date
        while current <= end_date:
            daily_results = self.replay_date(current, eval_hours=eval_hours, seed=seed)
            for r in daily_results:
                row: dict = {
                    "target_date": r.target_date,
                    "eval_hour": r.eval_hour,
                    "T0": r.T0,
                    "hard_floor": r.hard_floor,
                    "sigma": r.sigma,
                    "theta": r.theta,
                    "bias": r.bias,
                    "nwp_predicted_high": r.nwp_predicted_high,
                    "actual_high": r.actual_high,
                    "mean_max": r.mean_max,
                    "std_max": r.std_max,
                }
                for strike, prob in r.strike_probs.items():
                    row[f"prob_{strike:.1f}"] = prob
                for strike, bs in r.brier_scores.items():
                    row[f"brier_{strike:.1f}"] = bs
                rows.append(row)
            current += timedelta(days=1)

        if not rows:
            logger.warning(
                "replay_all.no_results",
                start_date=str(start_date),
                end_date=str(end_date),
            )
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        logger.info(
            "replay_all.done",
            n_rows=len(df),
            n_dates=df["target_date"].nunique(),
        )
        return df
