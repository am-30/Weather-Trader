"""
Historical replay engine for backtesting the Kalshi weather MC model.

Reconstructs the exact MCParams that would have been used at a given historical
hour on a given date, using only data that was available at that moment (no
lookahead / future leakage). Runs the MC simulation and compares the resulting
probability distribution to the actual NWS settlement outcome.

Two engines are provided:

ReplayEngine (original)
    Simple date-range loop.  Returns a flat DataFrame.  Used by Tab 5 system
    health backtesting runner and existing tests.  Do NOT modify.

ParameterizedReplayEngine (Model Lab — Phase L1)
    Takes a Scenario object, applies parameter overrides to the historically
    calibrated MCParams, and returns list[ParameterizedReplayResult].
    Designed for A/B comparison, parameter sensitivity, and optimization.

Usage::

    from datetime import date
    from kalshi_weather_trader.backtesting.replay_engine import (
        ParameterizedReplayEngine,
    )
    from kalshi_weather_trader.backtesting.scenarios import preset_production

    engine = ParameterizedReplayEngine()
    results = engine.replay_scenario(
        scenario=preset_production(),
        start_date=date(2026, 3, 10),
        end_date=date(2026, 3, 30),
    )
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


def _blend_cloudcover(forecasts: dict[str, NWPForecastDocument]) -> Optional[float]:
    """Equal-weight blend of mean_cloudcover_10_16 across models that have it."""
    values = [
        f.mean_cloudcover_10_16
        for f in forecasts.values()
        if f.mean_cloudcover_10_16 is not None
    ]
    return round(sum(values) / len(values), 1) if values else None


def _blend_ensemble_spread(forecasts: dict[str, NWPForecastDocument]) -> Optional[float]:
    """Equal-weight blend of ensemble_spread across models that have it."""
    values = [
        f.ensemble_spread
        for f in forecasts.values()
        if f.ensemble_spread is not None
    ]
    return round(sum(values) / len(values), 2) if values else None


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


# ===========================================================================
# Model Lab — Phase L1
# ===========================================================================
# The classes below are INDEPENDENT of the ReplayEngine above.  They use the
# Scenario system from backtesting.scenarios and return ParameterizedReplayResult
# objects (not DataFrames) for use by the Model Lab UI and metrics functions.
# ===========================================================================


@dataclass
class ParameterizedReplayResult:
    """All inputs and outputs from one (date, eval_hour) parameterized MC replay.

    Attributes:
        target_date:       Trading date replayed.
        eval_hour:         Eastern Time hour at which the replay was anchored.
        T0:                Starting temperature used (°F, from Kalman or ASOS).
        hard_floor:        Max ASOS observation up to eval_hour (°F).
        effective_floor:   hard_floor + persistence_offset applied to paths_max.
        sigma_used:        Sigma values actually used in MC.  Dict because
                           time-varying sigma has one value per block:
                           {"0-6": x, "6-10": x, ...} or {"scalar": x}.
        theta_used:        Theta values actually used:
                           {"am": x, "pm": x} or {"scalar": x}.
        bias_used:         Kalman bias correction used (°F).
        drift_used:        Drift adjustment included in attractor (0.0 if disabled).
        nwp_predicted_high: Max of the blended NWP hourly curve (°F).
        attractor_peak:    Approximate peak attractor value: max(nwp_curve) + bias_used.
                           (Exact anchor offset requires re-running the inner loop;
                           this is a useful approximation for diagnostics.)
        mean_max:          Mean of paths_max distribution (°F).
        std_max:           Std of paths_max distribution (°F).
        percentiles:       {10: val, 25: val, 50: val, 75: val, 90: val} of paths_max.
        strike_probs:      {strike: P(daily_max >= strike)} from MC.
        market_probs:      {str(strike): prob} — alias for UI display.
        actual_high:       NWS official daily high used for scoring (°F).
        prediction_error:  mean_max − actual_high (positive = over-predict).
        brier_components:  {str(strike): (prob − outcome)²} per strike.
    """

    target_date: date
    eval_hour: int
    T0: float
    hard_floor: float
    effective_floor: float
    sigma_used: dict
    theta_used: dict
    bias_used: float
    drift_used: float
    nwp_predicted_high: float
    attractor_peak: float
    mean_max: float
    std_max: float
    percentiles: dict
    strike_probs: dict
    market_probs: dict
    actual_high: float
    prediction_error: float
    brier_components: dict


@dataclass
class ComparisonResult:
    """Output of compare_scenarios(): two scenario replay runs + bootstrap comparison.

    Attributes:
        scenario_a_name: Display name for scenario A.
        scenario_b_name: Display name for scenario B.
        a_metrics:       compute_aggregate_metrics(a_results).
        b_metrics:       compute_aggregate_metrics(b_results).
        bootstrap:       BootstrapResult from compute_paired_bootstrap().
                         None if either scenario produced no results.
        a_results:       list[ParameterizedReplayResult] for scenario A.
        b_results:       list[ParameterizedReplayResult] for scenario B.
        common_dates:    Sorted list of dates present in both A and B results.
    """

    scenario_a_name: str
    scenario_b_name: str
    a_metrics: dict
    b_metrics: dict
    bootstrap: object      # BootstrapResult | None
    a_results: list        # list[ParameterizedReplayResult]
    b_results: list        # list[ParameterizedReplayResult]
    common_dates: list     # sorted list[date]


_OVERNIGHT_DECAY_HOURS = 8.0  # approximate gap from last eval point to next day midnight


def _replay_kalman_bias_intraday(
    asos_readings: list,
    nwp_curve: list,
    eval_hour: int,
    initial_bias: float = 0.0,
) -> float:
    """Re-run the current Kalman filter over historical ASOS readings.

    Starts the filter at midnight ET with ``initial_bias`` (0.0 for a cold
    start; yesterday's decayed bias for warm-start chaining), feeds ASOS
    readings chronologically with hourly NWP predict steps, and returns the
    bias estimate at eval_hour.  Uses the current filter configuration
    (H=[[1,1]], bias decay, covariance cap) regardless of what was live on the
    historical date, correcting for pre-Phase-A and pre-Phase-C stored states.

    Args:
        asos_readings: ASOS readings from midnight up to eval_hour cutoff.
        nwp_curve:     Blended NWP hourly temperature curve (ET-indexed, °F).
        eval_hour:     ET hour of the replay evaluation point.
        initial_bias:  Starting bias estimate (°F).  Pass yesterday's replayed
                       bias (decayed overnight) for multi-day chaining.

    Returns:
        Kalman bias estimate (°F) at eval_hour using current filter logic.
        Returns ``initial_bias`` if no ASOS readings or NWP curve are available.
    """
    from collections import defaultdict

    from kalshi_weather_trader.quant.kalman_filter import KalmanFilter

    if not asos_readings or not nwp_curve:
        return initial_bias

    # Start at midnight with warm-started (or cold-started) bias
    kf = KalmanFilter(
        initial_dt=0.0,
        initial_bias=initial_bias,
        nwp_current_hour=nwp_curve[0],
    )

    # Sort readings and group by ET hour
    def _obs_utc(r: ASOSReadingDocument) -> datetime:
        obs = r.observation_time_utc
        return obs if obs.tzinfo is not None else obs.replace(tzinfo=timezone.utc)

    by_hour: dict = defaultdict(list)
    for r in sorted(asos_readings, key=_obs_utc):
        hour_et = _obs_utc(r).astimezone(_EASTERN).hour
        by_hour[hour_et].append(r)

    # Update with any midnight-hour ASOS readings before the first predict step
    for r in by_hour.get(0, []):
        kf.update(asos_temp=r.temperature_f)

    # Walk hour by hour up to eval_hour: predict with NWP then update with ASOS
    for h in range(1, eval_hour + 1):
        nwp_h = nwp_curve[h] if h < len(nwp_curve) else None
        kf.predict(nwp_at_current_hour=nwp_h)
        for r in by_hour.get(h, []):
            kf.update(asos_temp=r.temperature_f)

    return kf.bias


def _apply_scenario_overrides(params, scenario, eval_hour: int) -> None:
    """Apply Scenario parameter overrides to an MCParams object in-place.

    The override logic follows the spec: None = "use historical calibrated",
    a value = "force this value regardless of what the DB says".

    Args:
        params:    MCParams built from build_mc_params_historical().
        scenario:  Scenario with override fields.
        eval_hour: ET hour of the replay (used for AM/PM drift selection).
    """
    from kalshi_weather_trader.backtesting.scenarios import Scenario

    # --- n_paths -------------------------------------------------------
    params.n_paths = scenario.n_paths

    # --- sigma ---------------------------------------------------------
    if not scenario.use_time_varying_sigma:
        params.sigma_by_block = None          # collapse to scalar
    if scenario.sigma_by_block_override is not None:
        params.sigma_by_block = scenario.sigma_by_block_override
    if scenario.sigma_override is not None:
        params.sigma = scenario.sigma_override
        params.sigma_by_block = None          # flat override wins over block

    # --- theta ---------------------------------------------------------
    if not scenario.use_time_varying_theta:
        params.theta_am = None
        params.theta_pm = None
    if scenario.theta_am_override is not None:
        params.theta_am = scenario.theta_am_override
    if scenario.theta_pm_override is not None:
        params.theta_pm = scenario.theta_pm_override
    if scenario.theta_override is not None:
        params.theta = scenario.theta_override
        params.theta_am = None
        params.theta_pm = None

    # --- sigma cap -----------------------------------------------------
    if scenario.ou_max_stationary_std_override is not None:
        params.ou_max_stationary_std = scenario.ou_max_stationary_std_override

    # --- persistence offset --------------------------------------------
    if not scenario.use_persistence_offset:
        params.persistence_filter_offset = 0.0
    elif scenario.persistence_filter_offset_override is not None:
        params.persistence_filter_offset = scenario.persistence_filter_offset_override

    # --- Kalman bias ---------------------------------------------------
    if scenario.kalman_bias_override is not None:
        params.bias = scenario.kalman_bias_override

    # --- drift ---------------------------------------------------------
    params.use_drift_in_attractor = scenario.use_drift_in_attractor
    if scenario.drift_am_override is not None and eval_hour < 12:
        params.drift_adj = scenario.drift_am_override
    elif scenario.drift_pm_override is not None and eval_hour >= 12:
        params.drift_adj = scenario.drift_pm_override

    # --- anchor offset -------------------------------------------------
    if not scenario.use_anchor_offset:
        params.anchor_weight_multiplier = 0.0
    else:
        params.anchor_weight_multiplier = scenario.anchor_weight_multiplier

    # --- cloud cover ---------------------------------------------------
    if not scenario.use_cloud_cover_adjustment:
        params.mean_cloudcover_10_16 = 50.0     # neutral: no sigma scaling

    # --- ensemble spread -----------------------------------------------
    if not scenario.use_ensemble_spread_adjustment:
        params.ensemble_spread = 0.0


class ParameterizedReplayEngine:
    """Replay engine for the Model Lab Scenario system.

    Applies Scenario parameter overrides to the historically-calibrated MCParams
    for each (date, eval_hour) pair, runs the MC, and scores against the NWS
    actual settlement.  Uses ReplayDataCache to avoid per-hour DB round-trips.

    When scenario.replay_kalman_bias is True, the engine re-runs the current
    Kalman filter logic over historical ASOS readings for each date rather than
    using the stored kalman_bias_estimate.  Bias is chained across days using an
    overnight decay (settings.kalman_bias_decay ^ 8 hours) so that multi-day
    NWP error accumulates in B just as it does in production.

    Example::

        from datetime import date
        from kalshi_weather_trader.backtesting.replay_engine import ParameterizedReplayEngine
        from kalshi_weather_trader.backtesting.scenarios import preset_production

        engine = ParameterizedReplayEngine()
        results = engine.replay_scenario(
            scenario=preset_production(),
            start_date=date(2026, 3, 10),
            end_date=date(2026, 3, 30),
        )
    """

    def __init__(self) -> None:
        # Keyed by date; holds the replayed bias at the last eval_hour of that
        # date so replay_scenario can warm-start the next day's filter.
        self._kalman_chain: dict[date, float] = {}

    def replay_single(
        self,
        target_date: date,
        eval_hour: int,
        scenario,
        cache,
    ) -> Optional[ParameterizedReplayResult]:
        """Replay one (date, eval_hour) pair under the given scenario.

        Args:
            target_date: Historical trading date.
            eval_hour:   ET hour to anchor the simulation at.
            scenario:    Scenario with parameter overrides.
            cache:       ReplayDataCache with pre-loaded ASOS/state/market data.

        Returns:
            ParameterizedReplayResult, or None if insufficient data exists
            (no ASOS readings, no settled outcome, etc.).

        Raises:
            Nothing — errors are logged and None is returned.
        """
        try:
            return self._replay_single_inner(target_date, eval_hour, scenario, cache)
        except Exception as exc:
            logger.warning(
                "parameterized_replay.hour_failed",
                target_date=str(target_date),
                eval_hour=eval_hour,
                error=str(exc),
            )
            return None

    def _replay_single_inner(
        self,
        target_date: date,
        eval_hour: int,
        scenario,
        cache,
    ) -> Optional[ParameterizedReplayResult]:
        # ------------------------------------------------------------------
        # 1. Compute cutoff UTC for this eval_hour
        # ------------------------------------------------------------------
        et_dt = _EASTERN.localize(
            datetime.combine(target_date, time(eval_hour, 0, 0))
        )
        cutoff_utc = et_dt.astimezone(timezone.utc)

        # ------------------------------------------------------------------
        # 2. Pull data from cache
        # ------------------------------------------------------------------
        past_asos = cache.get_asos_up_to(target_date, cutoff_utc)
        if not past_asos:
            return None

        state = cache.get_state(target_date)
        market = cache.get_market(target_date)

        if market is None or market.final_official_high is None:
            return None
        actual_high = float(market.final_official_high)

        # ------------------------------------------------------------------
        # 3. T0, hard_floor, NWP
        # ------------------------------------------------------------------
        asos_at_hour = min(
            past_asos,
            key=lambda r: abs(
                (r.observation_time_utc.replace(tzinfo=timezone.utc)
                 if r.observation_time_utc.tzinfo is None
                 else r.observation_time_utc) - cutoff_utc
            ),
        )
        hard_floor = max(r.temperature_f for r in past_asos)

        nwp_forecasts = get_nwp_forecasts_before_utc(target_date, cutoff_utc)
        model_weights = scenario.model_weights_override or (state.model_weights if state is not None else None)
        blended_curve = _blend_nwp(nwp_forecasts, model_weights)
        nwp_predicted_high = (
            max(blended_curve) if blended_curve else asos_at_hour.temperature_f
        )
        blended_cc = _blend_cloudcover(nwp_forecasts)
        blended_spread = _blend_ensemble_spread(nwp_forecasts)

        # ------------------------------------------------------------------
        # 4. Build base MCParams from historical calibration
        # ------------------------------------------------------------------
        params = build_mc_params_historical(
            past_date=target_date,
            hour_et=eval_hour,
            state=state,
            asos_at_hour=asos_at_hour,
            hard_floor=hard_floor,
            nwp_curve=blended_curve,
            mean_cloudcover=blended_cc,
            ensemble_spread=blended_spread,
        )

        # ------------------------------------------------------------------
        # 5. Kalman replay bias (optional) — replaces stored bias with a
        #    fresh estimate produced by running the current filter logic over
        #    the historical ASOS readings.  Must happen before scenario
        #    overrides so kalman_bias_override can still take precedence.
        #
        #    Warm-start chaining: if the previous calendar day was already
        #    replayed, carry that day's bias forward (decayed overnight) so
        #    multi-day NWP errors accumulate in B just as in production.
        # ------------------------------------------------------------------
        if scenario.replay_kalman_bias:
            from kalshi_weather_trader.config.settings import settings as _settings
            from datetime import timedelta as _td
            prev_date = target_date - _td(days=1)
            prev_bias = self._kalman_chain.get(prev_date, 0.0)
            warm_start = prev_bias * (_settings.kalman_bias_decay ** _OVERNIGHT_DECAY_HOURS)
            replayed_bias = _replay_kalman_bias_intraday(
                past_asos, blended_curve, eval_hour, initial_bias=warm_start
            )
            params.bias = replayed_bias
            # Update chain so later eval_hours and the next day both see this value
            self._kalman_chain[target_date] = replayed_bias

        # ------------------------------------------------------------------
        # 6. Apply scenario overrides
        # ------------------------------------------------------------------
        _apply_scenario_overrides(params, scenario, eval_hour)

        # ------------------------------------------------------------------
        # 7. Determine strikes
        # ------------------------------------------------------------------
        if (
            market is not None
            and hasattr(market, "floor_strike")
            and getattr(market, "floor_strike", None) is not None
            and hasattr(market, "cap_strike")
            and getattr(market, "cap_strike", None) is not None
        ):
            strikes = [
                float(s)
                for s in range(int(market.floor_strike), int(market.cap_strike) + 1)
            ]
        else:
            center = int(round(nwp_predicted_high))
            strikes = [float(s) for s in range(center - 8, center + 9)]

        # ------------------------------------------------------------------
        # 8. Run MC
        # ------------------------------------------------------------------
        mc_result = price_full_distribution(
            params=params,
            strikes=strikes,
            target_date=target_date,
            seed=scenario.random_seed,
        )

        # ------------------------------------------------------------------
        # 9. Build diagnostic summary fields
        # ------------------------------------------------------------------
        effective_floor = params.hard_floor + params.persistence_filter_offset

        if params.sigma_by_block:
            sigma_used = dict(params.sigma_by_block)
        else:
            sigma_used = {"scalar": params.sigma}

        if params.theta_am is not None or params.theta_pm is not None:
            theta_used = {
                k: v for k, v in {"am": params.theta_am, "pm": params.theta_pm}.items()
                if v is not None
            }
            if not theta_used:
                theta_used = {"scalar": params.theta}
        else:
            theta_used = {"scalar": params.theta}

        bias_used = params.bias
        drift_used = params.drift_adj if params.use_drift_in_attractor else 0.0
        attractor_peak = (
            max(params.nwp_curve) + bias_used if params.nwp_curve else params.T0 + bias_used
        )

        percentiles = {
            10: float(mc_result.percentile_10),
            25: float(mc_result.percentile_25),
            50: float(mc_result.percentile_50),
            75: float(mc_result.percentile_75),
            90: float(mc_result.percentile_90),
        }

        # ------------------------------------------------------------------
        # 9. Score
        # ------------------------------------------------------------------
        brier_components: dict = {}
        for strike, prob in mc_result.probabilities.items():
            outcome = 1.0 if actual_high >= strike else 0.0
            brier_components[str(strike)] = float((prob - outcome) ** 2)

        market_probs = {str(k): float(v) for k, v in mc_result.probabilities.items()}

        return ParameterizedReplayResult(
            target_date=target_date,
            eval_hour=eval_hour,
            T0=float(params.T0),
            hard_floor=float(hard_floor),
            effective_floor=float(effective_floor),
            sigma_used=sigma_used,
            theta_used=theta_used,
            bias_used=float(bias_used),
            drift_used=float(drift_used),
            nwp_predicted_high=float(nwp_predicted_high),
            attractor_peak=float(attractor_peak),
            mean_max=float(mc_result.mean_max),
            std_max=float(mc_result.std_max),
            percentiles=percentiles,
            strike_probs=dict(mc_result.probabilities),
            market_probs=market_probs,
            actual_high=actual_high,
            prediction_error=float(mc_result.mean_max - actual_high),
            brier_components=brier_components,
        )

    def replay_scenario(
        self,
        scenario,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        max_days: Optional[int] = None,
    ) -> list[ParameterizedReplayResult]:
        """Replay all settled dates in the given range under the scenario.

        A settled date is one where market.cli_settlement_confirmed=True and
        market.final_official_high is not None.

        Args:
            scenario:   Scenario with parameter overrides and eval_hours.
            start_date: First date to consider (inclusive). Defaults to 90 days ago.
            end_date:   Last date to consider (inclusive). Defaults to yesterday.
            max_days:   If set, use only the most recent N settled dates.

        Returns:
            list[ParameterizedReplayResult] sorted by (target_date, eval_hour).
        """
        from datetime import timedelta

        today = datetime.now(timezone.utc).astimezone(_EASTERN).date()
        if end_date is None:
            end_date = today - timedelta(days=1)
        if start_date is None:
            start_date = end_date - timedelta(days=90)

        # ------------------------------------------------------------------
        # Collect settled dates in range
        # ------------------------------------------------------------------
        settled_dates: list[date] = []
        current = start_date
        while current <= end_date:
            try:
                mkt = get_market(current)
                if (
                    mkt is not None
                    and mkt.cli_settlement_confirmed
                    and mkt.final_official_high is not None
                ):
                    settled_dates.append(current)
            except Exception as exc:
                logger.warning(
                    "parameterized_replay.market_check_failed",
                    date=str(current),
                    error=str(exc),
                )
            current += timedelta(days=1)

        if max_days is not None:
            settled_dates = settled_dates[-max_days:]

        if not settled_dates:
            logger.warning(
                "parameterized_replay.no_settled_dates",
                start_date=str(start_date),
                end_date=str(end_date),
            )
            return []

        logger.info(
            "parameterized_replay.starting",
            n_dates=len(settled_dates),
            scenario=scenario.name,
            eval_hours=scenario.eval_hours,
        )

        # ------------------------------------------------------------------
        # Bulk-load ASOS/state/market data
        # ------------------------------------------------------------------
        from kalshi_weather_trader.backtesting.scenarios import ReplayDataCache
        cache = ReplayDataCache.load(settled_dates)

        # Reset Kalman chain so a fresh replay_scenario() call starts clean
        self._kalman_chain.clear()

        # ------------------------------------------------------------------
        # Run replay for each (date, hour)
        # ------------------------------------------------------------------
        results: list[ParameterizedReplayResult] = []
        for d in settled_dates:
            for hour in scenario.eval_hours:
                result = self.replay_single(d, hour, scenario, cache)
                if result is not None:
                    results.append(result)

        results.sort(key=lambda r: (r.target_date, r.eval_hour))
        logger.info(
            "parameterized_replay.done",
            n_results=len(results),
            n_dates=len(settled_dates),
            scenario=scenario.name,
        )
        return results

    def compare_scenarios(
        self,
        scenario_a,
        scenario_b,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        max_days: Optional[int] = None,
        n_bootstrap: int = 1000,
    ) -> ComparisonResult:
        """Run two scenarios and return a side-by-side comparison with bootstrap.

        Replays both scenarios over the same settled-date range, computes aggregate
        metrics for each, then runs a paired bootstrap to test whether the Brier
        score difference is statistically significant.

        Args:
            scenario_a:  First scenario (baseline — typically production).
            scenario_b:  Second scenario to compare against A.
            start_date:  Date range start (inclusive). Defaults to 90 days ago.
            end_date:    Date range end (inclusive). Defaults to yesterday.
            max_days:    Restrict to most recent N settled dates.
            n_bootstrap: Bootstrap iterations for significance test.

        Returns:
            ComparisonResult with per-scenario metrics and BootstrapResult.
        """
        from kalshi_weather_trader.backtesting.metrics import (
            compute_aggregate_metrics,
            compute_paired_bootstrap,
        )

        a_results = self.replay_scenario(scenario_a, start_date, end_date, max_days)
        b_results = self.replay_scenario(scenario_b, start_date, end_date, max_days)

        a_metrics = compute_aggregate_metrics(a_results)
        b_metrics = compute_aggregate_metrics(b_results)

        bootstrap = None
        if a_results and b_results:
            try:
                bootstrap = compute_paired_bootstrap(
                    a_results, b_results, n_bootstrap=n_bootstrap
                )
            except ValueError as exc:
                logger.warning("compare_scenarios.bootstrap_failed", error=str(exc))

        dates_a = {r.target_date for r in a_results}
        dates_b = {r.target_date for r in b_results}
        common = sorted(dates_a & dates_b)

        return ComparisonResult(
            scenario_a_name=scenario_a.name,
            scenario_b_name=scenario_b.name,
            a_metrics=a_metrics,
            b_metrics=b_metrics,
            bootstrap=bootstrap,
            a_results=a_results,
            b_results=b_results,
            common_dates=common,
        )
