"""
Model calibration for the Kalshi weather trading system.

Runs nightly (00:05 Eastern) and on demand via the Streamlit UI.

Four calibration routines:
1. ``calibrate_model_weights``  — Brier score per NWP model → softmax weights
2. ``calibrate_intraday_drift`` — AM/PM prediction error → drift adjustments
3. ``calibrate_sigma``          — AR volatility from ASOS 5-min diffs
4. ``calibrate_theta``          — AR(1) fit on NWP departure → OU mean-reversion

Also provides ``record_snapshot()`` which collects current state, runs Monte
Carlo, and writes an ``IntradaySnapshotDocument``.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pytz
import structlog

from kalshi_weather_trader.config.settings import get_target_date, settings
from kalshi_weather_trader.db import db_manager

logger = structlog.get_logger(__name__)

_EASTERN = pytz.timezone("America/New_York")


# ---------------------------------------------------------------------------
# Helper: Brier score for one model
# ---------------------------------------------------------------------------


def _brier_score_for_model(
    model_name: str,
    lookback_days: int,
) -> Optional[float]:
    """Compute the average Brier score for one NWP model over recent days.

    Uses the proper probabilistic Brier score tied to Kalshi contract resolution:
    outcome = 1 if official_high >= kalshi_strike (the observable event that
    determines contract settlement), and P = N(predicted_high, sigma_model) CDF
    at the strike.  sigma_model is the empirical RMSE of the model's daily high
    forecasts over the same lookback window, so accuracy is model-specific.

    Falls back to the legacy approach (outcome = over/under predicted_high,
    scale = 2.0°F) when fewer than 2 settled dates have a known kalshi_strike.

    Args:
        model_name:    'HRRR', 'GFS', or 'ECMWF'.
        lookback_days: Number of past days to evaluate.

    Returns:
        Mean Brier score (lower is better), or None if insufficient data.

    Raises:
        Nothing — errors are logged.
    """
    from scipy.stats import norm

    from kalshi_weather_trader.db.db_manager import get_asos_readings_for_date
    from kalshi_weather_trader.quant.mc_params_builder import build_mc_params_historical
    from kalshi_weather_trader.quant.monte_carlo import _interpolate_cdf, price_full_distribution

    today = date.today()

    # Collect per-date data in one pass so we can compute sigma_model before
    # scoring (Pass 1), then score with that sigma (Pass 2).
    # Each entry: (past_date, predicted_high, official_high, kalshi_strike_or_None, nwp_doc_or_None)
    date_records: list[tuple[date, float, float, Optional[float], object]] = []

    for d in range(1, lookback_days + 1):
        past_date = today - timedelta(days=d)
        try:
            market = db_manager.get_market(past_date)
            if market is None or market.final_official_high is None:
                continue

            # Use the first fetch at or after 10 AM ET — the morning-of prediction
            # that would have been available when Kalshi trading was live.
            # get_latest_nwp_forecasts would use the last fetch of the day, which
            # benefits from intraday model revisions and introduces lookback bias.
            forecasts = db_manager.get_morning_nwp_forecasts(past_date)
            if model_name not in forecasts:
                continue

            predicted_high = forecasts[model_name].predicted_daily_high
            official_high = market.final_official_high
            # kalshi_strike lives in intraday_snapshots, not the markets row.
            # Use the most recent snapshot for this date that has a strike recorded.
            snap_list = db_manager.get_snapshots_for_date(past_date)
            strike_snaps = [s for s in snap_list if s.kalshi_strike is not None]
            strike = float(strike_snaps[-1].kalshi_strike) if strike_snaps else None
            date_records.append((past_date, predicted_high, official_high, strike, forecasts[model_name]))

        except Exception as exc:
            logger.warning(
                "calibrator.brier.date_failed",
                model=model_name,
                date=str(past_date),
                error=str(exc),
            )
            continue

    if len(date_records) < 2:
        return None

    # Pass 1 — compute model-specific RMSE across all available dates.
    errors_sq = [(pred - obs) ** 2 for _, pred, obs, _, _ in date_records]
    sigma_model = max(0.5, float(np.sqrt(np.mean(errors_sq))))
    logger.debug(
        "calibrator.brier.sigma_model",
        model=model_name,
        sigma_model=round(sigma_model, 3),
        n=len(errors_sq),
    )

    # Determine whether we have enough settled markets with known strikes for
    # the proper Brier formulation.
    dates_with_strike = [
        (past_date, pred, obs, strike, nwp_doc)
        for past_date, pred, obs, strike, nwp_doc in date_records
        if strike is not None
    ]
    use_proper = len(dates_with_strike) >= 2

    if not use_proper:
        logger.warning(
            "calibrator.brier.fallback_legacy",
            model=model_name,
            reason="fewer than 2 dates have kalshi_strike",
            n_with_strike=len(dates_with_strike),
        )

    scores: list[float] = []

    _BRIER_EVAL_HOUR = 10  # Reconstruct state at 10 AM ET

    if use_proper:
        # Pass 2 — proper Brier score: did the daily max clear the Kalshi strike?
        for past_date, predicted_high, official_high, strike, nwp_doc in dates_with_strike:
            p_yes: Optional[float] = None
            # ---- Try MC-based probability ----
            try:
                nwp_curve = nwp_doc.hourly_temps if nwp_doc is not None else []
                asos_readings = get_asos_readings_for_date(past_date)
                if asos_readings:
                    hour_10_utc = _EASTERN.localize(
                        datetime(past_date.year, past_date.month, past_date.day, _BRIER_EVAL_HOUR)
                    ).astimezone(timezone.utc)
                    before_10 = [r for r in asos_readings if r.observation_time_utc <= hour_10_utc]
                    if before_10:
                        asos_at_10 = min(
                            before_10,
                            key=lambda r: abs((r.observation_time_utc - hour_10_utc).total_seconds()),
                        )
                        hard_floor_at_10 = max(r.temperature_f for r in before_10)
                        past_state = db_manager.get_system_state(past_date)
                        mc_params = build_mc_params_historical(
                            past_date=past_date,
                            hour_et=_BRIER_EVAL_HOUR,
                            state=past_state,
                            asos_at_hour=asos_at_10,
                            hard_floor=hard_floor_at_10,
                            nwp_curve=nwp_curve,
                        )
                        # P(integer max >= strike) ≈ P(continuous max >= strike - 0.5)
                        boundary = strike - 0.5
                        mc_result = price_full_distribution(mc_params, [boundary], past_date)
                        p_yes = _interpolate_cdf(mc_result.probabilities, boundary)
            except Exception as exc:
                logger.debug("calibrator.brier.mc_failed", date=str(past_date), error=str(exc))

            if p_yes is None:
                # Gaussian fallback when historical ASOS data is unavailable
                p_yes = 1.0 - norm.cdf(strike, loc=predicted_high, scale=sigma_model)

            outcome = 1.0 if official_high >= strike else 0.0
            scores.append((p_yes - outcome) ** 2)
    else:
        # Legacy fallback: outcome = did official_high exceed predicted_high?
        for _, predicted_high, official_high, _, _ in date_records:
            p_yes = 1.0 - norm.cdf(official_high, loc=predicted_high, scale=2.0)
            outcome = 1.0 if official_high >= predicted_high else 0.0
            scores.append((p_yes - outcome) ** 2)

    if len(scores) < 2:
        return None

    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# 1. Model weight calibration
# ---------------------------------------------------------------------------


def calibrate_model_weights(
    target_date: Optional[date] = None,
    lookback_days: int = 14,
) -> dict[str, float]:
    """Calibrate NWP model weights using Brier scores over recent history.

    Computes Brier score for each model, inverts them (lower Brier → higher
    weight), applies softmax normalisation, and persists the result to
    system_state.

    Args:
        target_date:   Active trading date. Defaults to today's target.
        lookback_days: Days of history to evaluate. Defaults to 14.

    Returns:
        Dict mapping model_name → weight (sums to 1.0).

    Raises:
        Nothing — falls back to equal weights on any error.
    """
    if target_date is None:
        target_date = get_target_date()

    models = ["HRRR", "GFS", "ECMWF"]
    equal_weights = {"HRRR": round(1.0 / 3, 6), "GFS": round(1.0 / 3, 6), "ECMWF": round(1.0 / 3, 6)}
    default_weights = {"HRRR": 0.5, "GFS": 0.3, "ECMWF": 0.2}

    # Guard: require ≥ 10 settled dates with morning NWP forecasts before using
    # Brier-weighted model weights.  Model weights should reflect which NWP
    # model predicts the actual daily high most accurately — Kalshi strike data
    # is not required for that comparison.  A date qualifies if it has both a
    # confirmed final_official_high and at least one model's morning NWP
    # forecast available (the same data _brier_score_for_model() needs).
    # Threshold = 10: enough signal to distinguish models while still allowing
    # calibration in the first few weeks of operation.
    n_qualifying = 0
    for _d in range(1, lookback_days + 1):
        _past = date.today() - timedelta(days=_d)
        try:
            _mkt = db_manager.get_market(_past)
            if _mkt is None or _mkt.final_official_high is None:
                continue
            _forecasts = db_manager.get_morning_nwp_forecasts(_past)
            if _forecasts:
                n_qualifying += 1
        except Exception:
            continue
    if n_qualifying < 10:
        logger.info(
            "calibrator.weights.insufficient_data",
            n_qualifying=n_qualifying,
            required=10,
            using="equal_weights",
        )
        try:
            state = db_manager.get_system_state(target_date)
            if state:
                state.model_weights = equal_weights
                state.last_calibrated_utc = datetime.now(timezone.utc)
                state.last_updated_utc = datetime.now(timezone.utc)
                db_manager.upsert_system_state(state)
        except Exception as exc:
            logger.error("calibrator.weights.persist_failed", error=str(exc))
        return equal_weights

    brier_scores: dict[str, float] = {}
    for model in models:
        score = _brier_score_for_model(model, lookback_days)
        if score is not None:
            brier_scores[model] = score
        else:
            logger.warning("calibrator.weights.no_brier", model=model)

    if len(brier_scores) < 2:
        logger.warning(
            "calibrator.weights.insufficient_data",
            available=list(brier_scores.keys()),
        )
        return default_weights

    # Invert scores (lower Brier is better → higher inverse)
    inv_scores = {m: 1.0 / (s + 1e-8) for m, s in brier_scores.items()}

    # Softmax over inverse scores
    max_inv = max(inv_scores.values())
    exp_scores = {m: math.exp(v - max_inv) for m, v in inv_scores.items()}
    total = sum(exp_scores.values())
    weights = {m: round(v / total, 6) for m, v in exp_scores.items()}

    # Any missing model gets weight 0
    for model in models:
        if model not in weights:
            weights[model] = 0.0

    # Re-normalise to ensure sum = 1.0
    total_w = sum(weights.values())
    if total_w > 0:
        weights = {m: round(w / total_w, 6) for m, w in weights.items()}

    logger.info(
        "calibrator.weights.done",
        weights=weights,
        brier_scores={m: round(s, 6) for m, s in brier_scores.items()},
    )

    # Persist to system_state
    try:
        state = db_manager.get_system_state(target_date)
        if state:
            state.model_weights = weights
            state.last_calibrated_utc = datetime.now(timezone.utc)
            state.last_updated_utc = datetime.now(timezone.utc)
            db_manager.upsert_system_state(state)
    except Exception as exc:
        logger.error("calibrator.weights.persist_failed", error=str(exc))

    return weights


# ---------------------------------------------------------------------------
# 2. Intraday drift calibration
# ---------------------------------------------------------------------------


def calibrate_intraday_drift(
    target_date: Optional[date] = None,
    lookback_days: int = 14,
) -> tuple[float, float]:
    """Calibrate AM/PM drift adjustments from recent intraday snapshots.

    Pools snapshots across the past ``lookback_days`` settled trading days,
    groups them by morning (< 12:00 ET) and afternoon (>= 12:00 ET), and
    computes exponentially weighted mean(blended_predicted_high - final_official_high)
    as the drift correction.  Recent days are up-weighted by exp(-d/tau) where
    tau = settings.calibration_decay_tau_days.

    Args:
        target_date:   Active trading date. Defaults to today's target.
        lookback_days: Number of past days to include. Defaults to 14.

    Returns:
        Tuple of (morning_drift_adj, afternoon_drift_adj) in °F.
        Both default to 0.0 if fewer than 2 settled days exist.

    Raises:
        Nothing — errors are logged.
    """
    if target_date is None:
        target_date = get_target_date()

    tau = settings.calibration_decay_tau_days

    # Track (error, day_weight) pairs so we can compute weighted means.
    morning_errors: list[tuple[float, float]] = []
    afternoon_errors: list[tuple[float, float]] = []
    days_used: int = 0

    for d in range(1, lookback_days + 1):
        past_date = target_date - timedelta(days=d)
        day_weight = math.exp(-d / tau)
        try:
            market = db_manager.get_market(past_date)
            if market is None or market.final_official_high is None:
                continue

            snapshots = db_manager.get_snapshots_for_date(past_date)
            if not snapshots:
                continue

            official_high = market.final_official_high
            days_used += 1

            for snap in snapshots:
                try:
                    hour_et = int(snap.snapshot_time_eastern.split(":")[0])
                except (ValueError, AttributeError):
                    continue

                error = snap.blended_predicted_high - official_high
                if hour_et < 12:
                    morning_errors.append((error, day_weight))
                else:
                    afternoon_errors.append((error, day_weight))

        except Exception as exc:
            logger.warning(
                "calibrator.drift.date_failed",
                date=str(past_date),
                error=str(exc),
            )
            continue

    if days_used < 2:
        logger.warning(
            "calibrator.drift.insufficient_data",
            days_used=days_used,
            lookback_days=lookback_days,
        )
        return 0.0, 0.0

    def _weighted_mean_error(pairs: list[tuple[float, float]]) -> float:
        if not pairs:
            return 0.0
        total_w = sum(w for _, w in pairs)
        if total_w < 1e-10:
            return 0.0
        return sum(e * w for e, w in pairs) / total_w

    morning_adj = round(-_weighted_mean_error(morning_errors), 3)
    afternoon_adj = round(-_weighted_mean_error(afternoon_errors), 3)

    logger.info(
        "calibrator.drift.done",
        morning_adj=morning_adj,
        afternoon_adj=afternoon_adj,
        morning_n=len(morning_errors),
        afternoon_n=len(afternoon_errors),
        days_used=days_used,
        decay_tau_days=tau,
    )

    # Persist
    try:
        state = db_manager.get_system_state(target_date)
        if state:
            state.morning_drift_adjustment = morning_adj
            state.afternoon_drift_adjustment = afternoon_adj
            state.last_updated_utc = datetime.now(timezone.utc)
            db_manager.upsert_system_state(state)
    except Exception as exc:
        logger.error("calibrator.drift.persist_failed", error=str(exc))

    return morning_adj, afternoon_adj


# ---------------------------------------------------------------------------
# 3. Sigma calibration
# ---------------------------------------------------------------------------


def calibrate_sigma(
    target_date: Optional[date] = None,
    lookback_days: Optional[int] = None,
) -> float:
    """Calibrate the OU sigma (diffusion) from recent ASOS 5-minute diffs.

    Args:
        target_date:   Active trading date. Defaults to today's target.
        lookback_days: Days of ASOS history to use. Defaults to
                       settings.calibration_lookback_days (30).

    Returns:
        Calibrated sigma in °F/sqrt-hour.

    Raises:
        Nothing — returns settings.ou_sigma on failure.
    """
    from kalshi_weather_trader.ingestion.asos_fetcher import fetch_last_n_hours
    from kalshi_weather_trader.ingestion.nwp_fetcher import get_nwp_curve
    from kalshi_weather_trader.quant.monte_carlo import estimate_sigma_from_historical

    if target_date is None:
        target_date = get_target_date()
    if lookback_days is None:
        lookback_days = settings.calibration_lookback_days

    try:
        readings = fetch_last_n_hours(hours=lookback_days * 24)
        if len(readings) < 12:
            logger.warning("calibrator.sigma.insufficient_readings", n=len(readings))
            return settings.ou_sigma

        nwp_curves: dict[date, list[float]] = {}
        for d in {r.observation_time_utc.astimezone(_EASTERN).date() for r in readings}:
            try:
                curve = get_nwp_curve(d)
                if len(curve) >= 2:
                    nwp_curves[d] = curve
            except Exception as exc:
                logger.warning("calibrator.sigma.nwp_curve_fetch_failed", date=str(d), error=str(exc))

        sigma, sigma_by_block = estimate_sigma_from_historical(
            readings,
            nwp_curves=nwp_curves,
            decay_tau_days=settings.calibration_decay_tau_days,
        )
        logger.info(
            "calibrator.sigma.done",
            sigma=sigma,
            sigma_by_block=sigma_by_block,
            n_readings=len(readings),
            n_nwp_dates=len(nwp_curves),
        )

        # Persist
        state = db_manager.get_system_state(target_date)
        if state:
            state.sigma_volatility = sigma
            state.sigma_by_block = sigma_by_block if sigma_by_block else None
            state.last_updated_utc = datetime.now(timezone.utc)
            db_manager.upsert_system_state(state)

        return sigma
    except Exception as exc:
        logger.error("calibrator.sigma.failed", error=str(exc))
        return settings.ou_sigma


# ---------------------------------------------------------------------------
# 3b. Persistence filter offset calibration
# ---------------------------------------------------------------------------


def calibrate_persistence_offset(
    target_date: Optional[date] = None,
    lookback_days: int = 30,
) -> float:
    """Calibrate the ASOS-to-NWS daily max gap (persistence_filter_offset).

    The ASOS 0.5°C persistence filter causes the NWS-reported daily maximum
    to exceed the highest ASOS tabular reading by a systematic positive offset
    (~0.75°F empirically from 8 settled KBOS dates).  This function estimates
    that offset from settled history and persists it in system_state so
    run_simulation() can apply it when initialising paths_max.

    Algorithm:
        For each settled date in the lookback window:
            gap = final_official_high - max(temperature_f, max6h_f) across all ASOS readings
        offset = mean(all gaps, including zeros and negative)
        Floor at 0.0 (never apply a negative offset) and clamp to [0.0, 1.5].
        Require ≥ 5 qualifying dates.

    Phase A change: zeros are now included in the mean (previously excluded).
    Excluding zeros (days where ASOS matched NWS exactly) biased the estimate
    upward by 33% (0.75°F → 1.0°F with 8 dates). The [0.0, 0.5] clamp has
    also been raised to [0.0, 1.5] to accommodate empirical values near 0.75°F.

    Args:
        target_date:   Active trading date. Defaults to today's target.
        lookback_days: Days of settled history to use. Defaults to 30.

    Returns:
        Calibrated offset in °F, clamped to [0.0, 1.5].
        Returns settings.persistence_filter_offset if insufficient data.

    Raises:
        Nothing — returns default on any failure.
    """
    if target_date is None:
        target_date = get_target_date()

    try:
        gaps: list[float] = []

        for d in range(1, lookback_days + 1):
            past_date = target_date - timedelta(days=d)
            try:
                market = db_manager.get_market(past_date)
                if market is None or market.final_official_high is None:
                    continue
                day_readings = db_manager.get_asos_readings_for_date(past_date)
                if not day_readings:
                    continue
                asos_max = max(
                    max(r.temperature_f for r in day_readings),
                    max(
                        (r.max6h_f for r in day_readings if r.max6h_f is not None),
                        default=-999.0,
                    ),
                )
                gap = float(market.final_official_high) - asos_max
                gaps.append(gap)  # include all gaps: positive, zero, and negative
            except Exception as day_exc:
                logger.debug("calibrator.persistence_offset.day_failed", date=str(past_date), error=str(day_exc))

        if len(gaps) < 5:
            logger.warning(
                "calibrator.persistence_offset.insufficient_data",
                n_qualifying=len(gaps),
                required=5,
            )
            return settings.persistence_filter_offset

        offset = max(0.0, min(1.5, round(float(np.mean(gaps)), 3)))
        logger.info(
            "calibrator.persistence_offset.done",
            offset=offset,
            n_dates=len(gaps),
            mean_gap=round(float(np.mean(gaps)), 3),
        )

        # Persist
        state = db_manager.get_system_state(target_date)
        if state:
            state.persistence_filter_offset = offset
            state.last_updated_utc = datetime.now(timezone.utc)
            db_manager.upsert_system_state(state)

        return offset
    except Exception as exc:
        logger.error("calibrator.persistence_offset.failed", error=str(exc))
        return settings.persistence_filter_offset


# ---------------------------------------------------------------------------
# Phase B: ou_max_stationary_std calibration from hourly NWP RMSE
# ---------------------------------------------------------------------------

_MIN_BIAS_DECAY_PAIRS = 30    # AR(1) needs consecutive pairs, not independent dates
_BIAS_DECAY_CLIP_MIN  = 0.85  # half-life ~4.3h — fastest reasonable intraday decay
_BIAS_DECAY_CLIP_MAX  = 1.0   # no decay (random walk) — hard ceiling

_MIN_RMSE_DATES = 10          # guard: need enough dates for a stable RMSE estimate
_HOURLY_RMSE_SAFETY_FACTOR = 1.2  # cap = hourly_rmse × factor
#   1.2 (not 1.0) because hourly RMSE omits the extreme-hour tail that drives
#   the daily maximum; daily-high RMSE is typically 30-50% larger than per-hour
#   RMSE, and the OU process needs to cover those peak-hour deviations too.
_HOURLY_RMSE_MIN_ET_HOUR = 10  # exclude pre-fetch hours (0-9 ET)
#   The morning NWP fetch window is [10 AM, 1 PM) ET.  For hours 0-9, the NWP
#   has already assimilated ASOS observations from those hours, so the "forecast"
#   error for those hours is effectively analysis error (near zero) — not the
#   true out-of-sample forecast uncertainty we want to measure.  Including them
#   would downward-bias the RMSE.
_MAX_ASOS_GAP_MINUTES = 40.0  # consistent with theta calibration's gap tolerance
_RMSE_CAP_MIN = 0.5            # sanity floor (°F)
_RMSE_CAP_MAX = 5.0            # sanity ceiling (°F)


def _blend_morning_forecast_hourly(
    forecasts: dict,  # model_name → NWPForecastDocument
    model_weights: dict[str, float],
) -> list[float]:
    """Blend hourly_temps from morning forecast documents, renormalizing per-hour.

    Mirrors the logic in ``get_nwp_curve()`` but operates on
    ``NWPForecastDocument`` objects already in memory rather than querying the
    DB.  Models with shorter curves are dropped from the blend for hours where
    they have no data, and the remaining weights are renormalized to sum to 1.

    Args:
        forecasts:     Dict mapping model_name → NWPForecastDocument.
        model_weights: Calibrated model weights (from system_state or equal fallback).

    Returns:
        Blended hourly temperature list (°F).  Empty if forecasts is empty.
    """
    if not forecasts:
        return []
    n_hours = max(len(f.hourly_temps) for f in forecasts.values())
    available = {m: model_weights.get(m, 0.0) for m in forecasts}
    curve: list[float] = []
    for h in range(n_hours):
        h_contrib = {
            m: forecasts[m].hourly_temps[h]
            for m in available
            if h < len(forecasts[m].hourly_temps)
        }
        if not h_contrib:
            break
        h_weights = {m: available[m] for m in h_contrib}
        total_w = sum(h_weights.values())
        if total_w <= 0.0:
            # All models have zero weight — fall back to equal weighting.
            h_weights = {m: 1.0 for m in h_contrib}
            total_w = float(len(h_contrib))
        curve.append(sum(w / total_w * h_contrib[m] for m, w in h_weights.items()))
    return curve


def calibrate_ou_max_stationary_std(
    target_date: Optional[date] = None,
    lookback_days: Optional[int] = None,
) -> Optional[float]:
    """Calibrate ou_max_stationary_std from empirical hourly NWP RMSE (Phase B).

    For each past trading date in the lookback window:

    1.  Fetch the morning NWP forecast (first fetch in [10 AM, 1 PM) ET) for
        each model.  This avoids lookback bias: later intraday fetches have
        already assimilated ASOS observations and would understate forecast error.

    2.  Blend ``hourly_temps`` per model using ``model_weights``, renormalizing
        per-hour when some models have shorter curves.

    3.  Fetch ASOS readings for that date.

    4.  For each ET hour >= ``_HOURLY_RMSE_MIN_ET_HOUR`` (10 ET), find the
        nearest ASOS reading within ``_MAX_ASOS_GAP_MINUTES`` (40 min).
        Compute ``error = asos_temp - blended_nwp[hour]``.

    5.  Accumulate squared errors across all valid (date, hour) pairs.

    ``calibrated_cap = sqrt(mean(errors²)) × _HOURLY_RMSE_SAFETY_FACTOR (1.2)``

    Why hours >= 10 only:
        For hours 0–9 the morning NWP has already assimilated those observations;
        the resulting "forecast" error is near-zero analysis error, not genuine
        forecast uncertainty.  Including those hours would downward-bias the RMSE
        and produce a cap that is too tight for the afternoon hours that actually
        matter for same-day trading.

    Why 1.2× (not 1.0×):
        The hourly RMSE is computed across all post-fetch hours, including low-
        variance overnight hours.  The daily maximum error — which the OU process
        needs to cover — is systematically larger (daily high = max of ~14
        afternoon errors, which has heavier tails than a single-hour draw).
        The 1.2× factor partially compensates without being overly conservative.

    CLI settlement is NOT required:
        Hourly NWP-vs-ASOS errors are independent of NWS CLI settlement.  Dropping
        that requirement gives more qualifying dates, especially for recent dates
        where the CLI report may not yet have been confirmed.

    Persists to ``system_state.ou_max_stationary_std_calibrated`` and
    ``system_state.nwp_rmse_n_dates`` (= number of qualifying dates).

    Args:
        target_date:   Active trading date for state persistence.  Defaults to today.
        lookback_days: Days of history to scan.  Defaults to
                       ``settings.calibration_lookback_days`` (30).

    Returns:
        Calibrated cap in °F, clamped to [0.5, 5.0].
        Returns ``None`` if fewer than ``_MIN_RMSE_DATES`` dates have both
        morning NWP data and at least one valid ASOS pair in the eval window.

    Raises:
        Nothing — returns ``None`` on any failure.
    """
    if target_date is None:
        target_date = get_target_date()
    if lookback_days is None:
        lookback_days = settings.calibration_lookback_days

    try:
        # Load model weights for blending; fall back to equal weights.
        state = db_manager.get_system_state(target_date)
        model_weights: dict[str, float] = (
            state.model_weights
            if state and state.model_weights
            else {"HRRR": 1 / 3, "GFS": 1 / 3, "ECMWF": 1 / 3}
        )

        # Flat list of squared errors across all (date, hour) pairs.
        all_sq_errors: list[float] = []
        # Per-model squared errors for diagnostic logging.
        per_model_sq: dict[str, list[float]] = {}
        # Count of dates that contributed at least one valid pair.
        n_dates_with_data = 0

        for d in range(1, lookback_days + 1):
            past_date = target_date - timedelta(days=d)
            try:
                market = db_manager.get_market(past_date)
                if market is None:
                    # Date was not traded — skip.  (Don't require cli_settlement_confirmed
                    # since ASOS-vs-NWP accuracy is independent of NWS CLI availability.)
                    continue

                morning_forecasts = db_manager.get_morning_nwp_forecasts(past_date)
                if not morning_forecasts:
                    continue

                blended_curve = _blend_morning_forecast_hourly(morning_forecasts, model_weights)
                # Need at least one eval hour in the curve.
                if len(blended_curve) <= _HOURLY_RMSE_MIN_ET_HOUR:
                    continue

                asos_readings = db_manager.get_asos_readings_for_date(past_date)
                if not asos_readings:
                    continue

                date_had_valid_pair = False

                for hour_et in range(_HOURLY_RMSE_MIN_ET_HOUR, len(blended_curve)):
                    nwp_temp = blended_curve[hour_et]

                    try:
                        top_of_hour_utc = _EASTERN.localize(
                            datetime(
                                past_date.year, past_date.month, past_date.day,
                                hour_et, 0, 0,
                            )
                        ).astimezone(timezone.utc)
                    except Exception:
                        # DST boundary edge case — skip this hour.
                        continue

                    best = min(
                        asos_readings,
                        key=lambda r, toh=top_of_hour_utc: abs(
                            (r.observation_time_utc - toh).total_seconds()
                        ),
                    )
                    gap_minutes = (
                        abs((best.observation_time_utc - top_of_hour_utc).total_seconds())
                        / 60.0
                    )
                    if gap_minutes > _MAX_ASOS_GAP_MINUTES:
                        continue

                    blended_error = best.temperature_f - nwp_temp
                    all_sq_errors.append(blended_error ** 2)
                    date_had_valid_pair = True

                    # Per-model diagnostic errors (unblended).
                    for model_name, forecast in morning_forecasts.items():
                        if hour_et < len(forecast.hourly_temps):
                            model_err = best.temperature_f - forecast.hourly_temps[hour_et]
                            per_model_sq.setdefault(model_name, []).append(model_err ** 2)

                if date_had_valid_pair:
                    n_dates_with_data += 1

            except Exception as day_exc:
                logger.debug(
                    "calibrator.ou_max_std.day_failed",
                    date=str(past_date),
                    error=str(day_exc),
                )

        if n_dates_with_data < _MIN_RMSE_DATES:
            logger.warning(
                "calibrator.ou_max_std.insufficient_data",
                n_qualifying=n_dates_with_data,
                required=_MIN_RMSE_DATES,
                n_pairs=len(all_sq_errors),
            )
            return None

        hourly_rmse = float(np.sqrt(np.mean(all_sq_errors)))
        calibrated_cap = float(
            np.clip(hourly_rmse * _HOURLY_RMSE_SAFETY_FACTOR, _RMSE_CAP_MIN, _RMSE_CAP_MAX)
        )

        per_model_hourly_rmse = {
            m: round(float(np.sqrt(np.mean(sq_list))), 3)
            for m, sq_list in per_model_sq.items()
            if sq_list
        }
        logger.info(
            "calibrator.ou_max_std.done",
            hourly_rmse=round(hourly_rmse, 3),
            calibrated_cap=round(calibrated_cap, 3),
            per_model_hourly_rmse=per_model_hourly_rmse,
            n_dates=n_dates_with_data,
            n_pairs=len(all_sq_errors),
            safety_factor=_HOURLY_RMSE_SAFETY_FACTOR,
            min_eval_hour=_HOURLY_RMSE_MIN_ET_HOUR,
        )

        # Persist to today's system_state row.
        if state:
            state.ou_max_stationary_std_calibrated = calibrated_cap
            state.nwp_rmse_n_dates = n_dates_with_data
            state.last_updated_utc = datetime.now(timezone.utc)
            db_manager.upsert_system_state(state)

        return calibrated_cap

    except Exception as exc:
        logger.error("calibrator.ou_max_std.failed", error=str(exc))
        return None


# ---------------------------------------------------------------------------
# 3b. Kalman bias decay calibration (AR(1) Yule-Walker)
# ---------------------------------------------------------------------------


def calibrate_kalman_bias_decay(
    target_date: Optional[date] = None,
    lookback_days: Optional[int] = None,
) -> Optional[float]:
    """Calibrate kalman_bias_decay from the AR(1) structure of intraday NWP errors.

    The optimal F[1,1] for a bias state that follows B[t] = φ·B[t-1] + ε is
    exactly φ.  This function estimates φ via Yule-Walker from consecutive
    within-day NWP-error pairs:

        e[h] = asos_nearest(h) - blended_nwp[h]   for ET hours ≥ 10

    Only strictly consecutive hour pairs (e[h], e[h+1]) within the same date
    are used.  No cross-day pairs — the bias is reset at each warm-start so
    overnight autocorrelation is not informative for the intraday decay.

    Yule-Walker:
        φ_raw = Σ(e[t] · e[t+1]) / Σ(e[t]²)

    Clipped to [_BIAS_DECAY_CLIP_MIN, _BIAS_DECAY_CLIP_MAX] = [0.85, 1.0].
    Returns None if fewer than _MIN_BIAS_DECAY_PAIRS = 30 pairs are available.

    Persists the result to system_state.kalman_bias_decay_calibrated.

    Args:
        target_date:   Active trading date for state persistence.  Defaults to today.
        lookback_days: Days of history to scan.  Defaults to
                       settings.calibration_lookback_days.

    Returns:
        Calibrated decay factor in [0.85, 1.0], or None if insufficient data.
    """
    try:
        if target_date is None:
            target_date = get_target_date()
        if lookback_days is None:
            lookback_days = settings.calibration_lookback_days

        state = db_manager.get_system_state(target_date)
        model_weights = state.model_weights if state else {"HRRR": 0.5, "GFS": 0.3, "ECMWF": 0.2}

        end_date = target_date - timedelta(days=1)
        start_date = end_date - timedelta(days=lookback_days - 1)

        numerator = 0.0
        denominator = 0.0
        n_pairs = 0

        d = start_date
        while d <= end_date:
            # Use morning NWP fetch (first in [10 AM, 1 PM) ET window) to avoid
            # look-ahead bias — same constraint as calibrate_ou_max_stationary_std.
            morning_cutoff_utc = (
                datetime.combine(d, datetime.min.time())
                .replace(tzinfo=timezone.utc)
                .astimezone(_EASTERN)
                .replace(hour=13, minute=0, second=0, microsecond=0)
                .astimezone(timezone.utc)
            )
            forecasts = db_manager.get_nwp_forecasts_before_utc(d, morning_cutoff_utc)

            if not forecasts:
                d += timedelta(days=1)
                continue

            blended = _blend_morning_forecast_hourly(forecasts, model_weights)
            if len(blended) < _HOURLY_RMSE_MIN_ET_HOUR + 2:
                d += timedelta(days=1)
                continue

            # Fetch ASOS readings for this date (midnight-to-midnight UTC window).
            day_start_utc = datetime.combine(d, datetime.min.time()).replace(tzinfo=timezone.utc)
            day_end_utc = day_start_utc + timedelta(days=1)
            asos_rows = db_manager.get_asos_readings_since(day_start_utc, station_id="KBOS")
            asos_rows = [r for r in asos_rows if r.observation_time_utc < day_end_utc]

            if not asos_rows:
                d += timedelta(days=1)
                continue

            # Build per-ET-hour error map for hours ≥ _HOURLY_RMSE_MIN_ET_HOUR.
            error_by_et_hour: dict[int, float] = {}
            for et_hour in range(_HOURLY_RMSE_MIN_ET_HOUR, len(blended)):
                target_utc = (
                    datetime.combine(d, datetime.min.time())
                    .replace(tzinfo=_EASTERN)
                    .replace(hour=et_hour)
                    .astimezone(timezone.utc)
                )
                nearest = min(
                    asos_rows,
                    key=lambda r: abs((r.observation_time_utc - target_utc).total_seconds()),
                )
                gap_minutes = abs((nearest.observation_time_utc - target_utc).total_seconds()) / 60
                if gap_minutes > _MAX_ASOS_GAP_MINUTES:
                    continue
                error_by_et_hour[et_hour] = nearest.temperature_f - blended[et_hour]

            # Collect strictly consecutive (h, h+1) pairs only.
            sorted_hours = sorted(error_by_et_hour.keys())
            for i in range(len(sorted_hours) - 1):
                h0, h1 = sorted_hours[i], sorted_hours[i + 1]
                if h1 - h0 != 1:
                    continue  # gap in hours — skip, no cross-hour extrapolation
                e0 = error_by_et_hour[h0]
                e1 = error_by_et_hour[h1]
                numerator += e0 * e1
                denominator += e0 * e0
                n_pairs += 1

            d += timedelta(days=1)

        if n_pairs < _MIN_BIAS_DECAY_PAIRS or denominator < 1e-8:
            logger.warning(
                "calibrator.bias_decay.insufficient_data",
                n_pairs=n_pairs,
                required=_MIN_BIAS_DECAY_PAIRS,
            )
            return None

        phi_raw = numerator / denominator
        calibrated = float(np.clip(phi_raw, _BIAS_DECAY_CLIP_MIN, _BIAS_DECAY_CLIP_MAX))

        logger.info(
            "calibrator.bias_decay.done",
            phi_raw=round(phi_raw, 4),
            calibrated=round(calibrated, 4),
            n_pairs=n_pairs,
            clip_min=_BIAS_DECAY_CLIP_MIN,
            clip_max=_BIAS_DECAY_CLIP_MAX,
        )

        if state is not None:
            state.kalman_bias_decay_calibrated = calibrated
            state.last_updated_utc = datetime.now(timezone.utc)
            db_manager.upsert_system_state(state)

        return calibrated

    except Exception as exc:
        logger.error("calibrator.bias_decay.failed", error=str(exc))
        return None


# ---------------------------------------------------------------------------
# 4. Theta calibration
# ---------------------------------------------------------------------------


def _build_theta_ar1_pairs(
    by_date: dict,
    get_nwp_curve_fn,
    *,
    max_date: date,
    decay_tau_days: int,
) -> tuple[list[float], list[float], list[float], list[float], int]:
    """Build weighted AR(1) departure pairs for theta calibration.

    Shared helper for both calibrate_theta() (pooled) and
    calibrate_theta_by_regime() (AM/PM split).

    Returns:
        Tuple of (x_vals, y_vals, weights, sorted_dates_used, days_used)
        where each (x_vals[i], y_vals[i]) is an AR(1) lag-1 pair
        and weights[i] is the exponential day weight for that pair.
    """
    x_vals: list[float] = []
    y_vals: list[float] = []
    w_vals: list[float] = []
    days_used: int = 0

    for d, day_readings in sorted(by_date.items()):
        nwp_curve = get_nwp_curve_fn(d)
        if len(nwp_curve) < 12:
            continue

        days_back = (max_date - d).days
        day_weight = math.exp(-days_back / decay_tau_days)

        hourly_dep: dict[int, float] = {}
        for hour_et in range(24):
            if hour_et >= len(nwp_curve):
                continue
            try:
                top_of_hour_et = _EASTERN.localize(
                    datetime(d.year, d.month, d.day, hour_et, 0, 0)
                ).astimezone(timezone.utc)
            except Exception:
                continue
            best = min(
                day_readings,
                key=lambda r: abs(
                    (r.observation_time_utc - top_of_hour_et).total_seconds()
                ),
            )
            gap_minutes = abs(
                (best.observation_time_utc - top_of_hour_et).total_seconds()
            ) / 60.0
            if gap_minutes > 40:
                continue
            hourly_dep[hour_et] = best.temperature_f - nwp_curve[hour_et]

        sorted_hours = sorted(hourly_dep.keys())
        for i in range(len(sorted_hours) - 1):
            h0, h1 = sorted_hours[i], sorted_hours[i + 1]
            if h1 - h0 == 1:
                x_vals.append(hourly_dep[h0])
                y_vals.append(hourly_dep[h1])
                w_vals.append(day_weight)

        if len(sorted_hours) >= 2:
            days_used += 1

    return x_vals, y_vals, w_vals, days_used


def _weighted_phi(
    x_vals: list[float],
    y_vals: list[float],
    w_vals: list[float],
) -> Optional[float]:
    """Compute exponentially weighted AR(1) coefficient phi = Σ(w*x*y) / Σ(w*x²).

    Departures from NWP have zero mean by construction, so the weighted OLS
    formula through the origin is exact (no centering needed).

    Returns:
        phi clamped to [0.01, 0.99], or None if denominator < 1e-8.
    """
    x = np.array(x_vals, dtype=float)
    y = np.array(y_vals, dtype=float)
    w = np.array(w_vals, dtype=float)

    denom = float(np.sum(w * x * x))
    if denom < 1e-8:
        return None
    phi = float(np.sum(w * x * y) / denom)
    return max(0.01, min(0.99, phi))


def calibrate_theta(
    target_date: Optional[date] = None,
    lookback_days: Optional[int] = None,
) -> float:
    """Calibrate the OU mean-reversion speed (theta) via weighted AR(1) fit.

    Fits a weighted AR(1) model to hourly temperature departures from the NWP
    forecast. Recent days are up-weighted by exp(-d/tau) (tau=calibration_
    decay_tau_days). The AR(1) coefficient phi is related to theta by:
    theta = -ln(phi) / dt.

    Args:
        target_date:   Active trading date. Defaults to today's target.
        lookback_days: Days of history to use. Defaults to
                       settings.calibration_lookback_days (30).

    Returns:
        Calibrated theta (per hour). Bounded to [0.01, 2.0].

    Raises:
        Nothing — returns settings.ou_theta on failure.
    """
    if target_date is None:
        target_date = get_target_date()
    if lookback_days is None:
        lookback_days = settings.calibration_lookback_days

    try:
        from kalshi_weather_trader.ingestion.asos_fetcher import fetch_last_n_hours
        from kalshi_weather_trader.ingestion.nwp_fetcher import get_nwp_curve

        readings = fetch_last_n_hours(hours=lookback_days * 24)
        if len(readings) < 24:
            logger.warning("calibrator.theta.insufficient_readings", n=len(readings))
            return settings.ou_theta

        by_date: dict[date, list] = {}
        for r in readings:
            d = r.observation_time_utc.astimezone(_EASTERN).date()
            by_date.setdefault(d, []).append(r)

        max_date = max(by_date.keys())
        x_vals, y_vals, w_vals, days_used = _build_theta_ar1_pairs(
            by_date,
            get_nwp_curve,
            max_date=max_date,
            decay_tau_days=settings.calibration_decay_tau_days,
        )

        if len(x_vals) < 12:
            logger.warning(
                "calibrator.theta.insufficient_departure_pairs",
                n_pairs=len(x_vals),
                days_used=days_used,
            )
            return settings.ou_theta

        phi = _weighted_phi(x_vals, y_vals, w_vals)
        if phi is None:
            logger.warning("calibrator.theta.degenerate_variance")
            return settings.ou_theta

        dt_hours = 1.0
        theta = float(-np.log(phi) / dt_hours)
        theta = max(0.01, min(2.0, theta))
        theta = round(theta, 4)

        logger.info(
            "calibrator.theta.done",
            theta=theta,
            phi=round(phi, 4),
            n_pairs=len(x_vals),
            days_used=days_used,
            decay_tau_days=settings.calibration_decay_tau_days,
        )

        state = db_manager.get_system_state(target_date)
        if state:
            state.theta_decay = theta
            state.last_updated_utc = datetime.now(timezone.utc)
            db_manager.upsert_system_state(state)

        return theta
    except Exception as exc:
        logger.error("calibrator.theta.failed", error=str(exc))
        return settings.ou_theta


def calibrate_theta_by_regime(
    target_date: Optional[date] = None,
    lookback_days: Optional[int] = None,
) -> tuple[Optional[float], Optional[float]]:
    """Calibrate AM/PM regime-specific theta values via weighted AR(1) fit.

    Splits departure pairs into two pools by the source hour (h0):
      - AM regime: h0 in [6, 13) — morning solar heating phase. Lower theta
        expected (departures from NWP persist while clouds/albedo dominate).
      - PM regime: h0 in [13, 20) — convective-mixing peak phase. Higher theta
        expected (thermostat effect pulls temperature back to NWP).
      - Hours 0–5 and 20–23 (overnight/evening) contribute to the pooled
        calibrate_theta() only, not to regime-specific estimates.

    Requires ≥ 20 AR(1) pairs per regime for calibration. Falls back to (None,
    None) if insufficient data, causing run_simulation() to use scalar theta.

    Args:
        target_date:   Active trading date. Defaults to today's target.
        lookback_days: Days of history. Defaults to settings.calibration_lookback_days.

    Returns:
        Tuple of (theta_am, theta_pm). Each is a float in [0.01, 2.0] or None.
        None means the scalar theta_decay should be used for that regime.

    Raises:
        Nothing — returns (None, None) on failure.
    """
    _MIN_REGIME_PAIRS = 20  # minimum pairs for a reliable regime-specific estimate

    if target_date is None:
        target_date = get_target_date()
    if lookback_days is None:
        lookback_days = settings.calibration_lookback_days

    try:
        from kalshi_weather_trader.ingestion.asos_fetcher import fetch_last_n_hours
        from kalshi_weather_trader.ingestion.nwp_fetcher import get_nwp_curve

        readings = fetch_last_n_hours(hours=lookback_days * 24)
        if len(readings) < 24:
            logger.warning("calibrator.theta_regime.insufficient_readings", n=len(readings))
            return None, None

        by_date: dict[date, list] = {}
        for r in readings:
            d = r.observation_time_utc.astimezone(_EASTERN).date()
            by_date.setdefault(d, []).append(r)

        max_date = max(by_date.keys())
        # Get ALL pairs (pooled); we'll split by source hour index below.
        x_all, y_all, w_all, days_used = _build_theta_ar1_pairs(
            by_date,
            get_nwp_curve,
            max_date=max_date,
            decay_tau_days=settings.calibration_decay_tau_days,
        )

        # _build_theta_ar1_pairs doesn't track source hours, so we need to
        # rebuild the split. Re-run the bucketing and split into AM/PM.
        am_x: list[float] = []
        am_y: list[float] = []
        am_w: list[float] = []
        pm_x: list[float] = []
        pm_y: list[float] = []
        pm_w: list[float] = []

        for d, day_readings in sorted(by_date.items()):
            nwp_curve = get_nwp_curve(d)
            if len(nwp_curve) < 12:
                continue

            days_back = (max_date - d).days
            day_weight = math.exp(-days_back / settings.calibration_decay_tau_days)

            hourly_dep: dict[int, float] = {}
            for hour_et in range(24):
                if hour_et >= len(nwp_curve):
                    continue
                try:
                    top_of_hour_et = _EASTERN.localize(
                        datetime(d.year, d.month, d.day, hour_et, 0, 0)
                    ).astimezone(timezone.utc)
                except Exception:
                    continue
                best = min(
                    day_readings,
                    key=lambda r: abs(
                        (r.observation_time_utc - top_of_hour_et).total_seconds()
                    ),
                )
                gap_minutes = abs(
                    (best.observation_time_utc - top_of_hour_et).total_seconds()
                ) / 60.0
                if gap_minutes > 40:
                    continue
                hourly_dep[hour_et] = best.temperature_f - nwp_curve[hour_et]

            sorted_hours = sorted(hourly_dep.keys())
            for i in range(len(sorted_hours) - 1):
                h0, h1 = sorted_hours[i], sorted_hours[i + 1]
                if h1 - h0 != 1:
                    continue
                x, y, w = hourly_dep[h0], hourly_dep[h1], day_weight
                if 6 <= h0 < 13:      # AM source hour
                    am_x.append(x); am_y.append(y); am_w.append(w)
                elif 13 <= h0 < 20:   # PM source hour
                    pm_x.append(x); pm_y.append(y); pm_w.append(w)

        def _regime_theta(xs: list, ys: list, ws: list, regime: str) -> Optional[float]:
            if len(xs) < _MIN_REGIME_PAIRS:
                logger.info(
                    "calibrator.theta_regime.insufficient",
                    regime=regime,
                    n_pairs=len(xs),
                    required=_MIN_REGIME_PAIRS,
                )
                return None
            phi = _weighted_phi(xs, ys, ws)
            if phi is None:
                return None
            t = float(-np.log(phi) / 1.0)
            return round(max(0.01, min(2.0, t)), 4)

        theta_am = _regime_theta(am_x, am_y, am_w, "AM")
        theta_pm = _regime_theta(pm_x, pm_y, pm_w, "PM")

        logger.info(
            "calibrator.theta_regime.done",
            theta_am=theta_am,
            theta_pm=theta_pm,
            n_am=len(am_x),
            n_pm=len(pm_x),
            days_used=days_used,
        )

        state = db_manager.get_system_state(target_date)
        if state:
            state.theta_am = theta_am
            state.theta_pm = theta_pm
            state.last_updated_utc = datetime.now(timezone.utc)
            db_manager.upsert_system_state(state)

        return theta_am, theta_pm

    except Exception as exc:
        logger.error("calibrator.theta_regime.failed", error=str(exc))
        return None, None


# ---------------------------------------------------------------------------
# Snapshot recording
# ---------------------------------------------------------------------------


def record_snapshot(
    target_date: Optional[date] = None,
    is_forced: bool = False,
) -> None:
    """Collect current system state, run Monte Carlo, and persist a snapshot.

    This is called every 2 hours by the scheduler and on demand from the UI.

    Args:
        target_date: Active trading date. Defaults to today's target.
        is_forced:   True if triggered manually from the UI.

    Returns:
        None

    Raises:
        Nothing — all errors are logged.
    """
    from kalshi_weather_trader.db.schemas import IntradaySnapshotDocument
    from kalshi_weather_trader.ingestion.asos_fetcher import fetch_current_observation
    from kalshi_weather_trader.ingestion.kalshi_fetcher import get_kalshi_fetcher
    from kalshi_weather_trader.ingestion.nwp_fetcher import get_blended_forecast, get_nwp_curve
    from kalshi_weather_trader.quant.mc_params_builder import build_mc_params
    from kalshi_weather_trader.quant.monte_carlo import price_full_distribution

    if target_date is None:
        target_date = get_target_date()

    now_utc = datetime.now(timezone.utc)
    now_et = now_utc.astimezone(_EASTERN)
    snapshot_time_eastern = now_et.strftime("%H:%M")

    # Gather ASOS
    asos = fetch_current_observation()
    if asos is None:
        logger.error("calibrator.snapshot.no_asos")
        return

    # Gather market state
    try:
        market = db_manager.get_market(target_date)
    except Exception:
        market = None

    # Gather system state (Kalman + calibration params)
    try:
        state = db_manager.get_system_state(target_date)
    except Exception:
        state = None

    kalman_T = state.kalman_temp_estimate if state else asos.temperature_f

    # Gather NWP forecasts
    try:
        nwp_forecasts = db_manager.get_latest_nwp_forecasts(target_date)
        hrrr_high = nwp_forecasts.get("HRRR", None)
        gfs_high = nwp_forecasts.get("GFS", None)
        ecmwf_high = nwp_forecasts.get("ECMWF", None)

        hrrr_high_f = hrrr_high.predicted_daily_high if hrrr_high else None
        gfs_high_f = gfs_high.predicted_daily_high if gfs_high else None
        ecmwf_high_f = ecmwf_high.predicted_daily_high if ecmwf_high else None
    except Exception:
        hrrr_high_f = gfs_high_f = ecmwf_high_f = None

    blended = get_blended_forecast(target_date) or asos.temperature_f
    nwp_curve = get_nwp_curve(target_date)

    # Build shared MCParams (also derives hard_floor and kalman_B)
    mc_params = build_mc_params(target_date, state, asos, market, nwp_curve)
    hard_floor = mc_params.hard_floor
    kalman_B = mc_params.bias

    # Gather Kalshi market data
    kalshi_bid = kalshi_ask = kalshi_implied_prob = kalshi_strike = None
    market_ticker = None
    market_dict: dict | None = None
    try:
        fetcher = get_kalshi_fetcher()
        market_dict = fetcher.get_best_market_for_date(target_date)
        if market_dict:
            market_ticker = market_dict.get("ticker")
            kalshi_strike = fetcher.extract_strike_from_ticker(market_ticker or "")
            yes_bid = market_dict.get("yes_bid") or 0
            yes_ask = market_dict.get("yes_ask") or 0
            if yes_bid and yes_ask:
                kalshi_bid = round(yes_bid / 100.0, 4)
                kalshi_ask = round(yes_ask / 100.0, 4)
                kalshi_implied_prob = round((yes_bid + yes_ask) / 200.0, 4)
    except Exception as exc:
        logger.warning("calibrator.snapshot.kalshi_fetch_failed", error=str(exc))

    # Run Monte Carlo
    fair_value_prob = None
    edge = None
    if kalshi_strike is not None:
        try:
            from kalshi_weather_trader.quant.monte_carlo import compute_yes_prob

            # Include all threshold values (floor + cap) so bucket probabilities
            # can be computed correctly for any market type.
            snap_floor_raw = market_dict.get("floor_strike") if market_dict else None
            snap_cap_raw = market_dict.get("cap_strike") if market_dict else None
            mc_thresholds: set[float] = {kalshi_strike}
            if snap_floor_raw is not None:
                mc_thresholds.add(float(snap_floor_raw))
            if snap_cap_raw is not None:
                mc_thresholds.add(float(snap_cap_raw))
            mc_strikes = sorted(mc_thresholds)

            mc_result = price_full_distribution(mc_params, mc_strikes, target_date)

            # A.6 diagnostic: log MC inputs and outputs to track down 100% probability
            # issue. If mean_max >> nwp_peak, attractor inflation is still active.
            _eff_floor = mc_params.hard_floor + mc_params.persistence_filter_offset
            logger.info(
                "calibrator.snapshot.mc_diagnostic",
                strike=kalshi_strike,
                mean_max=round(mc_result.mean_max, 3),
                std_max=round(mc_result.std_max, 3),
                hard_floor=mc_params.hard_floor,
                effective_floor=round(_eff_floor, 3),
                bias=mc_params.bias,
                drift_adj=mc_params.drift_adj,
                use_drift_in_attractor=mc_params.use_drift_in_attractor,
            )

            # Compute market-correct P(YES) using floor/cap API fields directly
            fair_value_prob = compute_yes_prob(
                mc_result.probabilities, snap_floor_raw, snap_cap_raw
            )

            logger.info(
                "calibrator.snapshot.p_yes",
                p_yes=fair_value_prob,
                kalshi_strike=kalshi_strike,
                kalshi_ask=kalshi_ask,
            )

            if fair_value_prob is not None and kalshi_ask is not None:
                edge = round(fair_value_prob - kalshi_ask, 4)
        except Exception as exc:
            logger.error("calibrator.snapshot.mc_failed", error=str(exc))

    # Build and persist snapshot
    try:
        doc = IntradaySnapshotDocument(
            target_date=target_date,
            snapshot_time_utc=now_utc,
            snapshot_time_eastern=snapshot_time_eastern,
            current_asos_temp_f=asos.temperature_f,
            current_max_observed_f=hard_floor,
            hrrr_predicted_high=hrrr_high_f,
            gfs_predicted_high=gfs_high_f,
            ecmwf_predicted_high=ecmwf_high_f,
            blended_predicted_high=blended,
            kalman_temp_estimate=kalman_T,
            kalman_bias_estimate=kalman_B,
            kalshi_implied_prob_yes=kalshi_implied_prob,
            kalshi_bid=kalshi_bid,
            kalshi_ask=kalshi_ask,
            kalshi_strike=kalshi_strike,
            model_fair_value_prob=fair_value_prob,
            model_edge=edge,
            is_forced=is_forced,
        )
        db_manager.insert_snapshot(doc)
        logger.info(
            "calibrator.snapshot.done",
            time_et=snapshot_time_eastern,
            asos_temp=asos.temperature_f,
            hard_floor=hard_floor,
            edge=edge,
            forced=is_forced,
        )
    except Exception as exc:
        logger.error("calibrator.snapshot.persist_failed", error=str(exc))


# ---------------------------------------------------------------------------
# Full calibration cycle
# ---------------------------------------------------------------------------


def run_full_calibration(target_date: Optional[date] = None) -> None:
    """Run all four calibration routines in sequence.

    Called nightly at 00:05 Eastern by the scheduler.

    Args:
        target_date: Active trading date. Defaults to today's target.

    Returns:
        None

    Raises:
        Nothing — individual routine failures are logged.
    """
    if target_date is None:
        target_date = get_target_date()

    logger.info("calibrator.full_calibration.start", date=str(target_date))

    calibrate_model_weights(target_date)
    calibrate_intraday_drift(target_date)
    calibrate_sigma(target_date)
    calibrate_theta(target_date)
    calibrate_theta_by_regime(target_date)
    calibrate_persistence_offset(target_date)
    calibrate_ou_max_stationary_std(target_date)
    calibrate_kalman_bias_decay(target_date)

    # Always stamp today's system_state row with the calibration time.
    # calibrate_model_weights() only stamps when Brier scores succeed for ≥2
    # models, and it stamps target_date's row — which may be yesterday when
    # called from job_confirm_settlement().  Stage 6 reads today's row, so
    # we write today's row unconditionally here at the run_full_calibration level.
    try:
        today = get_target_date()
        today_state = db_manager.get_system_state(today)
        if today_state:
            today_state.last_calibrated_utc = datetime.now(timezone.utc)
            today_state.last_updated_utc = datetime.now(timezone.utc)
            db_manager.upsert_system_state(today_state)
    except Exception as exc:
        logger.error("calibrator.full_calibration.stamp_failed", error=str(exc))

    logger.info("calibrator.full_calibration.done", date=str(target_date))


# ---------------------------------------------------------------------------
# Historical ASOS backfill (one-time seeding for cold-start calibration)
# ---------------------------------------------------------------------------


def backfill_historical_asos(days: int = 60) -> tuple[int, int]:
    """Fetch historical KBOS ASOS readings and NWP forecasts for calibration seeding.

    Pulls ``days`` days of history for two data sources:

    **ASOS (IEM Mesonet)**: All readings from IEM starting before the earliest
    row already in the database (or ``days`` days ago if the DB is empty).
    Upserted with ON CONFLICT DO NOTHING — safe to call multiple times.

    **NWP (Open-Meteo)**: Fetches retrospective HRRR/GFS/ECMWF hourly curves
    for each date that has no NWP data in the database yet.  Dates that already
    have live NWP forecasts (from normal app operation) are skipped entirely —
    historical rows are never inserted for those dates.  For the dates that are
    backfilled, ``fetched_at_utc`` is set to noon ET on the historical date so
    that ``get_latest_nwp_forecasts`` always prefers live forecasts (which carry
    today's timestamp) over historical ones, even if the date guard is somehow
    bypassed.

    After this function returns, call ``calibrate_sigma(lookback_days=days)``
    and ``calibrate_theta(lookback_days=days)`` to benefit from the richer
    dataset.  The ordinary ``run_full_calibration()`` uses a 7-day window and
    will NOT automatically pick up the extended history.

    Args:
        days: Number of calendar days of history to fetch.  Defaults to 60.

    Returns:
        Tuple of ``(asos_inserted, nwp_dates_fetched)`` where
        ``asos_inserted`` is new ASOS rows added and ``nwp_dates_fetched``
        is the number of calendar dates for which at least one NWP model
        was successfully fetched and stored.

    Raises:
        Nothing — errors are logged and partial counts are returned.
    """
    from kalshi_weather_trader.ingestion.asos_fetcher import _fetch_iem_since
    from kalshi_weather_trader.ingestion.nwp_fetcher import _fetch_model

    # ------------------------------------------------------------------ ASOS
    asos_inserted = 0
    try:
        earliest_stored = db_manager.get_earliest_asos_reading()
        cutoff_utc = datetime.now(timezone.utc) - timedelta(days=days)
        since_utc = (
            min(earliest_stored.observation_time_utc, cutoff_utc)
            if earliest_stored is not None
            else cutoff_utc
        )

        logger.info("calibrator.backfill.asos_start", since=str(since_utc), days=days)
        readings = _fetch_iem_since(since_utc)
        for r in readings:
            try:
                db_manager.upsert_asos_reading(r)
                asos_inserted += 1
            except Exception as exc:
                logger.warning("calibrator.backfill.asos_upsert_failed", error=str(exc))
        logger.info(
            "calibrator.backfill.asos_done",
            fetched=len(readings),
            inserted=asos_inserted,
        )
    except Exception as exc:
        logger.error("calibrator.backfill.asos_failed", error=str(exc))

    # ------------------------------------------------------------------ NWP
    nwp_dates_fetched = 0
    today = datetime.now(timezone.utc).date()
    start_date = today - timedelta(days=days)

    for delta in range(days):
        hist_date = start_date + timedelta(days=delta)
        if hist_date >= today:
            break

        # Skip dates where live NWP already exists — never clobber real data.
        try:
            existing = db_manager.get_latest_nwp_forecasts(hist_date)
        except Exception:
            existing = {}
        if existing:
            continue

        # Fetch retrospective forecast from Open-Meteo for this date.
        # fetched_at_utc is set to noon ET on the historical date so that
        # get_latest_nwp_forecasts always ranks live forecasts (today's
        # timestamp) above these historical rows.
        noon_utc = _EASTERN.localize(
            datetime(hist_date.year, hist_date.month, hist_date.day, 12, 0, 0)
        ).astimezone(timezone.utc)

        fetched_any = False
        for model_name in ("HRRR", "GFS", "ECMWF"):
            try:
                doc = _fetch_model(model_name, hist_date)
                if doc is None:
                    continue
                doc = doc.model_copy(update={"fetched_at_utc": noon_utc})
                db_manager.upsert_nwp_forecast(doc)
                fetched_any = True
            except Exception as exc:
                logger.warning(
                    "calibrator.backfill.nwp_failed",
                    date=str(hist_date),
                    model=model_name,
                    error=str(exc),
                )

        if fetched_any:
            nwp_dates_fetched += 1
            logger.debug("calibrator.backfill.nwp_date_done", date=str(hist_date))

    logger.info(
        "calibrator.backfill.done",
        asos_inserted=asos_inserted,
        nwp_dates_fetched=nwp_dates_fetched,
    )
    return asos_inserted, nwp_dates_fetched
