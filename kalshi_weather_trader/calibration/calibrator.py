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

    For each past date, uses the model's predicted daily high as the mean of
    a Gaussian with std = 2°F to compute P(max >= official_high).  The Brier
    score is (forecast_prob - outcome)^2 where outcome = 1 (high was reached).

    Args:
        model_name:    'HRRR', 'GFS', or 'ECMWF'.
        lookback_days: Number of past days to evaluate.

    Returns:
        Mean Brier score (lower is better), or None if insufficient data.

    Raises:
        Nothing — errors are logged.
    """
    from scipy.stats import norm

    today = date.today()
    scores: list[float] = []

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

            # Gaussian CDF: P(actual_max >= predicted_high)
            # We treat "outcome" as 1 since the day resolved
            # Brier score: (p_yes - outcome)^2 where outcome = 1 if official_high
            # We compute P(X >= official_high) where X ~ N(predicted_high, 2.0)
            p_yes = 1.0 - norm.cdf(official_high, loc=predicted_high, scale=2.0)
            # outcome: did the temperature actually exceed predicted_high?
            outcome = 1.0 if official_high >= predicted_high else 0.0
            scores.append((p_yes - outcome) ** 2)

        except Exception as exc:
            logger.warning(
                "calibrator.brier.date_failed",
                model=model_name,
                date=str(past_date),
                error=str(exc),
            )
            continue

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
    default_weights = {"HRRR": 0.5, "GFS": 0.3, "ECMWF": 0.2}

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
    lookback_days: int = 7,
) -> tuple[float, float]:
    """Calibrate AM/PM drift adjustments from recent intraday snapshots.

    Pools snapshots across the past ``lookback_days`` settled trading days,
    groups them by morning (< 12:00 ET) and afternoon (>= 12:00 ET), and
    computes mean(blended_predicted_high - final_official_high) as the drift
    correction.  Using multiple days reduces noise from individual outlier days.

    Args:
        target_date:   Active trading date. Defaults to today's target.
        lookback_days: Number of past days to include. Defaults to 7.

    Returns:
        Tuple of (morning_drift_adj, afternoon_drift_adj) in °F.
        Both default to 0.0 if fewer than 2 settled days exist.

    Raises:
        Nothing — errors are logged.
    """
    if target_date is None:
        target_date = get_target_date()

    morning_errors: list[float] = []
    afternoon_errors: list[float] = []
    days_used: int = 0

    for d in range(1, lookback_days + 1):
        past_date = target_date - timedelta(days=d)
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
                    morning_errors.append(error)
                else:
                    afternoon_errors.append(error)

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

    morning_adj = round(-float(np.mean(morning_errors)), 3) if morning_errors else 0.0
    afternoon_adj = round(-float(np.mean(afternoon_errors)), 3) if afternoon_errors else 0.0

    logger.info(
        "calibrator.drift.done",
        morning_adj=morning_adj,
        afternoon_adj=afternoon_adj,
        morning_n=len(morning_errors),
        afternoon_n=len(afternoon_errors),
        days_used=days_used,
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
    lookback_days: int = 7,
) -> float:
    """Calibrate the OU sigma (diffusion) from recent ASOS 5-minute diffs.

    Args:
        target_date:   Active trading date. Defaults to today's target.
        lookback_days: Days of ASOS history to use. Defaults to 7.

    Returns:
        Calibrated sigma in °F/sqrt-hour.

    Raises:
        Nothing — returns settings.ou_sigma on failure.
    """
    from kalshi_weather_trader.ingestion.asos_fetcher import fetch_last_n_hours
    from kalshi_weather_trader.quant.monte_carlo import estimate_sigma_from_historical

    if target_date is None:
        target_date = get_target_date()

    try:
        readings = fetch_last_n_hours(hours=lookback_days * 24)
        if len(readings) < 12:
            logger.warning("calibrator.sigma.insufficient_readings", n=len(readings))
            return settings.ou_sigma

        sigma = estimate_sigma_from_historical(readings)
        logger.info("calibrator.sigma.done", sigma=sigma, n_readings=len(readings))

        # Persist
        state = db_manager.get_system_state(target_date)
        if state:
            state.sigma_volatility = sigma
            state.last_updated_utc = datetime.now(timezone.utc)
            db_manager.upsert_system_state(state)

        return sigma
    except Exception as exc:
        logger.error("calibrator.sigma.failed", error=str(exc))
        return settings.ou_sigma


# ---------------------------------------------------------------------------
# 4. Theta calibration
# ---------------------------------------------------------------------------


def calibrate_theta(
    target_date: Optional[date] = None,
    lookback_days: int = 7,
) -> float:
    """Calibrate the OU mean-reversion speed (theta) via AR(1) fit.

    Fits an AR(1) model to hourly temperature departures from the NWP forecast.
    The AR(1) coefficient phi is related to theta by: theta = -ln(phi) / dt

    Args:
        target_date:   Active trading date. Defaults to today's target.
        lookback_days: Days of history to use. Defaults to 7.

    Returns:
        Calibrated theta (per hour). Bounded to [0.01, 2.0].

    Raises:
        Nothing — returns settings.ou_theta on failure.
    """
    if target_date is None:
        target_date = get_target_date()

    try:
        from kalshi_weather_trader.ingestion.asos_fetcher import fetch_last_n_hours

        readings = fetch_last_n_hours(hours=lookback_days * 24)
        if len(readings) < 24:
            logger.warning("calibrator.theta.insufficient_readings", n=len(readings))
            return settings.ou_theta

        # Resample to hourly by picking the reading nearest each hour
        temps = np.array([r.temperature_f for r in readings], dtype=float)

        # Compute hourly departures from rolling mean (proxy for NWP departure)
        if len(temps) < 4:
            return settings.ou_theta

        # AR(1) on the hourly temperatures
        # Sub-sample to hourly (every 12th 5-min reading)
        hourly_temps = temps[::12]
        if len(hourly_temps) < 4:
            return settings.ou_theta

        y = hourly_temps[1:]
        x = hourly_temps[:-1]

        # OLS: phi = cov(x,y) / var(x)
        phi = float(np.cov(x, y)[0, 1] / np.var(x))
        phi = max(0.01, min(0.99, phi))  # bound phi to (0, 1)

        dt_hours = 1.0  # hourly AR
        theta = float(-np.log(phi) / dt_hours)
        theta = max(0.01, min(2.0, theta))  # bound theta
        theta = round(theta, 4)

        logger.info("calibrator.theta.done", theta=theta, phi=round(phi, 4))

        # Persist
        state = db_manager.get_system_state(target_date)
        if state:
            state.theta_decay = theta
            state.last_updated_utc = datetime.now(timezone.utc)
            db_manager.upsert_system_state(state)

        return theta
    except Exception as exc:
        logger.error("calibrator.theta.failed", error=str(exc))
        return settings.ou_theta


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
    from kalshi_weather_trader.quant.monte_carlo import MCParams, price_full_distribution

    if target_date is None:
        target_date = get_target_date()

    now_utc = datetime.now(timezone.utc)
    now_et = now_utc.astimezone(_EASTERN)
    snapshot_time_eastern = now_et.strftime("%H:%M")
    hour_et = now_et.hour

    # Gather ASOS
    asos = fetch_current_observation()
    if asos is None:
        logger.error("calibrator.snapshot.no_asos")
        return

    # Gather market state
    try:
        market = db_manager.get_market(target_date)
        hard_floor = market.current_max_observed if market else asos.temperature_f
    except Exception:
        hard_floor = asos.temperature_f

    # Gather system state (Kalman + calibration params)
    try:
        state = db_manager.get_system_state(target_date)
    except Exception:
        state = None

    kalman_T = state.kalman_temp_estimate if state else asos.temperature_f
    kalman_B = state.kalman_bias_estimate if state else 0.0
    theta = state.theta_decay if state else settings.ou_theta
    sigma = state.sigma_volatility if state else settings.ou_sigma

    # Determine drift adjustment (AM vs PM) using hour_et computed above
    drift_adj = 0.0
    if state:
        drift_adj = (
            state.morning_drift_adjustment
            if hour_et < 12
            else state.afternoon_drift_adjustment
        )

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

            # hour_offset: use ET hour; for a future day use the same DST-aware
            # offset as trader.py (1 during EDT, 0 during EST).
            is_future_day = target_date > now_et.date()
            is_dst = bool(now_et.dst())
            snap_hour_offset = (1 if is_dst else 0) if is_future_day else hour_et

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

            mc_params = MCParams(
                T0=kalman_T,
                hard_floor=hard_floor,
                nwp_curve=nwp_curve,
                bias=kalman_B,
                theta=theta,
                sigma=sigma,
                drift_adj=drift_adj,
                hour_offset=snap_hour_offset,
                is_future_day=is_future_day,
            )
            mc_result = price_full_distribution(mc_params, mc_strikes, target_date)

            # Compute market-correct P(YES) using floor/cap API fields directly
            fair_value_prob = compute_yes_prob(
                mc_result.probabilities, snap_floor_raw, snap_cap_raw
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

    logger.info("calibrator.full_calibration.done", date=str(target_date))
