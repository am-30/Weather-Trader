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
            strike = (
                float(market.kalshi_strike) if market.kalshi_strike is not None else None
            )
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
    from kalshi_weather_trader.ingestion.nwp_fetcher import get_nwp_curve
    from kalshi_weather_trader.quant.monte_carlo import estimate_sigma_from_historical

    if target_date is None:
        target_date = get_target_date()

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

        sigma = estimate_sigma_from_historical(readings, nwp_curves=nwp_curves)
        logger.info("calibrator.sigma.done", sigma=sigma, n_readings=len(readings), n_nwp_dates=len(nwp_curves))

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
        from kalshi_weather_trader.ingestion.nwp_fetcher import get_nwp_curve

        readings = fetch_last_n_hours(hours=lookback_days * 24)
        if len(readings) < 24:
            logger.warning("calibrator.theta.insufficient_readings", n=len(readings))
            return settings.ou_theta

        # Group readings by ET date.
        by_date: dict[date, list] = {}
        for r in readings:
            d = r.observation_time_utc.astimezone(_EASTERN).date()
            by_date.setdefault(d, []).append(r)

        # Build AR(1) pairs on NWP departure residuals.
        # For each date we bucket readings to the nearest top-of-hour,
        # compute departure = T_obs - nwp_curve[hour_et], then collect
        # consecutive within-day (x, y) lag-1 pairs.
        x_vals: list[float] = []
        y_vals: list[float] = []
        days_used = 0

        for d, day_readings in sorted(by_date.items()):
            nwp_curve = get_nwp_curve(d)
            if len(nwp_curve) < 12:
                logger.debug(
                    "calibrator.theta.date_skipped_no_curve",
                    date=str(d),
                    curve_len=len(nwp_curve),
                )
                continue

            # Bucket to hourly: for each ET hour 0–23, pick the reading
            # whose observation_time_utc is closest to the top of that hour.
            hourly_dep: dict[int, float] = {}
            for hour_et in range(24):
                if hour_et >= len(nwp_curve):
                    continue
                # Top of this hour in UTC for the ET date d
                try:
                    top_of_hour_et = _EASTERN.localize(
                        datetime(d.year, d.month, d.day, hour_et, 0, 0)
                    ).astimezone(timezone.utc)
                except Exception:
                    continue
                # Find the reading nearest to this timestamp
                best = min(
                    day_readings,
                    key=lambda r: abs(
                        (r.observation_time_utc - top_of_hour_et).total_seconds()
                    ),
                )
                # Only use it if within 40 minutes of top-of-hour (gap guard)
                gap_minutes = abs(
                    (best.observation_time_utc - top_of_hour_et).total_seconds()
                ) / 60.0
                if gap_minutes > 40:
                    continue
                hourly_dep[hour_et] = best.temperature_f - nwp_curve[hour_et]

            # Collect consecutive within-day lag-1 AR(1) pairs.
            sorted_hours = sorted(hourly_dep.keys())
            for i in range(len(sorted_hours) - 1):
                h0, h1 = sorted_hours[i], sorted_hours[i + 1]
                if h1 - h0 == 1:  # strictly consecutive hours only
                    x_vals.append(hourly_dep[h0])
                    y_vals.append(hourly_dep[h1])

            if len(sorted_hours) >= 2:
                days_used += 1

        if len(x_vals) < 12:
            logger.warning(
                "calibrator.theta.insufficient_departure_pairs",
                n_pairs=len(x_vals),
                days_used=days_used,
            )
            return settings.ou_theta

        x = np.array(x_vals, dtype=float)
        y = np.array(y_vals, dtype=float)

        # OLS: phi = cov(x, y) / var(x)
        var_x = float(np.var(x))
        if var_x < 1e-8:
            logger.warning("calibrator.theta.degenerate_variance")
            return settings.ou_theta

        phi = float(np.cov(x, y)[0, 1] / var_x)
        phi = max(0.01, min(0.99, phi))

        dt_hours = 1.0  # hourly AR(1)
        theta = float(-np.log(phi) / dt_hours)
        theta = max(0.01, min(2.0, theta))
        theta = round(theta, 4)

        logger.info(
            "calibrator.theta.done",
            theta=theta,
            phi=round(phi, 4),
            n_pairs=len(x_vals),
            days_used=days_used,
        )

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
