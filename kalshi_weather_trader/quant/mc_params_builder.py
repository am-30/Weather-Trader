"""
Shared factory for constructing ``MCParams`` with consistent fallback logic.

All four MCParams construction sites (trader.py, calibrator.py, and two
locations in app.py) import this single function so that fallback behaviour
is defined once and cannot silently diverge.

Fallback priority:
    T0          : state.kalman_temp_estimate → asos_reading.temperature_f → 0.0
    hard_floor  : market.current_max_observed → asos_reading.temperature_f → T0
    nwp_curve   : blended_nwp_curve (if non-empty) → [T0] * 24 (flat fallback)
    drift_adj   : state AM/PM field chosen by current ET hour → 0.0 if no state
    hour_offset : ET hour of day (or DST-aware day-start offset for future day)
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Optional

import pytz

from kalshi_weather_trader.config.settings import settings
from kalshi_weather_trader.db.schemas import (
    ASOSReadingDocument,
    MarketDocument,
    SystemStateDocument,
)
from kalshi_weather_trader.quant.monte_carlo import MCParams

_EASTERN = pytz.timezone("America/New_York")


def build_mc_params_historical(
    past_date: date,
    hour_et: int,
    state: Optional[SystemStateDocument],
    asos_at_hour: Optional[ASOSReadingDocument],
    hard_floor: float,
    nwp_curve: list[float],
    mean_cloudcover: Optional[float] = None,
    ensemble_spread: Optional[float] = None,
) -> MCParams:
    """Build MCParams for historical evaluation at a fixed ET hour.

    Used by Brier scoring to reconstruct the simulation state at ~10 AM on
    each historical trading day.  Unlike ``build_mc_params()``, this function
    takes explicit ``hour_et`` and ``hard_floor`` rather than deriving them
    from ``datetime.now()``.

    Args:
        past_date:        Historical trading date being evaluated.
        hour_et:          ET hour at which to anchor the simulation (typically 10).
        state:            SystemStateDocument for past_date, or None.
        asos_at_hour:     ASOS reading closest to hour_et on past_date, or None.
        hard_floor:       Max ASOS temperature observed up to hour_et on past_date.
        nwp_curve:        Blended NWP hourly curve (ET-indexed) for past_date.
        mean_cloudcover:  Blended mean cloud cover (0–100) from NWP forecasts, or
                          None to use neutral default (50.0, no regime scaling).
        ensemble_spread:  Blended ensemble spread (°F) from NWP forecasts, or
                          None to use neutral default (0.0, no inflation).

    Returns:
        MCParams with day_fraction_remaining = (24 - hour_et) / 24.0,
        hour_offset = hour_et, is_future_day = False.

    Raises:
        Nothing.
    """
    if asos_at_hour is not None:
        T0 = asos_at_hour.temperature_f
    elif state is not None:
        T0 = state.kalman_temp_estimate
    else:
        T0 = hard_floor

    _raw_kalman_B = state.kalman_bias_estimate if state is not None else 0.0
    _cap = settings.kalman_bias_mc_cap
    kalman_B = max(-_cap, min(_cap, _raw_kalman_B))
    theta = state.theta_decay if state is not None else settings.ou_theta
    sigma = state.sigma_volatility if state is not None else settings.ou_sigma
    sigma = max(sigma, settings.ou_sigma_floor)
    sigma_by_block = state.sigma_by_block if state is not None else None
    theta_am = state.theta_am if state is not None else None
    theta_pm = state.theta_pm if state is not None else None
    persistence_offset = (
        state.persistence_filter_offset
        if state is not None
        else settings.persistence_filter_offset
    )
    ou_max_std = (
        state.ou_max_stationary_std_calibrated
        if state is not None and state.ou_max_stationary_std_calibrated is not None
        else settings.ou_max_stationary_std
    )
    drift_adj = 0.0
    if state is not None:
        drift_adj = (
            state.morning_drift_adjustment
            if hour_et < 12
            else state.afternoon_drift_adjustment
        )
    effective_curve: list[float] = nwp_curve if nwp_curve else [T0] * 24
    day_fraction = (24.0 - hour_et) / 24.0

    _ensemble_spread = ensemble_spread if ensemble_spread is not None else 0.0
    _mean_cloudcover = mean_cloudcover if mean_cloudcover is not None else 50.0

    return MCParams(
        T0=T0,
        hard_floor=hard_floor,
        nwp_curve=effective_curve,
        bias=kalman_B,
        theta=theta,
        sigma=sigma,
        drift_adj=drift_adj,
        use_drift_in_attractor=True,
        hour_offset=hour_et,
        day_fraction_remaining=day_fraction,
        is_future_day=False,
        persistence_filter_offset=persistence_offset,
        sigma_by_block=sigma_by_block,
        theta_am=theta_am,
        theta_pm=theta_pm,
        ou_max_stationary_std=ou_max_std,
        ensemble_spread=_ensemble_spread,
        mean_cloudcover_10_16=_mean_cloudcover,
        daily_max_bias=state.nwp_daily_max_bias if state is not None else 0.0,
    )


def build_mc_params(
    target_date: date,
    state: Optional[SystemStateDocument],
    asos_reading: Optional[ASOSReadingDocument],
    market: Optional[MarketDocument],
    blended_nwp_curve: list[float],
) -> MCParams:
    """Construct an ``MCParams`` instance with consistent fallback logic.

    Args:
        target_date:       Active trading date.
        state:             Current ``SystemStateDocument``, or None if unavailable.
        asos_reading:      Latest ASOS observation, or None if unavailable.
        market:            ``MarketDocument`` for today, or None if unavailable.
        blended_nwp_curve: Blended hourly NWP temperature curve (ET-indexed).
                           Pass an empty list when no NWP data is available.

    Returns:
        Fully-populated ``MCParams`` ready to pass to ``price_full_distribution``.

    Raises:
        Nothing — all branches have defined fallbacks.
    """
    now_et = datetime.now(timezone.utc).astimezone(_EASTERN)
    hour_et = now_et.hour

    # -----------------------------------------------------------------------
    # After the 6 PM rollover, target_date is tomorrow but the Kalman filter
    # continues tracking today's calendar date (now_et.date()).  The DB row
    # for target_date (tomorrow) does not exist until midnight.  Fall back to
    # today's system_state so the MC has the converged bias and calibration
    # parameters rather than defaults.
    # -----------------------------------------------------------------------
    effective_state = state
    if effective_state is None and target_date > now_et.date():
        from kalshi_weather_trader.db import db_manager as _mc_db
        try:
            effective_state = _mc_db.get_system_state(now_et.date())
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # Starting temperature (Kalman estimate → ASOS → zero)
    # -----------------------------------------------------------------------
    if effective_state is not None:
        T0 = effective_state.kalman_temp_estimate
    elif asos_reading is not None:
        T0 = asos_reading.temperature_f
    else:
        T0 = 0.0

    # -----------------------------------------------------------------------
    # Kalman parameters
    # -----------------------------------------------------------------------
    _raw_kalman_B = effective_state.kalman_bias_estimate if effective_state is not None else 0.0
    _cap = settings.kalman_bias_mc_cap
    kalman_B = max(-_cap, min(_cap, _raw_kalman_B))
    theta = effective_state.theta_decay if effective_state is not None else settings.ou_theta
    sigma = effective_state.sigma_volatility if effective_state is not None else settings.ou_sigma
    sigma = max(sigma, settings.ou_sigma_floor)
    sigma_by_block = effective_state.sigma_by_block if effective_state is not None else None
    theta_am = effective_state.theta_am if effective_state is not None else None
    theta_pm = effective_state.theta_pm if effective_state is not None else None
    persistence_offset = (
        effective_state.persistence_filter_offset
        if effective_state is not None
        else settings.persistence_filter_offset
    )
    ou_max_std = (
        effective_state.ou_max_stationary_std_calibrated
        if effective_state is not None
        and effective_state.ou_max_stationary_std_calibrated is not None
        else settings.ou_max_stationary_std
    )

    # -----------------------------------------------------------------------
    # Hard floor (observed daily max → ASOS temp → T0)
    # -----------------------------------------------------------------------
    if market is not None and market.current_max_observed is not None:
        hard_floor = float(market.current_max_observed)
    elif asos_reading is not None:
        hard_floor = asos_reading.temperature_f
    else:
        hard_floor = T0

    # -----------------------------------------------------------------------
    # Intraday drift (AM vs PM split; zero when no state)
    # -----------------------------------------------------------------------
    drift_adj = 0.0
    if effective_state is not None:
        drift_adj = (
            effective_state.morning_drift_adjustment
            if hour_et < 12
            else effective_state.afternoon_drift_adjustment
        )

    # -----------------------------------------------------------------------
    # NWP curve and hour_offset.
    #
    # Post-6 PM rollover (is_future_day):
    #   Stitch today's remaining NWP hours (now → 11 PM tonight) onto the
    #   front of tomorrow's full-day curve (midnight → 11 PM tomorrow).
    #   hour_offset=0 means "start of stitched curve = current wall-clock time",
    #   so the OU anchor is physically valid and is_future_day is set False.
    #   day_frac_override drives n_steps to cover the full stitched window.
    #
    #   Fallback (today NWP not in DB): use tomorrow's curve at midnight,
    #   with anchor suppressed (original is_future_day=True behaviour).
    #
    # Same-day: use the blended curve and the current ET hour directly.
    # -----------------------------------------------------------------------
    is_future_day = target_date > now_et.date()
    day_frac_override: Optional[float] = None
    bridge_steps: int = 0

    if is_future_day:
        from kalshi_weather_trader.ingestion.nwp_fetcher import get_stitched_nwp_curve
        stitched, bridge_hours = get_stitched_nwp_curve(now_et.date(), target_date, hour_et)
        if stitched:
            effective_curve = stitched
            hour_offset = 0        # position 0 = current wall-clock time in stitched curve
            is_future_day = False  # anchor valid: T0 is at the start of the stitched curve
            day_frac_override = len(stitched) / 24.0
            # 12 five-minute steps per hour (dt = 5/60 h, fixed throughout the system).
            # bridge_steps tells run_simulation to evolve paths through tonight's
            # pre-window hours without counting those temperatures toward paths_max.
            bridge_steps = bridge_hours * 12
        else:
            # Fallback: today NWP not yet in DB — simulate from midnight tomorrow
            effective_curve = blended_nwp_curve if blended_nwp_curve else [T0] * 24
            is_dst = bool(now_et.dst())
            hour_offset = 1 if is_dst else 0
            # is_future_day stays True (anchor suppressed); bridge_steps stays 0
    else:
        effective_curve = blended_nwp_curve if blended_nwp_curve else [T0] * 24
        hour_offset = hour_et
        # bridge_steps = 0: every step is within the active NWS window

    # Ensemble spread and cloud cover for regime adjustment
    try:
        from kalshi_weather_trader.db import db_manager as _mc_db2
        _ensemble_spread = _mc_db2.get_latest_ensemble_spread(target_date) or 0.0
        _mean_cloudcover = _mc_db2.get_blended_cloudcover(target_date) or 50.0
    except Exception:
        _ensemble_spread = 0.0
        _mean_cloudcover = 50.0

    return MCParams(
        T0=T0,
        hard_floor=hard_floor,
        nwp_curve=effective_curve,
        bias=kalman_B,
        theta=theta,
        sigma=sigma,
        drift_adj=drift_adj,
        use_drift_in_attractor=True,
        hour_offset=hour_offset,
        is_future_day=is_future_day,
        day_fraction_remaining=day_frac_override,
        bridge_steps=bridge_steps,
        persistence_filter_offset=persistence_offset,
        sigma_by_block=sigma_by_block,
        theta_am=theta_am,
        theta_pm=theta_pm,
        ou_max_stationary_std=ou_max_std,
        ensemble_spread=_ensemble_spread,
        mean_cloudcover_10_16=_mean_cloudcover,
        daily_max_bias=effective_state.nwp_daily_max_bias if effective_state is not None else 0.0,
        # n_paths intentionally omitted — MCParams defaults to settings.mc_n_paths
    )
