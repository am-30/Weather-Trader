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
) -> MCParams:
    """Build MCParams for historical evaluation at a fixed ET hour.

    Used by Brier scoring to reconstruct the simulation state at ~10 AM on
    each historical trading day.  Unlike ``build_mc_params()``, this function
    takes explicit ``hour_et`` and ``hard_floor`` rather than deriving them
    from ``datetime.now()``.

    Args:
        past_date:    Historical trading date being evaluated.
        hour_et:      ET hour at which to anchor the simulation (typically 10).
        state:        SystemStateDocument for past_date, or None.
        asos_at_hour: ASOS reading closest to hour_et on past_date, or None.
        hard_floor:   Max ASOS temperature observed up to hour_et on past_date.
        nwp_curve:    Blended NWP hourly curve (ET-indexed) for past_date.

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

    kalman_B = state.kalman_bias_estimate if state is not None else 0.0
    theta = state.theta_decay if state is not None else settings.ou_theta
    sigma = state.sigma_volatility if state is not None else settings.ou_sigma
    # 10 AM is morning — use morning drift
    drift_adj = state.morning_drift_adjustment if state is not None else 0.0
    effective_curve: list[float] = nwp_curve if nwp_curve else [T0] * 24
    day_fraction = (24.0 - hour_et) / 24.0

    return MCParams(
        T0=T0,
        hard_floor=hard_floor,
        nwp_curve=effective_curve,
        bias=kalman_B,
        theta=theta,
        sigma=sigma,
        drift_adj=drift_adj,
        hour_offset=hour_et,
        day_fraction_remaining=day_fraction,
        is_future_day=False,
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
    # Starting temperature (Kalman estimate → ASOS → zero)
    # -----------------------------------------------------------------------
    if state is not None:
        T0 = state.kalman_temp_estimate
    elif asos_reading is not None:
        T0 = asos_reading.temperature_f
    else:
        T0 = 0.0

    # -----------------------------------------------------------------------
    # Kalman parameters
    # -----------------------------------------------------------------------
    kalman_B = state.kalman_bias_estimate if state is not None else 0.0
    theta = state.theta_decay if state is not None else settings.ou_theta
    sigma = state.sigma_volatility if state is not None else settings.ou_sigma

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
    if state is not None:
        drift_adj = (
            state.morning_drift_adjustment
            if hour_et < 12
            else state.afternoon_drift_adjustment
        )

    # -----------------------------------------------------------------------
    # NWP curve — flat fallback at T0 when empty
    # -----------------------------------------------------------------------
    effective_curve: list[float] = blended_nwp_curve if blended_nwp_curve else [T0] * 24

    # -----------------------------------------------------------------------
    # hour_offset: ET-indexed position in the NWP curve.
    # For a future day (post-6 PM rollover) simulation always starts at the
    # NWS observation window start: midnight EST = curve index 1 in EDT,
    # index 0 in EST.  For today, use the current ET hour directly.
    # -----------------------------------------------------------------------
    is_future_day = target_date > now_et.date()
    is_dst = bool(now_et.dst())
    hour_offset = (1 if is_dst else 0) if is_future_day else hour_et

    return MCParams(
        T0=T0,
        hard_floor=hard_floor,
        nwp_curve=effective_curve,
        bias=kalman_B,
        theta=theta,
        sigma=sigma,
        drift_adj=drift_adj,
        hour_offset=hour_offset,
        is_future_day=is_future_day,
        # n_paths intentionally omitted — MCParams defaults to settings.mc_n_paths
    )
