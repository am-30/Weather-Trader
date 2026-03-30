"""
Ornstein-Uhlenbeck Monte Carlo simulation engine.

Prices the probability of the daily maximum temperature exceeding a
strike temperature for the KBOS market, using the receding-horizon approach.

Process:
    dT = theta * (mu_t - T_t) * dt + sigma * sqrt(dt) * Z
    where:
        theta = mean-reversion speed (per hour)
        mu_t  = NWP forecast + Kalman bias correction + intraday drift
        sigma = diffusion coefficient (°F / sqrt-hour)
        dt    = 5/60 (5-minute steps)
        Z     ~ N(0,1) pre-generated matrix

Hard floor guarantee:
    paths_max is initialised at hard_floor, so P(max >= strike) = 1
    for all strikes <= hard_floor, regardless of future path realisation.

Performance:
    The full N×M random matrix is pre-generated before the loop,
    making inner loop work pure NumPy vectorised operations.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Optional

import numpy as np
import pytz
import structlog

from kalshi_weather_trader.config.settings import get_remaining_day_fraction, settings
from kalshi_weather_trader.db.schemas import MonteCarloResult

logger = structlog.get_logger(__name__)

_EASTERN = pytz.timezone("America/New_York")
_DT_HOURS = 5.0 / 60.0   # 5-minute steps in hours
_STEPS_PER_HOUR = int(1.0 / _DT_HOURS)   # 12 steps/hour

# NWS reports daily maximum temperature in whole degrees Fahrenheit.
# Kalshi B-bucket cap_strike is INCLUSIVE: B38.5 with cap=39 covers {38°F, 39°F}.
#
# Because NWS rounds to the nearest integer, the settlement boundary between
# consecutive buckets is at the half-integer midpoint, not at the integer itself.
# A continuous MC path of 39.6°F rounds to 40°F → B40.5, not B38.5.
#
# Correct continuous boundaries for integer-settled buckets:
#   Bottom (cap=X):          max < X - 0.5      → CDF boundary at cap - 0.5
#   Middle (floor=X, cap=Y): X-0.5 ≤ max < Y+0.5 → CDF boundaries at floor-0.5, cap+0.5
#   Top    (floor=X):        max ≥ X - 0.5      → CDF boundary at floor - 0.5
_TEMP_RESOLUTION = 1.0   # °F — NWS integer temperature resolution
_HALF_STEP = _TEMP_RESOLUTION / 2.0   # 0.5°F — rounding boundary offset

# ---------------------------------------------------------------------------
# Time-varying sigma block definitions (Issue B1)
# ---------------------------------------------------------------------------
#
# Boston Logan (KBOS) intraday temperature volatility varies ~2-3x across the
# day.  Partitioning into 5 ET-hour blocks lets the MC simulation use a tighter
# sigma in low-volatility periods (overnight, evening) and a wider sigma during
# the active solar-heating window (morning ramp).
#
# Block labels are the string keys used in sigma_by_block dicts throughout the
# system.  SIGMA_BLOCKS is the authoritative list of (start_hour, end_hour) pairs
# where end_hour is exclusive (standard Python half-open interval convention).

SIGMA_BLOCKS: list[tuple[int, int]] = [(0, 6), (6, 10), (10, 14), (14, 18), (18, 24)]
SIGMA_BLOCK_LABELS: list[str] = ["0-6", "6-10", "10-14", "14-18", "18-24"]


def _sigma_block_for_hour(et_hour: float) -> str:
    """Return the sigma block label for a fractional ET hour.

    Args:
        et_hour: Fractional Eastern-time hour in [0, 24).

    Returns:
        Block label string, e.g. '6-10'.

    Raises:
        Nothing.
    """
    h = et_hour % 24.0
    for (start, end), label in zip(SIGMA_BLOCKS, SIGMA_BLOCK_LABELS):
        if start <= h < end:
            return label
    return SIGMA_BLOCK_LABELS[-1]  # fallback: last block (18-24)


# ---------------------------------------------------------------------------
# Monte Carlo params dataclass
# ---------------------------------------------------------------------------


class MCParams:
    """Bundle of parameters for a single Monte Carlo run.

    Attributes:
        T0:          Current temperature estimate (Kalman output, °F).
        hard_floor:  Minimum observed max for the day (°F).
        nwp_curve:   Hourly temperature curve (list of °F values, one per hour).
        bias:        Kalman bias correction to add to nwp_curve.
        theta:       OU mean-reversion speed (per hour).
        sigma:       OU diffusion coefficient (°F/sqrt-hour).
        drift_adj:   Intraday drift adjustment (morning or afternoon).
        hour_offset: Current hour-of-day index into nwp_curve.
        n_paths:     Number of simulation paths.
        day_fraction_remaining: Fraction of day still to simulate.
        is_future_day: True after 6 PM ET rollover; suppresses NWP anchor offset.
    """

    def __init__(
        self,
        T0: float,
        hard_floor: float,
        nwp_curve: list[float],
        bias: float = 0.0,
        theta: Optional[float] = None,
        sigma: Optional[float] = None,
        drift_adj: float = 0.0,
        hour_offset: int = 0,
        n_paths: Optional[int] = None,
        day_fraction_remaining: Optional[float] = None,
        is_future_day: bool = False,
        bridge_steps: int = 0,
        persistence_filter_offset: Optional[float] = None,
        sigma_by_block: Optional[dict[str, float]] = None,
        theta_am: Optional[float] = None,
        theta_pm: Optional[float] = None,
        ou_max_stationary_std: Optional[float] = None,
        use_drift_in_attractor: bool = False,
        ensemble_spread: float = 0.0,
        mean_cloudcover_10_16: float = 50.0,
    ) -> None:
        """Initialise Monte Carlo parameters.

        Args:
            T0:                   Starting temperature in °F.
            hard_floor:           Hard minimum for paths_max in °F.
            nwp_curve:            Hourly NWP temperature curve.
            bias:                 Kalman bias correction in °F.
            theta:                OU mean-reversion. Defaults to settings value.
            sigma:                OU diffusion. Defaults to settings value.
            drift_adj:            Additive intraday drift correction.
            hour_offset:          Current hour index into nwp_curve.
            n_paths:              Number of paths. Defaults to settings value.
            day_fraction_remaining: Fraction of day left. Auto-computed if None.
            is_future_day:        True when simulating the next trading day after
                                  the 6 PM ET rollover. Suppresses the NWP anchor
                                  offset because T0 (tonight's temperature) is from
                                  a different time period than NWP[hour_offset]
                                  (tomorrow's forecast).
            bridge_steps:         Number of simulation steps covering the
                                  pre-observation-window bridge period (tonight's
                                  hours before the NWS window opens at midnight
                                  EST). During these steps paths evolve normally
                                  but paths_max is NOT updated — bridge temperatures
                                  are outside the target trading day's settlement
                                  window and must not contaminate the daily max.
                                  Defaults to 0 (same-day simulation: all steps
                                  count toward paths_max).
            persistence_filter_offset: Expected gap between the ASOS tabular max
                                  and the true NWS daily max (°F). Applied as an
                                  upward offset to hard_floor when initialising
                                  paths_max. Defaults to settings.persistence_filter_offset
                                  (0.3°F). The hard_floor in the DB is never modified.
            sigma_by_block:       Per-ET-hour-block sigma dict, e.g.
                                  {"0-6": 0.25, "6-10": 0.7, ...}. When provided,
                                  run_simulation() precomputes per-step noise from
                                  the appropriate block sigma (with OU cap applied
                                  per-block). Falls back to scalar sigma when None.
            theta_am:             Mean-reversion speed for AM regime (ET hours 6–13).
                                  Physical: solar-forced morning hours have slower
                                  mean-reversion than the convective PM hours. When
                                  provided, overrides scalar theta for steps whose
                                  ET hour falls in [6, 13). Falls back to theta when
                                  None. Expected to be lower than theta_pm.
            theta_pm:             Mean-reversion speed for PM regime (ET hours 13–20).
                                  Physical: convective mixing near peak temperature
                                  acts as a thermostat. When provided, overrides
                                  scalar theta for steps in [13, 20). Falls back to
                                  theta when None. Expected to be higher than theta_am.
            ou_max_stationary_std: Cap on the OU stationary std (sigma / sqrt(2*theta)).
                                  Populated by mc_params_builder from
                                  state.ou_max_stationary_std_calibrated when available;
                                  otherwise falls back to settings.ou_max_stationary_std.
                                  Defaults to settings.ou_max_stationary_std (1.5).
            use_drift_in_attractor: When False (default), drift_adj is excluded from
                                  the OU attractor formula:
                                    mu_t = nwp[h] + anchor_offset + bias
                                  When True, drift_adj is included (legacy behaviour):
                                    mu_t = nwp[h] + anchor_offset + bias + drift_adj
                                  Phase A fix: with H=[[1,1]], Kalman bias already absorbs
                                  the same systematic NWP error that drift_adj was
                                  compensating for. Adding both caused a ~3.2°F inflation
                                  (bias=2.02 + drift=1.15) when the true correction is
                                  ~1.5–2.5°F. drift_adj continues to flow through MCParams
                                  for diagnostic comparison against kalman_bias regardless.
            ensemble_spread:      Std of ensemble member daily highs (°F). When above
                                  settings.ensemble_spread_threshold, sigma is inflated by
                                  settings.ensemble_spread_inflation. 0.0 = no adjustment.
            mean_cloudcover_10_16: Mean NWP cloud cover for ET hours 10-16 (%). Controls
                                  sigma scaling: >80% → ×0.8 (overcast, NWP more accurate);
                                  <20% → ×1.1 (clear, more convective variability);
                                  20-80% → ×1.0 (neutral). Default 50.0 = neutral.

        Returns:
            None

        Raises:
            Nothing.
        """
        self.T0 = T0
        self.hard_floor = hard_floor
        self.nwp_curve = nwp_curve
        self.bias = bias
        self.theta = theta if theta is not None else settings.ou_theta
        self.sigma = sigma if sigma is not None else settings.ou_sigma
        self.drift_adj = drift_adj
        self.hour_offset = hour_offset
        self.n_paths = n_paths if n_paths is not None else settings.mc_n_paths
        self.day_fraction_remaining = (
            day_fraction_remaining
            if day_fraction_remaining is not None
            else get_remaining_day_fraction()
        )
        self.is_future_day = is_future_day
        self.bridge_steps = bridge_steps
        self.persistence_filter_offset = (
            persistence_filter_offset
            if persistence_filter_offset is not None
            else settings.persistence_filter_offset
        )
        self.sigma_by_block = sigma_by_block  # None → use scalar sigma
        self.theta_am = theta_am  # None → use scalar theta for AM hours
        self.theta_pm = theta_pm  # None → use scalar theta for PM hours
        self.ou_max_stationary_std = (
            ou_max_stationary_std
            if ou_max_stationary_std is not None
            else settings.ou_max_stationary_std
        )
        self.use_drift_in_attractor = use_drift_in_attractor
        self.ensemble_spread = ensemble_spread
        self.mean_cloudcover_10_16 = mean_cloudcover_10_16


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------


def run_simulation(params: MCParams, seed: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
    """Run the Ornstein-Uhlenbeck Monte Carlo simulation.

    Args:
        params: ``MCParams`` bundle with all simulation parameters.
        seed:   Optional integer seed for the NumPy random generator. Pass an
                integer for reproducible results (e.g. backtesting). Defaults
                to None (non-deterministic).

    Returns:
        Tuple of (paths_current, paths_max) — both shape (n_paths,).
        ``paths_max`` contains the running maximum of each path, initialised
        at ``params.hard_floor`` to enforce the hard floor guarantee.

    Raises:
        Nothing — uses only NumPy operations.
    """
    n_paths = params.n_paths
    theta = params.theta
    sigma = params.sigma
    dt = _DT_HOURS
    sqrt_dt = np.sqrt(dt)

    # Regime adjustment factor: ensemble spread × cloud cover
    # Applied to sigma BEFORE the OU cap so the cap remains a hard physical ceiling.
    _spread_factor = (
        settings.ensemble_spread_inflation
        if params.ensemble_spread > settings.ensemble_spread_threshold
        else 1.0
    )
    if params.mean_cloudcover_10_16 > settings.cloudcover_overcast_threshold:
        _cloud_factor = settings.cloudcover_overcast_factor
    elif params.mean_cloudcover_10_16 < settings.cloudcover_clear_threshold:
        _cloud_factor = settings.cloudcover_clear_factor
    else:
        _cloud_factor = 1.0
    _regime_factor = _spread_factor * _cloud_factor
    if _regime_factor != 1.0:
        logger.debug(
            "mc.regime_factor",
            ensemble_spread=round(params.ensemble_spread, 2),
            mean_cloudcover=round(params.mean_cloudcover_10_16, 1),
            spread_factor=round(_spread_factor, 3),
            cloud_factor=round(_cloud_factor, 3),
            regime_factor=round(_regime_factor, 3),
        )

    # Scale sigma by regime_factor before the OU cap check
    sigma = sigma * _regime_factor

    # Cap sigma so the OU stationary std stays within a physically meaningful bound.
    #
    # Stationary std of the OU process = sigma / sqrt(2 * theta).
    # With calibrated sigma=1.385 and theta=0.1559 this is 2.48°F — far too large.
    # Result: per-step noise (sigma*sqrt(dt) = 0.4°F) is 31× the restoring force
    # (theta*(mu-T)*dt ≈ 0.013°F at 1°F gap).  The process is a near-random walk
    # at 5-minute resolution; paths spike 5–7°F above the NWP attractor before
    # mean reversion catches up, locking in wildly inflated paths_max values.
    #
    # The cap is: sigma_used = min(sigma, max_stationary_std * sqrt(2 * theta))
    # Physical interpretation: temperature at equilibrium deviates from the NWP
    # attractor by at most max_stationary_std°F (≈ NWP same-day intraday RMSE).
    # Populated from state.ou_max_stationary_std_calibrated when available
    # (Phase 3), otherwise falls back to settings.ou_max_stationary_std (2.0).
    # Tune via OU_MAX_STATIONARY_STD env var before Phase 3 data is available.
    sigma_max = params.ou_max_stationary_std * (2.0 * theta) ** 0.5 if theta > 0 else float("inf")
    _sigma_cap_fired = False
    if theta > 0 and sigma > sigma_max:
        stationary_std_uncapped = sigma / (2.0 * theta) ** 0.5
        noise_uncapped = sigma * np.sqrt(dt)
        restoring_at_1f = theta * 1.0 * dt
        logger.debug(
            "mc.sigma_capped",
            sigma_calibrated=round(sigma, 3),
            sigma_capped=round(sigma_max, 3),
            stationary_std_uncapped=round(stationary_std_uncapped, 2),
            max_stationary_std=params.ou_max_stationary_std,
            noise_restoring_ratio_uncapped=round(noise_uncapped / restoring_at_1f, 1),
        )
        sigma = sigma_max
        _sigma_cap_fired = True
    # Log the effective noise/restoring ratio at the used sigma so cap decisions
    # can be evaluated without having to recompute manually.
    if theta > 0:
        _noise_step = sigma * np.sqrt(dt)
        _restoring = theta * 1.0 * dt
        logger.debug(
            "mc.sigma_effective",
            sigma_used=round(sigma, 3),
            stationary_std=round(sigma / (2.0 * theta) ** 0.5, 2),
            noise_restoring_ratio=round(_noise_step / _restoring, 1),
            cap_active=_sigma_cap_fired,
        )

    # Total simulation steps based on remaining day fraction
    total_hours = 24.0 * params.day_fraction_remaining
    n_steps = max(1, int(total_hours * _STEPS_PER_HOUR))

    # bridge_steps: how many leading steps cover the pre-window bridge period
    # (tonight's temperatures before the NWS observation window opens).
    # Clamped to [0, n_steps - 1] so at least the final step always counts.
    bridge_steps = max(0, min(params.bridge_steps, n_steps - 1))

    # Pre-generate the full random matrix — O(n_steps × n_paths)
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_steps, n_paths))

    # Precompute per-step noise standard deviation (sigma * sqrt(dt)).
    # When sigma_by_block is available, look up the block for each step's ET hour
    # and apply the OU cap independently per block.  This avoids branching inside
    # the hot loop and keeps the vectorised structure intact.
    if params.sigma_by_block:
        step_noise = np.empty(n_steps, dtype=float)
        for s in range(n_steps):
            et_hour = (params.hour_offset + s * _DT_HOURS) % 24.0
            block = _sigma_block_for_hour(et_hour)
            raw_block_sigma = params.sigma_by_block.get(block, params.sigma) * _regime_factor
            capped = min(raw_block_sigma, sigma_max)
            step_noise[s] = capped * sqrt_dt
        logger.debug("mc.sigma_by_block.used", n_blocks=len(params.sigma_by_block))
    else:
        step_noise = np.full(n_steps, sigma * sqrt_dt)

    # Precompute per-step mean-reversion speed (theta).
    # When theta_am or theta_pm is provided, look up the regime for each step's
    # ET hour: AM = [6, 13), PM = [13, 20), otherwise fall back to scalar theta.
    # Precomputation avoids per-step branching in the hot loop.
    if params.theta_am is not None or params.theta_pm is not None:
        step_theta = np.empty(n_steps, dtype=float)
        for s in range(n_steps):
            et_hour = (params.hour_offset + s * _DT_HOURS) % 24.0
            if 6.0 <= et_hour < 13.0 and params.theta_am is not None:
                step_theta[s] = params.theta_am
            elif 13.0 <= et_hour < 20.0 and params.theta_pm is not None:
                step_theta[s] = params.theta_pm
            else:
                step_theta[s] = theta   # scalar fallback for overnight hours
        logger.debug(
            "mc.theta_regime.used",
            theta_am=params.theta_am,
            theta_pm=params.theta_pm,
        )
    else:
        step_theta = np.full(n_steps, theta)  # uniform scalar (backward-compatible)

    # Initialise path arrays.
    # persistence_filter_offset raises the effective hard floor by the expected
    # ASOS-to-NWS gap (default 0.3°F) without modifying the stored DB value.
    paths_current = np.full(n_paths, params.T0, dtype=float)
    effective_floor = params.hard_floor + params.persistence_filter_offset
    paths_max = np.full(n_paths, effective_floor, dtype=float)
    logger.debug(
        "mc.effective_floor",
        hard_floor=params.hard_floor,
        persistence_offset=params.persistence_filter_offset,
        effective_floor=round(effective_floor, 2),
    )

    nwp_len = len(params.nwp_curve)

    # Compute the NWP anchor offset once before the loop.
    # The raw gap (T0 - NWP[hour_offset]) is scaled by a weight that reflects how
    # much of the day has elapsed toward the forecast peak:
    #
    #   peak_hour_idx  = argmax of the WINDOW portion of nwp_curve
    #                    (first index >= bridge_hours / STEPS_PER_HOUR, so that
    #                    pre-window bridge temperatures do not dominate the peak
    #                    search and force anchor_weight to 1.0 prematurely)
    #   hours_to_peak  = max(0, peak_hour_idx - hour_offset)
    #   anchor_weight  = 1 - hours_to_peak / peak_hour_idx   (0 at day start → 1 at peak)
    #
    # Early morning (far from peak): weight ≈ 0 → near-zero offset → OU warms naturally
    #   toward the NWP curve.  A gap at 9 AM is mostly just the day not yet having warmed;
    #   applying the full offset would depress the entire forecast by the morning shortfall.
    # At/past peak: weight = 1 → full offset applied → if T0 is still below NWP by peak
    #   time, that is genuine evidence of a cooler-than-forecast day.
    # Positive offset (T0 > NWP): weight scales it the same way — an above-NWP reading
    #   at 9 AM is weakly informative; the same gap at 2 PM is strongly informative.
    #
    # Bridge handling: when bridge_steps > 0 (stitched post-rollover simulation), tonight's
    # pre-window temperatures are the first elements of the curve and are typically the
    # highest (declining front scenario).  Using argmax over the full curve would set
    # peak_hour_idx = 0 → anchor_weight = 1.0 everywhere → constant upward shift that
    # inflates the window-period distribution.  Restricting the search to the window
    # portion (bridge_hours onward) gives a peak that is physically meaningful for the
    # next day's settlement.
    #
    # As Kalman bias converges over weeks, systematic NWP errors are absorbed into bias
    # and T0 ≈ NWP[hour_offset], so the offset naturally trends toward zero.
    #
    # Formula (use_drift_in_attractor=False): mu_t = nwp_curve[h] + anchor_offset + bias
    if nwp_len > 0 and params.hour_offset < nwp_len:
        nwp_reference = params.nwp_curve[params.hour_offset]
        # Search for the peak within the window portion only (from bridge onward).
        window_start_hour = bridge_steps // _STEPS_PER_HOUR
        if window_start_hour > 0 and window_start_hour < nwp_len:
            peak_hour_idx = window_start_hour + int(np.argmax(params.nwp_curve[window_start_hour:]))
        else:
            peak_hour_idx = int(np.argmax(params.nwp_curve))
        hours_to_peak = max(0, peak_hour_idx - params.hour_offset)
        anchor_weight = (1.0 - hours_to_peak / peak_hour_idx) if peak_hour_idx > 0 else 1.0
    else:
        nwp_reference = params.T0
        anchor_weight = 1.0
    # Subtract Kalman bias from the gap before scaling by anchor_weight.
    # With H=[[1,1]], T0 = nwp_current + dT + B, so the raw gap contains both
    # the transient departure dT and the persistent bias B.  But params.bias = B
    # is already added to mu_t separately (line: mu_t = nwp_curve[...] + offset + bias + drift).
    # Without this correction the bias is double-counted: once via offset, once via params.bias.
    # Fix: anchor offset only captures the residual gap after Kalman correction.
    raw_gap = params.T0 - nwp_reference
    gap_after_bias = raw_gap - params.bias
    nwp_anchor_offset = gap_after_bias * anchor_weight

    if params.is_future_day:
        # Pre-market simulation: no intraday observations exist yet for the
        # target date.  T0 is tonight's temperature, not tomorrow's; comparing
        # it to NWP[hour_offset] produces a physically meaningless gap.
        # Kalman bias already carries forward systematic NWP error.
        nwp_anchor_offset = 0.0

    # Phase A: drift_adj is excluded from the attractor by default.
    # With H=[[1,1]], Kalman bias absorbs the same systematic NWP error that
    # drift_adj was compensating for; adding both inflated mu_t by ~3.2°F on
    # March 29 (bias=2.02 + drift=1.15), pushing P(YES) to 100%.
    # drift_adj is kept in MCParams for diagnostic comparison vs kalman_bias.
    # Set use_drift_in_attractor=True only for A/B backtesting.
    _drift_term = params.drift_adj if params.use_drift_in_attractor else 0.0

    for step in range(n_steps):
        # Current hour index (with offset for time-of-day)
        hour_idx = min(
            params.hour_offset + int(step / _STEPS_PER_HOUR),
            nwp_len - 1,
        ) if nwp_len > 0 else 0

        # Mean-reversion target: NWP + anchor offset + Kalman bias [+ drift if flag set]
        if nwp_len > 0:
            mu_t = params.nwp_curve[hour_idx] + nwp_anchor_offset + params.bias + _drift_term
        else:
            # If no NWP curve, revert toward current estimate
            mu_t = params.T0 + params.bias + _drift_term

        # OU step: dT = theta_step*(mu_t - T_t)*dt + sigma_block*sqrt(dt)*Z
        dT = step_theta[step] * (mu_t - paths_current) * dt + step_noise[step] * Z[step]
        paths_current = paths_current + dT

        # Update running maximum only after the bridge period ends.
        # During bridge_steps the simulation warms up the path to the correct
        # temperature at the NWS window boundary (midnight EST) without letting
        # pre-window readings inflate the daily max for the next trading day.
        if step >= bridge_steps:
            paths_max = np.maximum(paths_max, paths_current)

    return paths_current, paths_max


def price_full_distribution(
    params: MCParams,
    strikes: list[float],
    target_date: Optional[date] = None,
    seed: Optional[int] = None,
) -> MonteCarloResult:
    """Run one simulation and price all strikes from the resulting path distribution.

    The simulation is run once; all strike probabilities are computed from the
    same ``paths_max`` array, making this O(n_paths + n_strikes) rather than
    O(n_paths × n_strikes).

    Args:
        params:      ``MCParams`` with all simulation parameters.
        strikes:     List of integer strike temperatures (°F) to price.
        target_date: Trading date for the result document. Defaults to today.
        seed:        Optional integer seed for reproducible results (e.g.
                     backtesting). Passed through to ``run_simulation()``.

    Returns:
        ``MonteCarloResult`` with probabilities and distribution statistics.

    Raises:
        Nothing — errors are logged and a default result is returned.
    """
    from kalshi_weather_trader.config.settings import get_target_date

    if target_date is None:
        target_date = get_target_date()

    try:
        _paths_current, paths_max = run_simulation(params, seed=seed)

        # Compute P(paths_max >= strike) for every strike
        probs: dict[float, float] = {}
        for strike in strikes:
            p = float(np.mean(paths_max >= float(strike)))
            probs[strike] = round(p, 6)

        n_steps = max(1, int(24.0 * params.day_fraction_remaining * _STEPS_PER_HOUR))

        result = MonteCarloResult(
            target_date=target_date,
            computed_at_utc=datetime.now(timezone.utc),
            n_paths=params.n_paths,
            n_steps=n_steps,
            hard_floor=params.hard_floor,
            probabilities=probs,
            percentile_10=float(np.percentile(paths_max, 10)),
            percentile_25=float(np.percentile(paths_max, 25)),
            percentile_50=float(np.percentile(paths_max, 50)),
            percentile_75=float(np.percentile(paths_max, 75)),
            percentile_90=float(np.percentile(paths_max, 90)),
            mean_max=float(np.mean(paths_max)),
            std_max=float(np.std(paths_max)),
        )

        logger.info(
            "mc.price_distribution.done",
            n_paths=params.n_paths,
            n_steps=n_steps,
            hard_floor=params.hard_floor,
            mean_max=round(result.mean_max, 2),
            std_max=round(result.std_max, 2),
            strikes=strikes,
        )
        return result

    except Exception as exc:
        logger.error("mc.price_distribution.failed", error=str(exc), exc_info=True)
        # Return a safe default (all 50% — neutral)
        return MonteCarloResult(
            target_date=target_date,
            n_paths=params.n_paths,
            n_steps=1,
            hard_floor=params.hard_floor,
            probabilities={s: 0.5 for s in strikes},
            percentile_10=params.T0,
            percentile_25=params.T0,
            percentile_50=params.T0,
            percentile_75=params.T0,
            percentile_90=params.T0,
            mean_max=params.T0,
            std_max=0.0,
        )


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------


def _interpolate_cdf(probs: dict[float, float], temp: float) -> float:
    """Return P(paths_max >= temp) via linear interpolation of the MC CDF.

    If temp is exactly in probs, returns that value directly.
    If temp is below all keys, returns 1.0 (certainly exceeded).
    If temp is above all keys, returns 0.0 (certainly not exceeded).
    Otherwise, linearly interpolates between nearest lower and upper keys.

    Args:
        probs: Dict mapping strike → P(max >= strike), from price_full_distribution.
        temp:  Temperature in °F to look up.

    Returns:
        Float in [0.0, 1.0].

    Raises:
        Nothing.
    """
    if not probs:
        return 0.5

    if temp in probs:
        return probs[temp]

    sorted_keys = sorted(probs.keys())

    if temp <= sorted_keys[0]:
        return 1.0

    if temp >= sorted_keys[-1]:
        return 0.0

    # Find the bracketing keys
    lo = sorted_keys[0]
    hi = sorted_keys[-1]
    for k in sorted_keys:
        if k <= temp:
            lo = k
        else:
            hi = k
            break

    # Linear interpolation: CDF is monotone decreasing in temp
    alpha = (temp - lo) / (hi - lo)
    return probs[lo] * (1.0 - alpha) + probs[hi] * alpha


def compute_yes_prob(
    cumulative_probs: dict[float, float],
    floor_raw: Optional[float],
    cap_raw: Optional[float],
) -> float:
    """Convert raw cumulative exceedance probabilities to a market-correct P(YES).

    Uses the Kalshi API ``floor_strike`` / ``cap_strike`` fields to determine
    which of three market semantics applies:

    - Bottom bucket (floor=None, cap=X):  YES if daily max < X  → 1 − P(max≥X)
    - Middle bucket (floor=X, cap=Y):     YES if X ≤ max < Y    → P(max≥X) − P(max≥Y)
    - Top bucket   (floor=X, cap=None):   YES if max ≥ X        → P(max≥X)

    Uses CDF interpolation so that boundaries not explicitly in cumulative_probs
    are handled correctly rather than falling back to arbitrary defaults.

    Args:
        cumulative_probs: Dict mapping temperature → P(paths_max >= temperature)
                          from ``price_full_distribution``.
        floor_raw:        API ``floor_strike`` value in °F, or None for bottom bucket.
        cap_raw:          API ``cap_strike`` value in °F, or None for top bucket.

    Returns:
        Float in [0.0, 1.0] representing the market-correct P(YES).

    Raises:
        Nothing.
    """
    floor_f = float(floor_raw) if floor_raw is not None else None
    cap_f = float(cap_raw) if cap_raw is not None else None

    if floor_f is None and cap_f is not None:
        # Bottom bucket: YES if integer max < cap (e.g. T38 → max ≤ 37°F).
        # NWS rounding: continuous paths_max < cap - 0.5 settle to integers < cap.
        p_cap = _interpolate_cdf(cumulative_probs, cap_f - _HALF_STEP)
        return max(0.0, min(1.0, 1.0 - p_cap))

    if floor_f is not None and cap_f is None:
        # Top bucket: YES if integer max ≥ floor + 1 (e.g. T45 → max ≥ 46°F).
        # NWS rounding: continuous paths_max ≥ floor + 0.5 settle to integers ≥ floor + 1.
        return max(0.0, min(1.0, _interpolate_cdf(cumulative_probs, floor_f + _HALF_STEP)))

    if floor_f is not None and cap_f is not None:
        # Middle bucket: YES if floor ≤ integer max ≤ cap (both inclusive per Kalshi API).
        # e.g. B38.5 (floor=38, cap=39) covers {38°F, 39°F}.
        # NWS rounding: continuous boundary is [floor-0.5, cap+0.5).
        # 39.6°F rounds to 40 → B40.5, not B38.5, so upper bound is cap+0.5 not cap+1.
        p_floor = _interpolate_cdf(cumulative_probs, floor_f - _HALF_STEP)
        p_cap = _interpolate_cdf(cumulative_probs, cap_f + _HALF_STEP)
        return max(0.0, min(1.0, p_floor - p_cap))

    # Both None — fallback (should not occur with valid Kalshi market data)
    return 0.5


def compute_normalized_market_probs(
    markets: list[dict],
    cumulative_probs: dict[float, float],
) -> tuple[dict[str, float], float, list[tuple[float, float, float]]]:
    """Compute P(YES) for each market with normalization and partition diagnostics.

    Computes raw P(YES) for each market using compute_yes_prob, then:
    - Verifies partition completeness by checking adjacency (cap[i] == floor[i+1])
    - Logs any gaps with their probability mass
    - Normalizes probabilities to sum to 1.0 if |Σ - 1.0| ≤ 0.10
    - Logs a loud warning if |Σ - 1.0| > 0.10 (structural problem)

    Args:
        markets: List of normalised Kalshi market dicts (with floor_strike,
                 cap_strike, ticker).
        cumulative_probs: Dict from price_full_distribution: strike → P(max >= strike).

    Returns:
        Tuple of:
          - Dict mapping ticker → normalized P(YES) float in [0.0, 1.0]
          - Raw sum (pre-normalization) as a diagnostic float
          - List of gap tuples: (gap_low_temp, gap_high_temp, gap_probability)

    Raises:
        Nothing.
    """
    # Step 1: compute raw probabilities per ticker
    raw_probs: dict[str, float] = {}
    for m in markets:
        ticker = m.get("ticker", "")
        if not ticker:
            continue
        floor_raw = m.get("floor_strike")
        cap_raw = m.get("cap_strike")
        raw_probs[ticker] = compute_yes_prob(cumulative_probs, floor_raw, cap_raw)

    sum_raw = sum(raw_probs.values())

    # Step 2: detect partition gaps by sorting markets by their lower boundary
    def _floor_key(m: dict) -> float:
        f = m.get("floor_strike")
        return float(f) if f is not None else float("-inf")

    sorted_markets = sorted(
        [m for m in markets if m.get("ticker")],
        key=_floor_key,
    )

    gaps: list[tuple[float, float, float]] = []
    for i in range(len(sorted_markets) - 1):
        m_cur = sorted_markets[i]
        cap_i = m_cur.get("cap_strike")
        floor_next = sorted_markets[i + 1].get("floor_strike")

        if cap_i is None or floor_next is None:
            continue

        cap_f = float(cap_i)
        floor_f = float(floor_next)

        # Bottom bucket (floor=None) has an exclusive cap → next expected floor = cap.
        # Middle→top: top bucket covers integers ≥ floor+1, continuous boundary at floor+0.5,
        # which equals the preceding middle bucket's cap+0.5 — so expected next floor = cap.
        # Middle→middle: inclusive cap → next expected floor = cap + 1.
        is_bottom = m_cur.get("floor_strike") is None
        is_next_top = sorted_markets[i + 1].get("cap_strike") is None
        if is_bottom or is_next_top:
            expected_next_floor = cap_f
        else:
            expected_next_floor = cap_f + _TEMP_RESOLUTION

        if abs(expected_next_floor - floor_f) > 1e-6:
            # Gap between the exclusive upper bound of market i and floor of market i+1
            gap_prob = abs(
                _interpolate_cdf(cumulative_probs, expected_next_floor)
                - _interpolate_cdf(cumulative_probs, floor_f)
            )
            gaps.append((expected_next_floor, floor_f, gap_prob))
            logger.warning(
                "mc.partition.gap_detected",
                gap_low=expected_next_floor,
                gap_high=floor_f,
                gap_prob=round(gap_prob, 4),
            )

    # Step 3: normalize or warn
    deviation = abs(sum_raw - 1.0)

    if deviation <= 0.05:
        # Tier 1: normal — normalize silently
        normalized = {t: p / sum_raw for t, p in raw_probs.items()} if sum_raw > 0 else dict(raw_probs)
        logger.debug("mc.partition.normalized", sum_raw=round(sum_raw, 4), n_markets=len(raw_probs))

    elif deviation <= 0.10:
        # Tier 2: suspicious — normalize but warn
        normalized = {t: p / sum_raw for t, p in raw_probs.items()} if sum_raw > 0 else dict(raw_probs)
        logger.warning("mc.partition.sum_suspicious", sum_raw=round(sum_raw, 4),
                       n_markets=len(raw_probs), n_gaps=len(gaps))

    else:
        # Tier 3: structural problem — do NOT normalize, log ERROR
        logger.error("mc.partition.sum_severely_wrong", sum_raw=round(sum_raw, 4),
                     n_markets=len(raw_probs), n_gaps=len(gaps))
        normalized = dict(raw_probs)

    return normalized, sum_raw, gaps


def _interp_nwp(curve: list[float], hour_frac: float) -> float:
    """Linearly interpolate an NWP hourly curve at fractional ET hour.

    Args:
        curve:     Hourly temps (°F), index = ET hour.
        hour_frac: Fractional ET hour, e.g. 9.5 = 9:30 AM.

    Returns:
        Interpolated °F, clamped to curve bounds.

    Raises:
        Nothing.
    """
    if not curve:
        return 0.0
    hour_frac = max(0.0, min(float(len(curve) - 1), hour_frac))
    lo = int(hour_frac)
    hi = min(lo + 1, len(curve) - 1)
    alpha = hour_frac - lo
    return curve[lo] * (1.0 - alpha) + curve[hi] * alpha


def estimate_sigma_from_historical(
    readings: list,
    nwp_curves: Optional[dict[date, list[float]]] = None,
    decay_tau_days: Optional[int] = None,
) -> tuple[float, dict[str, float]]:
    """Estimate the OU diffusion coefficient (sigma) from ASOS history.

    Uses hourly-bucket temperature differences instead of consecutive 5-minute
    diffs. The ASOS 0.5°C persistence filter causes sensor readings to jump in
    discrete 0.9°F increments; at 5-minute resolution these jumps inflate
    mean(dT²/dt) by 3-4× relative to true temperature volatility. By bucketing
    to the nearest top-of-hour (same approach as ``calibrate_theta``), each
    difference spans a full hour and averages through multiple sensor steps,
    recovering the true hourly volatility.

    Also computes per-ET-hour-block sigmas (Issue B1: time-varying sigma).
    The day is partitioned into SIGMA_BLOCKS; each block uses only the dT
    residuals whose source hour falls within that block. Blocks with fewer than
    10 samples fall back to the pooled sigma.

    Algorithm:
        1. Group readings by ET date.
        2. For each date, find the reading nearest to each top-of-hour; skip
           hours with no reading within 40 minutes (data gap guard).
        3. For consecutive within-day hour pairs, compute dT = T[h+1] - T[h].
        4. Detrend by NWP when available: dT_used = dT - (nwp[h+1] - nwp[h]).
           Cross-midnight pairs are skipped (NWP detrend unreliable).
        5. contribution = dT_used² / 1.0  (dt = 1 hour exactly)
        6. sigma = sqrt(mean(contributions))
        7. sigma_by_block = {label: sqrt(mean(block_contributions))}

    Args:
        readings:        List of ``ASOSReadingDocument`` objects, oldest-first.
                         Must have ``observation_time_utc`` and ``temperature_f``.
        nwp_curves:      Optional dict mapping date → hourly NWP curve (ET-indexed).
                         If None or a date is missing, falls back to raw dT for that
                         interval.
        decay_tau_days:  Exponential decay time constant (days) for weighting.
                         Each day's contribution is scaled by exp(-d/tau) where d is
                         the number of days before the most recent date in readings.
                         None → use settings.calibration_decay_tau_days. Set to a
                         very large value (e.g. 10000) for effectively flat weighting.

    Returns:
        Tuple of:
          - Pooled sigma in °F / sqrt-hour, clamped to [0.1, 1.5].
            Returns (settings.ou_sigma, {}) if fewer than 3 valid pairs.
          - sigma_by_block dict mapping block label → sigma (°F/sqrt-hour).
            Any block with < 10 samples (unweighted count) gets pooled sigma.

    Raises:
        Nothing.
    """
    import math as _math

    _SIGMA_CAP = 1.5   # physical ceiling for Boston Logan intraday volatility
    _SIGMA_FLOOR = 0.05  # physical floor for per-block sigma
    _GAP_GUARD_MIN = 40  # max minutes from top-of-hour to accept a reading
    _MIN_BLOCK_SAMPLES = 10  # minimum samples to trust a per-block estimate

    tau = decay_tau_days if decay_tau_days is not None else settings.calibration_decay_tau_days

    if len(readings) < 3:
        logger.warning("mc.estimate_sigma.insufficient_data", n=len(readings))
        return settings.ou_sigma, {}

    # Group readings by ET date.
    by_date: dict[date, list] = {}
    for r in readings:
        d = r.observation_time_utc.astimezone(_EASTERN).date()
        by_date.setdefault(d, []).append(r)

    # Most recent date in the dataset — weights computed relative to this.
    max_date = max(by_date.keys())

    # Exponentially weighted accumulators (replace flat list of dT² values).
    # weighted_sum_sq: Σ weight * dT²  (pooled and per-block)
    # total_weight:    Σ weight          (for normalisation)
    # n_pairs:         unweighted count  (for the _MIN_BLOCK_SAMPLES guard)
    pooled_weighted_sum_sq: float = 0.0
    pooled_total_weight: float = 0.0
    pooled_n_pairs: int = 0
    block_weighted_sum_sq: dict[str, float] = {lbl: 0.0 for lbl in SIGMA_BLOCK_LABELS}
    block_total_weight: dict[str, float] = {lbl: 0.0 for lbl in SIGMA_BLOCK_LABELS}
    block_n_pairs: dict[str, int] = {lbl: 0 for lbl in SIGMA_BLOCK_LABELS}

    n_detrended = 0
    n_raw = 0
    days_used = 0

    for d, day_readings in sorted(by_date.items()):
        # Day weight: recent days count more, older days count less.
        days_back = (max_date - d).days
        day_weight = _math.exp(-days_back / tau)
        # Bucket to hourly: for each ET hour 0-23, pick the reading nearest
        # to the top of that hour (within the gap guard).
        hourly_temp: dict[int, float] = {}
        for hour_et in range(24):
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
            gap_min = abs(
                (best.observation_time_utc - top_of_hour_et).total_seconds()
            ) / 60.0
            if gap_min <= _GAP_GUARD_MIN:
                hourly_temp[hour_et] = best.temperature_f

        # Compute dT between consecutive within-day hours.
        sorted_hours = sorted(hourly_temp.keys())
        if len(sorted_hours) < 2:
            continue
        days_used += 1

        for i in range(len(sorted_hours) - 1):
            h0, h1 = sorted_hours[i], sorted_hours[i + 1]
            if h1 - h0 != 1:
                continue  # non-consecutive hours — skip gap
            dT = hourly_temp[h1] - hourly_temp[h0]
            dT_used = dT

            if nwp_curves is not None and d in nwp_curves:
                curve = nwp_curves[d]
                if len(curve) >= h1 + 1:
                    nwp_dT = curve[h1] - curve[h0]
                    dT_used = dT - nwp_dT
                    n_detrended += 1
                else:
                    n_raw += 1
            else:
                n_raw += 1

            sq = dT_used ** 2  # dt = 1.0 hour
            # Accumulate with exponential day weight.
            pooled_weighted_sum_sq += day_weight * sq
            pooled_total_weight += day_weight
            pooled_n_pairs += 1
            # Assign to the block of the source hour h0.
            block_label = _sigma_block_for_hour(float(h0))
            block_weighted_sum_sq[block_label] += day_weight * sq
            block_total_weight[block_label] += day_weight
            block_n_pairs[block_label] += 1

    if pooled_n_pairs < 3:
        logger.warning(
            "mc.estimate_sigma.insufficient_valid_intervals",
            valid=pooled_n_pairs,
            days_used=days_used,
        )
        return settings.ou_sigma, {}

    # Weighted pooled sigma: sqrt(Σ(w*dT²) / Σ(w))
    sigma = float(np.sqrt(pooled_weighted_sum_sq / pooled_total_weight))
    sigma = max(0.1, min(_SIGMA_CAP, round(sigma, 3)))

    # Per-block sigmas using same weighted formula.
    # Fall back to pooled sigma for blocks with fewer than _MIN_BLOCK_SAMPLES
    # unweighted pairs (insufficient data for reliable block estimate).
    sigma_by_block: dict[str, float] = {}
    for lbl in SIGMA_BLOCK_LABELS:
        if block_n_pairs[lbl] >= _MIN_BLOCK_SAMPLES:
            block_sigma = float(np.sqrt(block_weighted_sum_sq[lbl] / block_total_weight[lbl]))
            block_sigma = max(_SIGMA_FLOOR, min(_SIGMA_CAP, round(block_sigma, 3)))
        else:
            block_sigma = sigma  # fall back to pooled
        sigma_by_block[lbl] = block_sigma

    logger.info(
        "mc.estimate_sigma.done",
        sigma=sigma,
        sigma_by_block=sigma_by_block,
        n_valid=pooled_n_pairs,
        n_detrended=n_detrended,
        n_raw=n_raw,
        days_used=days_used,
        decay_tau_days=tau,
    )
    return sigma, sigma_by_block
