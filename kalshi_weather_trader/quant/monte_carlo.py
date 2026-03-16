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
import structlog

from kalshi_weather_trader.config.settings import get_remaining_day_fraction, settings
from kalshi_weather_trader.db.schemas import MonteCarloResult

logger = structlog.get_logger(__name__)

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


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------


def run_simulation(params: MCParams) -> tuple[np.ndarray, np.ndarray]:
    """Run the Ornstein-Uhlenbeck Monte Carlo simulation.

    Args:
        params: ``MCParams`` bundle with all simulation parameters.

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

    # Total simulation steps based on remaining day fraction
    total_hours = 24.0 * params.day_fraction_remaining
    n_steps = max(1, int(total_hours * _STEPS_PER_HOUR))

    # Pre-generate the full random matrix — O(n_steps × n_paths)
    rng = np.random.default_rng()
    Z = rng.standard_normal((n_steps, n_paths))

    # Initialise path arrays
    paths_current = np.full(n_paths, params.T0, dtype=float)
    paths_max = np.full(n_paths, params.hard_floor, dtype=float)  # hard floor init

    nwp_len = len(params.nwp_curve)

    # Compute the NWP anchor offset once before the loop.
    # This re-anchors the NWP trajectory to T0 so the attractor at step 0 equals
    # T0 exactly, and subsequent steps follow the NWP's *rate of change* rather than
    # its absolute level.  Without this, a cold-start Kalman bias of 0.0 causes the OU
    # process to immediately pull every path from T0 up toward the raw NWP value,
    # inflating paths_max by the gap between T0 and NWP[hour_offset].
    #
    # Formula: mu_t = nwp_curve[hour_idx] + nwp_anchor_offset + bias + drift_adj
    #          where nwp_anchor_offset = T0 - nwp_curve[hour_offset]
    # As kalman_B converges over time, bias absorbs the systematic NWP error and
    # (T0 - nwp_curve[hour_offset]) trends toward zero naturally.
    if nwp_len > 0 and params.hour_offset < nwp_len:
        nwp_reference = params.nwp_curve[params.hour_offset]
    else:
        nwp_reference = params.T0
    nwp_anchor_offset = params.T0 - nwp_reference

    for step in range(n_steps):
        # Current hour index (with offset for time-of-day)
        hour_idx = min(
            params.hour_offset + int(step / _STEPS_PER_HOUR),
            nwp_len - 1,
        ) if nwp_len > 0 else 0

        # Mean-reversion target: NWP delta from anchor + T0 + Kalman bias + intraday drift
        if nwp_len > 0:
            mu_t = params.nwp_curve[hour_idx] + nwp_anchor_offset + params.bias + params.drift_adj
        else:
            # If no NWP curve, revert toward current estimate
            mu_t = params.T0 + params.bias + params.drift_adj

        # OU step: dT = theta*(mu_t - T_t)*dt + sigma*sqrt(dt)*Z
        dT = theta * (mu_t - paths_current) * dt + sigma * sqrt_dt * Z[step]
        paths_current = paths_current + dT

        # Update running maximum (element-wise)
        paths_max = np.maximum(paths_max, paths_current)

    return paths_current, paths_max


def price_full_distribution(
    params: MCParams,
    strikes: list[float],
    target_date: Optional[date] = None,
) -> MonteCarloResult:
    """Run one simulation and price all strikes from the resulting path distribution.

    The simulation is run once; all strike probabilities are computed from the
    same ``paths_max`` array, making this O(n_paths + n_strikes) rather than
    O(n_paths × n_strikes).

    Args:
        params:      ``MCParams`` with all simulation parameters.
        strikes:     List of integer strike temperatures (°F) to price.
        target_date: Trading date for the result document. Defaults to today.

    Returns:
        ``MonteCarloResult`` with probabilities and distribution statistics.

    Raises:
        Nothing — errors are logged and a default result is returned.
    """
    from kalshi_weather_trader.config.settings import get_target_date

    if target_date is None:
        target_date = get_target_date()

    try:
        _paths_current, paths_max = run_simulation(params)

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
    if abs(sum_raw - 1.0) <= 0.10:
        if sum_raw > 0:
            normalized = {t: p / sum_raw for t, p in raw_probs.items()}
        else:
            normalized = dict(raw_probs)
        logger.debug(
            "mc.partition.normalized",
            sum_raw=round(sum_raw, 4),
            n_markets=len(raw_probs),
        )
    else:
        logger.error(
            "mc.partition.sum_severely_wrong",
            sum_raw=round(sum_raw, 4),
            n_markets=len(raw_probs),
            n_gaps=len(gaps),
        )
        normalized = dict(raw_probs)

    return normalized, sum_raw, gaps


def estimate_sigma_from_historical(readings: list) -> float:
    """Estimate the OU diffusion coefficient (sigma) from ASOS history.

    Time-normalises each consecutive temperature difference by the actual
    elapsed time between readings, so gaps caused by app downtime or missing
    data do not inflate the estimate.  Intervals wider than 30 minutes are
    excluded — they represent outages, not real temperature volatility.

    Formula per valid interval i:
        contribution_i = (dT_i)^2 / dt_i_hours
    sigma = sqrt(mean(contributions))   (units: °F / sqrt-hour)

    Args:
        readings: List of ``ASOSReadingDocument`` objects, oldest-first.
                  Must have ``observation_time_utc`` and ``temperature_f``.

    Returns:
        Estimated sigma in °F / sqrt-hour, clamped to [0.1, 4.0].
        Returns settings.ou_sigma if fewer than 3 valid intervals remain
        after gap filtering.

    Raises:
        Nothing.
    """
    _MAX_GAP_HOURS = 0.5   # ignore intervals > 30 minutes (app downtime)
    _SIGMA_CAP = 4.0        # physical ceiling: ~3 std devs above typical Boston value

    if len(readings) < 3:
        logger.warning("mc.estimate_sigma.insufficient_data", n=len(readings))
        return settings.ou_sigma

    contributions: list[float] = []
    skipped = 0

    for i in range(1, len(readings)):
        dt_hours = (
            readings[i].observation_time_utc - readings[i - 1].observation_time_utc
        ).total_seconds() / 3600.0

        if dt_hours <= 0 or dt_hours > _MAX_GAP_HOURS:
            skipped += 1
            continue

        dT = readings[i].temperature_f - readings[i - 1].temperature_f
        contributions.append(dT ** 2 / dt_hours)

    if len(contributions) < 3:
        logger.warning(
            "mc.estimate_sigma.insufficient_valid_intervals",
            total=len(readings) - 1,
            skipped=skipped,
            valid=len(contributions),
        )
        return settings.ou_sigma

    sigma = float(np.sqrt(np.mean(contributions)))
    sigma = max(0.1, min(_SIGMA_CAP, round(sigma, 3)))

    logger.info(
        "mc.estimate_sigma.done",
        sigma=sigma,
        n_valid=len(contributions),
        n_skipped=skipped,
    )
    return sigma
