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

    for step in range(n_steps):
        # Current hour index (with offset for time-of-day)
        hour_idx = min(
            params.hour_offset + int(step / _STEPS_PER_HOUR),
            nwp_len - 1,
        ) if nwp_len > 0 else 0

        # Mean-reversion target: NWP + Kalman bias + intraday drift
        if nwp_len > 0:
            mu_t = params.nwp_curve[hour_idx] + params.bias + params.drift_adj
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

    This avoids relying on the T/B ticker prefix which is ambiguous: both the
    bottom bucket (T38 → "<38°F") and the top bucket (T45 → ">45°F") use the "T"
    prefix but require opposite probability directions.

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
        # Bottom bucket: YES if max < cap (e.g. T38 → "<38°F")
        p_cap = cumulative_probs.get(cap_f, 0.5)
        return max(0.0, min(1.0, 1.0 - p_cap))

    if floor_f is not None and cap_f is None:
        # Top bucket: YES if max >= floor (e.g. T45 → ">45°F")
        return max(0.0, min(1.0, cumulative_probs.get(floor_f, 0.5)))

    if floor_f is not None and cap_f is not None:
        # Middle bucket: YES if floor <= max < cap (e.g. B40.5 → "40.5–42.5°F")
        p_floor = cumulative_probs.get(floor_f, 0.5)
        p_cap = cumulative_probs.get(cap_f, 0.0)
        return max(0.0, min(1.0, p_floor - p_cap))

    # Both None — fallback (should not occur with valid Kalshi market data)
    return 0.5


def estimate_sigma_from_historical(readings: list) -> float:
    """Estimate the OU diffusion coefficient (sigma) from ASOS history.

    Computes the standard deviation of 5-minute temperature differences and
    annualises to per-sqrt-hour units:  sigma = std(diffs) * sqrt(12)

    Args:
        readings: List of ``ASOSReadingDocument`` objects, oldest-first.

    Returns:
        Estimated sigma in °F / sqrt-hour.  Returns settings.ou_sigma if
        insufficient data (< 3 readings).

    Raises:
        Nothing.
    """
    if len(readings) < 3:
        logger.warning("mc.estimate_sigma.insufficient_data", n=len(readings))
        return settings.ou_sigma

    temps = np.array([r.temperature_f for r in readings], dtype=float)
    diffs = np.diff(temps)
    sigma = float(np.std(diffs) * np.sqrt(_STEPS_PER_HOUR))
    sigma = max(0.1, round(sigma, 3))  # enforce a floor to avoid zero sigma

    logger.info("mc.estimate_sigma.done", sigma=sigma, n_diffs=len(diffs))
    return sigma
