"""
Scenario system for the KBOS Model Lab (Tab 6).

A Scenario is a complete set of parameter overrides for a backtest replay run.
The replay engine starts from the historically-calibrated MCParams for each date
and applies whatever overrides the Scenario specifies.  Fields left at their
sentinel value (None for scalars, default for flags) mean "use the historical
calibrated value" — so Scenario(name="Production") with all defaults is a true
production replay.

Classes
-------
Scenario
    Dataclass of parameter overrides.  Custom __hash__/__eq__ make it usable
    as a Streamlit @st.cache_data key despite containing dict/list fields.

ReplayDataCache
    Bulk-loads ASOS, system_state, and market data for a set of settled dates
    so that replay_single() does not issue one DB round-trip per hour.

Functions
---------
preset_production, preset_pre_phase_a, preset_no_drift, preset_no_anchor,
preset_half_anchor, preset_no_corrections, preset_tight_sigma_cap,
preset_flat_sigma, preset_flat_theta, preset_cloud_cover,
preset_ensemble_spread, preset_aggressive_corrections, preset_conservative

ALL_PRESETS   : list of all preset Scenario instances (for CLI / testing)
PRESET_MAP    : OrderedDict[name → Scenario] for Streamlit selectbox
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from datetime import date, datetime, timezone
from typing import Optional

import pytz
import structlog

from kalshi_weather_trader.db.db_manager import (
    get_asos_readings_for_date,
    get_market,
    get_system_state,
)
from kalshi_weather_trader.db.schemas import (
    ASOSReadingDocument,
    MarketDocument,
    SystemStateDocument,
)

logger = structlog.get_logger(__name__)

_EASTERN = pytz.timezone("America/New_York")


# ---------------------------------------------------------------------------
# Scenario dataclass
# ---------------------------------------------------------------------------


@dataclass
class Scenario:
    """A complete set of parameter overrides for a backtest replay run.

    All scalar overrides default to None, meaning "use the historically
    calibrated value for each date".  Structural toggles default to the
    current production configuration.

    The custom __hash__ / __eq__ make Scenario usable as a Streamlit
    @st.cache_data key despite containing dict and list fields.
    """

    name: str

    # ------------------------------------------------------------------
    # Structural toggles
    # ------------------------------------------------------------------
    use_drift_in_attractor: bool = False        # Phase A default: off
    use_anchor_offset: bool = True
    use_time_varying_sigma: bool = True
    use_time_varying_theta: bool = True
    use_cloud_cover_adjustment: bool = False
    use_ensemble_spread_adjustment: bool = False
    use_persistence_offset: bool = True

    # ------------------------------------------------------------------
    # Scalar parameter overrides (None = use historical calibrated value)
    # ------------------------------------------------------------------
    sigma_override: Optional[float] = None
    sigma_by_block_override: Optional[dict] = None      # {block_label: sigma}
    theta_override: Optional[float] = None
    theta_am_override: Optional[float] = None
    theta_pm_override: Optional[float] = None
    ou_max_stationary_std_override: Optional[float] = None
    persistence_filter_offset_override: Optional[float] = None
    drift_am_override: Optional[float] = None
    drift_pm_override: Optional[float] = None
    kalman_bias_override: Optional[float] = None
    daily_max_bias_override: Optional[float] = None     # None = use state.nwp_daily_max_bias; 0.0 = disable
    anchor_weight_multiplier: float = 1.0               # 0 = off, 1 = normal
    model_weights_override: Optional[dict] = None       # e.g. {"HRRR": 0.7, "GFS": 0.2, "ECMWF": 0.1}

    # ------------------------------------------------------------------
    # Cloud cover parameters (used when use_cloud_cover_adjustment=True)
    # ------------------------------------------------------------------
    cloud_cover_overcast_threshold: float = 80.0
    cloud_cover_clear_threshold: float = 20.0
    cloud_cover_overcast_sigma_factor: float = 0.8      # matches settings default
    cloud_cover_clear_sigma_factor: float = 1.1         # matches settings default

    # ------------------------------------------------------------------
    # Ensemble spread parameters (used when use_ensemble_spread_adjustment=True)
    # ------------------------------------------------------------------
    ensemble_spread_threshold: float = 3.0              # matches settings default
    ensemble_spread_sigma_factor: float = 1.3           # matches settings default

    # ------------------------------------------------------------------
    # MC run configuration
    # ------------------------------------------------------------------
    n_paths: int = 10_000
    random_seed: int = 42
    eval_hours: list = field(default_factory=lambda: [8, 10, 12, 14, 16])

    # ------------------------------------------------------------------
    # Kalman replay mode
    # ------------------------------------------------------------------
    # When True, the replay engine discards the stored kalman_bias_estimate
    # and re-runs the current filter logic (H=[[1,1]], bias decay, covariance
    # cap) over historical ASOS readings to produce a fresh bias at eval_hour.
    # This corrects for dates whose stored bias was written by an older filter
    # configuration (pre-Phase A drift removal, pre-Phase C decay).
    replay_kalman_bias: bool = False

    # ------------------------------------------------------------------
    # Hashing / equality
    # ------------------------------------------------------------------

    def __hash__(self) -> int:
        """Hash the scenario for use as a Streamlit cache key.

        Converts unhashable fields (dict, list) to sorted tuples so the
        overall hash is deterministic and consistent.
        """
        items = []
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, dict):
                v = tuple(sorted(v.items()))
            elif isinstance(v, list):
                v = tuple(v)
            items.append((f.name, v))
        return hash(tuple(items))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Scenario):
            return NotImplemented
        return all(getattr(self, f.name) == getattr(other, f.name) for f in fields(self))


# ---------------------------------------------------------------------------
# ReplayDataCache — bulk-load per-date data to avoid per-hour DB round-trips
# ---------------------------------------------------------------------------


@dataclass
class ReplayDataCache:
    """Pre-loaded ASOS, system_state, and market data for a set of dates.

    NWP forecasts are NOT pre-loaded because the replay engine needs them
    filtered by cutoff UTC (to prevent future leakage), which varies per
    eval_hour.  They are fetched individually by the replay engine via
    get_nwp_forecasts_before_utc().

    Args:
        _asos:    {date: list[ASOSReadingDocument]} — all readings for the date.
        _states:  {date: Optional[SystemStateDocument]}
        _markets: {date: Optional[MarketDocument]}
    """

    _asos: dict
    _states: dict
    _markets: dict

    @classmethod
    def load(cls, settled_dates: list[date]) -> "ReplayDataCache":
        """Bulk-load data for all settled_dates from the database.

        Args:
            settled_dates: Dates to pre-load (already filtered for settlement).

        Returns:
            ReplayDataCache with all data loaded.
        """
        asos_map: dict = {}
        state_map: dict = {}
        market_map: dict = {}

        for d in settled_dates:
            try:
                asos_map[d] = get_asos_readings_for_date(d)
            except Exception as exc:
                logger.warning("cache.asos_load_failed", date=str(d), error=str(exc))
                asos_map[d] = []

            try:
                state_map[d] = get_system_state(d)
            except Exception as exc:
                logger.warning("cache.state_load_failed", date=str(d), error=str(exc))
                state_map[d] = None

            try:
                market_map[d] = get_market(d)
            except Exception as exc:
                logger.warning("cache.market_load_failed", date=str(d), error=str(exc))
                market_map[d] = None

        logger.info(
            "replay_cache.loaded",
            n_dates=len(settled_dates),
            asos_loaded=sum(1 for v in asos_map.values() if v),
            states_loaded=sum(1 for v in state_map.values() if v is not None),
        )
        return cls(_asos=asos_map, _states=state_map, _markets=market_map)

    def get_asos_up_to(self, d: date, cutoff_utc: datetime) -> list[ASOSReadingDocument]:
        """Return ASOS readings for date d that were available at cutoff_utc.

        Args:
            d:          Trading date.
            cutoff_utc: UTC timestamp representing the eval_hour boundary.

        Returns:
            Readings with observation_time_utc <= cutoff_utc.
        """
        all_readings = self._asos.get(d, [])
        result = []
        for r in all_readings:
            obs = r.observation_time_utc
            if obs.tzinfo is None:
                obs = obs.replace(tzinfo=timezone.utc)
            if obs <= cutoff_utc:
                result.append(r)
        return result

    def get_state(self, d: date) -> Optional[SystemStateDocument]:
        """Return the SystemStateDocument for date d, or None."""
        return self._states.get(d)

    def get_market(self, d: date) -> Optional[MarketDocument]:
        """Return the MarketDocument for date d, or None."""
        return self._markets.get(d)


# ---------------------------------------------------------------------------
# Preset scenario functions
# ---------------------------------------------------------------------------


def preset_production() -> Scenario:
    """Current production configuration (post-Phase B tuning, 2026-05-02).

    All overrides are None — historical calibrated values are used for each
    date.  This is the ground-truth baseline for all comparisons.

    Phase A removed drift from attractor; Phase B re-enables it after backtest
    confirmed Brier improves from 0.1069 → 0.0872 (p=0.003) with drift ON.
    Kalman bias cap (±3.5°F) applied in build_mc_params_historical() via settings.
    """
    return Scenario(
        name="Production (Current)",
        use_drift_in_attractor=True,    # Phase B: drift re-enabled (validated by backtest)
        use_anchor_offset=True,
        use_time_varying_sigma=True,
        use_time_varying_theta=True,
        use_persistence_offset=True,
    )


def preset_pre_phase_a() -> Scenario:
    """Configuration BEFORE Phase A fixes.

    Enables drift in the attractor and uses the loose sigma cap and original
    persistence offset.  Used to verify that Phase A improved accuracy.
    """
    return Scenario(
        name="Pre-Phase A (Drift + Loose Cap)",
        use_drift_in_attractor=True,
        ou_max_stationary_std_override=2.0,
        persistence_filter_offset_override=0.30,
    )


def preset_no_drift() -> Scenario:
    """Remove drift from the attractor entirely (same as production since Phase A)."""
    return Scenario(
        name="No Drift",
        use_drift_in_attractor=False,
    )


def preset_no_anchor() -> Scenario:
    """Remove the NWP anchor offset.

    Tests whether the T0-to-NWP gap correction helps or hurts accuracy.
    Gemini Finding 3 suggested this might be over-correcting on some days.
    """
    return Scenario(
        name="No Anchor Offset",
        use_anchor_offset=False,
    )


def preset_half_anchor() -> Scenario:
    """Reduce anchor offset influence by 50%."""
    return Scenario(
        name="Half Anchor (50%)",
        anchor_weight_multiplier=0.5,
    )


def preset_no_corrections() -> Scenario:
    """Bare NWP + OU diffusion.  No Kalman bias, no drift, no anchor.

    The 'dumb baseline' — how well does raw NWP + diffusion do?
    """
    return Scenario(
        name="Raw NWP Only",
        use_drift_in_attractor=False,
        use_anchor_offset=False,
        kalman_bias_override=0.0,
    )


def preset_tight_sigma_cap() -> Scenario:
    """Tighter sigma cap: stationary std ≤ 1.5°F (down from 2.0°F default)."""
    return Scenario(
        name="Tight Sigma Cap (1.5°F)",
        ou_max_stationary_std_override=1.5,
    )


def preset_flat_sigma() -> Scenario:
    """Use a single pooled sigma instead of time-varying per-block sigmas."""
    return Scenario(
        name="Flat Sigma (No Blocks)",
        use_time_varying_sigma=False,
    )


def preset_flat_theta() -> Scenario:
    """Use a single scalar theta instead of the AM/PM regime split."""
    return Scenario(
        name="Flat Theta (No AM/PM)",
        use_time_varying_theta=False,
    )


def preset_cloud_cover() -> Scenario:
    """Enable cloud cover sigma adjustment.

    Overcast (>80%): σ × 0.8 — NWP more accurate when sky is stable.
    Clear (<20%): σ × 1.1 — more convective variability on sunny days.
    """
    return Scenario(
        name="Cloud Cover Active",
        use_cloud_cover_adjustment=True,
        cloud_cover_overcast_sigma_factor=0.8,
        cloud_cover_clear_sigma_factor=1.1,
    )


def preset_ensemble_spread() -> Scenario:
    """Enable ensemble spread sigma inflation.

    When ensemble member std > 3°F, inflate σ by 1.3× to reflect model
    disagreement about the day's temperature trajectory.
    """
    return Scenario(
        name="Ensemble Spread Active",
        use_ensemble_spread_adjustment=True,
        ensemble_spread_threshold=3.0,
        ensemble_spread_sigma_factor=1.3,
    )


def preset_aggressive_corrections() -> Scenario:
    """Everything on: drift + anchor + cloud + ensemble + tight cap.

    Useful as an upper bound on correction complexity.
    """
    return Scenario(
        name="All Corrections (Aggressive)",
        use_drift_in_attractor=True,
        use_anchor_offset=True,
        use_cloud_cover_adjustment=True,
        use_ensemble_spread_adjustment=True,
        ou_max_stationary_std_override=1.5,
    )


def preset_conservative() -> Scenario:
    """Minimal corrections: NWP + Kalman bias only, no drift, no anchor.

    Tight sigma cap bets on NWP accuracy over diffusion spread.
    """
    return Scenario(
        name="Conservative (Minimal Corrections)",
        use_drift_in_attractor=False,
        use_anchor_offset=False,
        use_cloud_cover_adjustment=False,
        use_ensemble_spread_adjustment=False,
        ou_max_stationary_std_override=1.2,
    )


def preset_no_daily_max_bias() -> Scenario:
    """Production config with daily-max bias correction forced to zero.

    Use in Compare mode against Production to isolate the contribution of
    nwp_daily_max_bias (D3 fix). If Production wins at 10AM, the EMA is adding
    real signal; if not, the bias may be overfit or the EMA not yet converged.
    """
    return Scenario(
        name="No Daily-Max Bias",
        daily_max_bias_override=0.0,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_PRESETS: list[Scenario] = [
    preset_production(),
    preset_pre_phase_a(),
    preset_no_drift(),
    preset_no_anchor(),
    preset_half_anchor(),
    preset_no_corrections(),
    preset_tight_sigma_cap(),
    preset_flat_sigma(),
    preset_flat_theta(),
    preset_cloud_cover(),
    preset_ensemble_spread(),
    preset_aggressive_corrections(),
    preset_conservative(),
    preset_no_daily_max_bias(),
]

# Ordered dict for Streamlit selectbox — preserves insertion order (Python 3.7+)
PRESET_MAP: dict[str, Scenario] = {s.name: s for s in ALL_PRESETS}
