"""
Application configuration loaded from environment variables via pydantic-settings.

All secrets must be defined in a ``.env`` file or in the process environment
(Replit Secrets are automatically injected as environment variables).
Never hardcode credentials.

Module-level helpers ``get_target_date()``, ``get_trading_day_bounds()``,
``get_nws_day_bounds()``, and ``get_remaining_day_fraction()`` are importable
without instantiating Settings.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from typing import Optional

import pytz
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_EASTERN = pytz.timezone("America/New_York")
_ROLLOVER_HOUR_EASTERN = 18  # 6 PM Eastern → shift target to tomorrow

# NWS uses Local Standard Time (EST = UTC-5 fixed) year-round for climate records.
# The daily observation window is midnight-to-midnight EST regardless of DST.
_EST_FIXED = pytz.FixedOffset(-300)  # UTC-5, no DST adjustment


# ---------------------------------------------------------------------------
# Module-level helpers (no Settings instantiation required)
# ---------------------------------------------------------------------------


def get_target_date() -> date:
    """Return the trading target date in Eastern time.

    After 6 PM Eastern the target shifts to tomorrow (pre-market positioning
    for the following day's maximum temperature).

    Args:
        None

    Returns:
        ``datetime.date`` representing the active trading target date.

    Raises:
        Nothing.
    """
    now_et = datetime.now(_EASTERN)
    if now_et.hour >= _ROLLOVER_HOUR_EASTERN:
        return (now_et + timedelta(days=1)).date()
    return now_et.date()


def get_nws_day_bounds(target_date: date) -> tuple[datetime, datetime]:
    """Return UTC start and end datetimes for the NWS observation day.

    NWS uses Local Standard Time (EST = UTC-5 fixed) year-round for climate
    records.  The daily maximum temperature observation window is
    midnight-to-midnight EST, which corresponds to 01:00–00:59 EDT during
    daylight saving time.  This is the window that Kalshi uses to settle the
    daily maximum temperature market.

    Args:
        target_date: The calendar date whose NWS observation window is needed.

    Returns:
        Tuple of (day_start_utc, day_end_utc) as timezone-aware UTC datetimes.

    Raises:
        Nothing.
    """
    est_midnight = _EST_FIXED.localize(
        datetime(target_date.year, target_date.month, target_date.day, 0, 0, 0)
    )
    utc_start = est_midnight.astimezone(pytz.utc)
    utc_end = utc_start + timedelta(days=1)
    return utc_start, utc_end


def get_trading_day_bounds() -> tuple[datetime, datetime]:
    """Return UTC start and end datetimes for the current trading target date.

    Aligned to the NWS observation window: midnight-to-midnight EST (UTC-5
    fixed), which is 01:00–00:59 EDT during daylight saving time.

    Args:
        None

    Returns:
        Tuple of (day_start_utc, day_end_utc) as timezone-aware UTC datetimes.

    Raises:
        Nothing.
    """
    return get_nws_day_bounds(get_target_date())


def get_remaining_day_fraction() -> float:
    """Return the fraction of the NWS observation day that remains as a float in [0, 1].

    The observation window runs midnight-to-midnight EST (UTC-5 fixed).  During
    the gap between midnight EDT and midnight EST (01:00 EDT, i.e. before the
    NWS window opens), this returns 1.0 — the full day is still ahead.

    Used by Monte Carlo to scale the number of simulation steps.

    Args:
        None

    Returns:
        Float in ``[0.0, 1.0]`` representing remaining fraction of the day.

    Raises:
        Nothing.
    """
    now_utc = datetime.now(pytz.utc)
    target = get_target_date()
    day_start_utc, day_end_utc = get_nws_day_bounds(target)
    total_seconds = (day_end_utc - day_start_utc).total_seconds()
    elapsed_seconds = max(0.0, (now_utc - day_start_utc).total_seconds())
    return max(0.0, min(1.0, 1.0 - elapsed_seconds / total_seconds))


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """Validated, typed application settings sourced from environment variables.

    All fields are required unless a default is provided.  Fields with no
    default and no environment variable will raise a ``ValidationError`` at
    startup, making mis-configuration fail fast.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------
    database_url: str = Field(..., description="PostgreSQL connection string (DATABASE_URL)")

    # ------------------------------------------------------------------
    # Kalshi RSA authentication
    # ------------------------------------------------------------------
    kalshi_access_key: str = Field(
        default="",
        description="Kalshi API key ID (short string from API settings page)",
    )
    kalshi_private_key: str = Field(
        default="",
        description="Full PEM private key content including headers",
    )
    kalshi_env: str = Field(
        default="demo",
        description="'demo' or 'prod' — which Kalshi environment to target",
    )
    kalshi_api_base_url: str = Field(
        default="https://api.elections.kalshi.com/trade-api/v2",
        description="Base URL for the Kalshi trading API",
    )

    # ------------------------------------------------------------------
    # Trading parameters
    # ------------------------------------------------------------------
    dry_run: bool = Field(
        default=True,
        description="If True, log trades but never submit real orders",
    )
    edge_threshold: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Minimum probability edge required to trade",
    )
    max_trade_size_usd: float = Field(
        default=50.0,
        ge=1.0,
        description="Maximum USD to risk on a single trade (Kelly denominator)",
    )
    max_contracts_per_market: int = Field(
        default=10,
        ge=1,
        description="Hard cap on contracts per market per trade signal",
    )
    kelly_fraction: float = Field(
        default=0.25,
        gt=0.0,
        le=1.0,
        description="Fractional Kelly multiplier (0.25 = quarter Kelly)",
    )

    # ------------------------------------------------------------------
    # Paper trading parameters
    # ------------------------------------------------------------------
    paper_budget_mode: str = Field(
        default="flat",
        description="'flat' = $10/day split among signals; 'kelly' = fixed bankroll with Kelly sizing",
    )
    paper_daily_budget: float = Field(
        default=10.0,
        gt=0.0,
        description="Per-day budget in USD for flat paper trading mode",
    )
    paper_initial_bankroll: float = Field(
        default=100.0,
        gt=0.0,
        description="Starting bankroll in USD for Kelly paper trading mode",
    )
    paper_entry_max_ask_cents: int = Field(
        default=50,
        ge=1,
        le=99,
        description="Paper trading will not enter if ask >= this threshold (cents)",
    )
    paper_limit_sell_cents: int = Field(
        default=75,
        ge=1,
        le=99,
        description="Paper trading closes position when market bid reaches this price (cents)",
    )

    # ------------------------------------------------------------------
    # Kalman filter noise parameters
    # ------------------------------------------------------------------
    kalman_q_temp: float = Field(
        default=0.1,
        gt=0.0,
        description="Process noise variance for temperature state",
    )
    kalman_q_bias: float = Field(
        default=0.01,  # was 0.05; reduced to slow bias-state response and prevent
                       # day-to-day sign-flipping driven by ASOS 1°C quantization noise.
                       # Backtest Section 2: Pearson corr(Kalman_B, NWP_err)=0.005 —
                       # the filter was tracking noise, not real NWP error.
        gt=0.0,
        description="Process noise variance for model-bias state",
    )
    kalman_bias_mc_cap: float = Field(
        default=3.5,
        gt=0.0,
        description=(
            "Maximum |Kalman bias| (°F) applied to the MC attractor. "
            "Extreme bias spikes (+4.62 / -3.99°F observed) are truncated before "
            "being added to mu_t. Sweep A (2026-05-02): cap at ±3.5°F improves "
            "Brier by ~0.02 at p<0.001 vs uncapped. Filter state is unchanged — "
            "only what enters the MC is clamped. Overridable via KALMAN_BIAS_MC_CAP."
        ),
    )
    kalman_r_obs: float = Field(
        default=0.4,   # was 0.6 (increased in f661ca9 to dampen noise).
                       # Reduced to 0.4 now that warm-start is in place: bias no
                       # longer cold-starts at 0.0 each day, so the filter no longer
                       # needs to over-trust its prior during morning reconvergence.
                       # Physical ASOS accuracy is ±0.5°F → R ≈ 0.25; 0.4 is a
                       # conservative halfway point. Monitor for oscillation on live
                       # days; revert via env var KALMAN_R_OBS=0.6 if needed.
        gt=0.0,
        description="Observation noise variance for ASOS readings",
    )
    kalman_max_nwp_delta: float = Field(
        default=5.0,
        gt=0.0,
        description=(
            "Maximum absolute NWP hourly delta accepted by Kalman predict step (°F/hr). "
            "Clamps corrupt or physically implausible model spikes before they shift "
            "the temperature estimate. Typical Boston diurnal ramp is 1–3°F/hr."
        ),
    )
    kalman_p_max_diagonal: float = Field(
        default=2.0,
        gt=0.0,
        description=(
            "Hard cap on each diagonal element of the Kalman covariance matrix P (°F²). "
            "Applied after every update() call and after warm-start construction in "
            "load_or_initialize_filter(). Prevents covariance explosion from compounding "
            "warm-start inflation and gap inflation (e.g. P=[[40,-40],[-40,40]] observed "
            "March 29 after warm-start × 1.2 stacked on top of prior pathological state). "
            "A converged 2D Kalman with Q_temp=0.1, R=0.4 settles to P[0,0]≈0.1–0.5; "
            "cap of 2.0 allows sqrt(2)≈1.4°F uncertainty — well above equilibrium. "
            "Overridable via env var KALMAN_P_MAX_DIAGONAL."
        ),
    )
    kalman_innovation_gate_sigma: float = Field(
        default=4.0,
        gt=0.0,
        description=(
            "Mahalanobis distance threshold (σ units) for the Kalman innovation gate. "
            "If |innovation| / sqrt(S[0,0]) exceeds this value, the update() step is "
            "skipped entirely and a warning is logged. Rejects corrupt ASOS readings "
            "without contaminating the filter state. NOTE: effective °F ceiling depends "
            "on S = H@P@H.T + R; with a converged, anti-correlated P the ceiling can be "
            "as low as 3-5°F. A stuck-detection counter in update() reinitialises the "
            "filter state after 10 consecutive rejections so the filter cannot freeze "
            "permanently on real weather shifts. "
            "Overridable via env var KALMAN_INNOVATION_GATE_SIGMA."
        ),
    )
    kalman_bias_decay: float = Field(
        default=0.95,
        gt=0.0,
        le=1.0,
        description=(
            "Per-hour decay factor for the bias state in the Kalman state transition. "
            "Applied as F[1,1] = kalman_bias_decay ** dt_hours in each predict() call, "
            "so the decay is correctly scaled regardless of the time step used. "
            "0.95/hour → half-life ≈ 13.5 hours: long enough to track genuine all-day "
            "NWP cold/warm bias; short enough to prevent transient intraday warming from "
            "accumulating in B. A genuine persistent NWP error (same sign all day) keeps "
            "B nonzero via repeated innovations; transient temperature dynamics decay away "
            "once innovations stop. Set closer to 1.0 for slower-moving bias tracking; "
            "closer to 0.9 for faster decay. Overridable via env var KALMAN_BIAS_DECAY."
        ),
    )

    # ------------------------------------------------------------------
    # Ornstein-Uhlenbeck process parameters
    # ------------------------------------------------------------------
    ou_theta: float = Field(
        default=0.7,   # was 0.3; backtest Sweep D (2026-05-02): theta=0.7 is
                       # optimal (Brier 0.0898, p=0.021 vs production 0.1069).
                       # 0.3 is uncalibrated fallback — calibrate_theta() requires
                       # ≥12 AR(1) pairs and silently falls back to default when
                       # insufficient historical data. theta=0.7 → half-life ≈1.0h
                       # (stronger mean-reversion keeps OU paths near NWP attractor).
        gt=0.0,
        description="Mean-reversion speed (per hour)",
    )
    ou_sigma: float = Field(
        default=0.6,
        gt=0.0,
        description="Volatility (degrees F per sqrt-hour)",
    )
    ou_max_stationary_std: float = Field(
        default=1.5,
        gt=0.0,
        description=(
            "Hard cap on the OU process stationary standard deviation (°F). "
            "Enforced in run_simulation() by capping sigma to "
            "max_stationary_std * sqrt(2 * theta) before the simulation loop. "
            "With theta_am≈0.29, sigma_max = 1.5 * sqrt(0.58) ≈ 1.14°F — caps the "
            "morning block sigma of 1.41°F to 1.14°F (Phase A reduction from 2.0). "
            "The 2.0 default was raised March 29 from 1.0 but proved too loose: with "
            "drift also in the attractor, paths could reach 9°F above raw NWP peak. "
            "Phase 3 calibration will replace this with hourly NWP RMSE × 1.0 once "
            "≥10 CLI-confirmed dates are available. Overridable via env var "
            "OU_MAX_STATIONARY_STD."
        ),
    )
    persistence_filter_offset: float = Field(
        default=0.75,
        ge=0.0,
        le=1.5,
        description=(
            "Expected gap between the ASOS tabular maximum temperature and the true "
            "NWS daily maximum (°F). Caused by the 0.5°C ASOS persistence filter: the "
            "sensor only updates when temperature changes by ≥0.5°C, so the true intraday "
            "peak often falls between threshold crossings and is not reflected in tabular "
            "readings. Applied as an offset to hard_floor when initialising paths_max in "
            "run_simulation(); the hard_floor stored in the DB is never modified. "
            "Calibrated from historical data (calibrate_persistence_offset()); empirical "
            "data (8 settled dates, mean gap 0.75°F) supports values near 0.75–1.0°F. "
            "Clamp raised from [0.0, 0.5] to [0.0, 1.5] in Phase A. "
            "Overridable via env var PERSISTENCE_FILTER_OFFSET."
        ),
    )
    nwp_daily_max_bias_alpha: float = Field(
        default=0.1,
        gt=0.0,
        le=1.0,
        description=(
            "EMA smoothing factor for nwp_daily_max_bias calibration. "
            "alpha=0.1 gives an effective window of ~9 days. "
            "Higher = faster adaptation to recent NWP error patterns."
        ),
    )
    calibration_lookback_days: int = Field(
        default=30,
        ge=7,
        le=90,
        description=(
            "Default rolling window (days) for sigma and theta calibration. "
            "30 days provides ~120 samples per 5-hour ET block for per-block "
            "sigma estimation while avoiding excessive seasonal drift. "
            "Overridable via env var CALIBRATION_LOOKBACK_DAYS."
        ),
    )
    calibration_decay_tau_days: int = Field(
        default=10,
        ge=1,
        le=60,
        description=(
            "Exponential decay time constant (days) for calibration weighting. "
            "A reading from tau days ago has weight exp(-1) ≈ 0.37 relative to "
            "today. tau=10 means last week still matters (~0.5 weight) but a "
            "reading from 30 days ago contributes little (~0.05 weight). "
            "Overridable via env var CALIBRATION_DECAY_TAU_DAYS."
        ),
    )
    mc_n_paths: int = Field(
        default=10_000,
        ge=1_000,
        description="Number of Monte Carlo simulation paths",
    )

    # ------------------------------------------------------------------
    # Scheduler intervals
    # ------------------------------------------------------------------
    asos_fetch_interval_minutes: int = Field(
        default=2,
        ge=1,
        description=(
            "How often the scheduler fires the ASOS fetch job (minutes). "
            "Kept low (2 min) so new METARs are captured quickly, but the "
            "actual API call rate is governed by asos_min_fetch_interval_minutes."
        ),
    )
    asos_min_fetch_interval_minutes: int = Field(
        default=4,
        ge=1,
        description=(
            "Minimum time between real API calls in fetch_current_observation() "
            "(minutes). If the last call was more recent than this, the job "
            "returns the cached DB reading without contacting any external API. "
            "IEM/METAR data refreshes at METAR frequency (~20-60 min for routine "
            "reports, faster for special reports), so 4 min is a safe floor that "
            "prevents hammering servers while still catching most new readings "
            "within one scheduler tick of their arrival."
        ),
    )
    nwp_fetch_interval_minutes: int = Field(default=60, ge=1)
    trade_eval_interval_minutes: int = Field(default=5, ge=1)
    snapshot_interval_hours: int = Field(default=2, ge=1)
    rollover_check_interval_minutes: int = Field(default=30, ge=1)
    ensemble_fetch_interval_minutes: int = Field(default=60, ge=1, description="How often to fetch NWP ensemble data (minutes). Matches NWP fetch interval.")
    ensemble_spread_threshold: float = Field(
        default=3.0,
        gt=0.0,
        description=(
            "Ensemble spread threshold (°F) above which sigma is inflated. "
            "When std(ensemble_member_daily_highs) > this value, the atmosphere is "
            "genuinely uncertain (likely a frontal day). "
            "Overridable via env var ENSEMBLE_SPREAD_THRESHOLD."
        ),
    )
    ensemble_spread_inflation: float = Field(
        default=1.3,
        gt=0.0,
        description=(
            "Multiplicative sigma inflation factor when ensemble spread exceeds threshold. "
            "Applied to raw block sigma BEFORE the OU sigma cap so the cap remains a hard ceiling. "
            "Overridable via env var ENSEMBLE_SPREAD_INFLATION."
        ),
    )
    cloudcover_overcast_threshold: float = Field(
        default=80.0,
        ge=0.0,
        le=100.0,
        description=(
            "Mean cloud cover (%) for hours 10-16 ET above which the day is classified as overcast. "
            "On heavily overcast days NWP is more accurate (less convective variability). "
            "Overridable via env var CLOUDCOVER_OVERCAST_THRESHOLD."
        ),
    )
    cloudcover_clear_threshold: float = Field(
        default=20.0,
        ge=0.0,
        le=100.0,
        description=(
            "Mean cloud cover (%) for hours 10-16 ET below which the day is classified as clear. "
            "On clear days convective variability is higher (solar insolation uncertainty). "
            "Overridable via env var CLOUDCOVER_CLEAR_THRESHOLD."
        ),
    )
    cloudcover_overcast_factor: float = Field(
        default=0.8,
        gt=0.0,
        description=(
            "Multiplicative sigma reduction on overcast days (cloudcover > overcast_threshold). "
            "Overridable via env var CLOUDCOVER_OVERCAST_FACTOR."
        ),
    )
    cloudcover_clear_factor: float = Field(
        default=1.1,
        gt=0.0,
        description=(
            "Multiplicative sigma inflation on clear days (cloudcover < clear_threshold). "
            "Overridable via env var CLOUDCOVER_CLEAR_FACTOR."
        ),
    )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_level: str = Field(default="INFO")

    # ------------------------------------------------------------------
    # NWS / weather station
    # ------------------------------------------------------------------
    nws_station: str = Field(
        default="KBOS",
        description="ICAO station code — hardcoded to Boston Logan",
    )
    nws_api_base_url: str = Field(
        default="https://api.weather.gov",
        description="NWS API base URL",
    )
    iem_api_base_url: str = Field(
        default="https://mesonet.agron.iastate.edu",
        description="IEM Mesonet API base URL (primary ASOS source)",
    )
    aviationweather_api_base_url: str = Field(
        default="https://aviationweather.gov",
        description="Aviation Weather Center API base URL (secondary ASOS source)",
    )
    asos_staleness_minutes: int = Field(
        default=30,
        ge=1,
        description="Max age of any observation before it is considered stale",
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("kalshi_env")
    @classmethod
    def validate_kalshi_env(cls, v: str) -> str:
        """Ensure kalshi_env is one of the accepted values.

        Args:
            v: Raw value from environment.

        Returns:
            Lowercased, validated value.

        Raises:
            ValueError: If the value is not 'demo' or 'prod'.
        """
        lowered = v.lower()
        if lowered not in {"demo", "prod"}:
            raise ValueError(f"kalshi_env must be 'demo' or 'prod', got: {v!r}")
        return lowered

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure log_level is a valid Python logging level.

        Args:
            v: Raw value from environment.

        Returns:
            Uppercased, validated log level string.

        Raises:
            ValueError: If the value is not a recognised level.
        """
        upper = v.upper()
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if upper not in valid:
            raise ValueError(f"log_level must be one of {valid}, got: {v!r}")
        return upper



settings = Settings()
