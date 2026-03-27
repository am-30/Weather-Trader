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
    # Kalman filter noise parameters
    # ------------------------------------------------------------------
    kalman_q_temp: float = Field(
        default=0.1,
        gt=0.0,
        description="Process noise variance for temperature state",
    )
    kalman_q_bias: float = Field(
        default=0.05,
        gt=0.0,
        description="Process noise variance for model-bias state",
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

    # ------------------------------------------------------------------
    # Ornstein-Uhlenbeck process parameters
    # ------------------------------------------------------------------
    ou_theta: float = Field(
        default=0.3,   # was 0.1; 0.1 → half-life ≈7h (too slow for Boston);
                       # 0.3 → half-life ≈2.3h (matches observed anomaly decay).
                       # Before calibration accumulates live data (first trading day),
                       # this default governs OU path width. 0.1 produced near-
                       # random-walk paths that priced edges too wide.
        gt=0.0,
        description="Mean-reversion speed (per hour)",
    )
    ou_sigma: float = Field(
        default=0.6,
        gt=0.0,
        description="Volatility (degrees F per sqrt-hour)",
    )
    ou_max_stationary_std: float = Field(
        default=1.0,
        gt=0.0,
        description=(
            "Hard cap on the OU process stationary standard deviation (°F). "
            "Enforced in run_simulation() by capping sigma to "
            "max_stationary_std * sqrt(2 * theta) before the simulation loop. "
            "Physically: the equilibrium deviation of temperature from the NWP "
            "attractor should approximate the NWP intraday RMSE for KBOS "
            "(~1–1.5°F for same-day hourly forecasts). Without this cap, a "
            "calibrated sigma >> max_stationary_std * sqrt(2*theta) produces "
            "near-random-walk paths where per-step noise is 31× the restoring "
            "force, causing paths to spike far above the declining NWP attractor "
            "and grossly inflate P(daily_max). Overridable via env var "
            "OU_MAX_STATIONARY_STD."
        ),
    )
    persistence_filter_offset: float = Field(
        default=0.3,
        ge=0.0,
        le=0.5,
        description=(
            "Expected gap between the ASOS tabular maximum temperature and the true "
            "NWS daily maximum (°F). Caused by the 0.5°C ASOS persistence filter: the "
            "sensor only updates when temperature changes by ≥0.5°C, so the true intraday "
            "peak often falls between threshold crossings and is not reflected in tabular "
            "readings. Applied as an offset to hard_floor when initialising paths_max in "
            "run_simulation(); the hard_floor stored in the DB is never modified. "
            "Calibrated from historical data (calibrate_persistence_offset()); default 0.3°F "
            "is conservative for KBOS. Overridable via env var PERSISTENCE_FILTER_OFFSET."
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
