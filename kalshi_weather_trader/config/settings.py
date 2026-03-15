"""
Application configuration loaded from environment variables via pydantic-settings.

All secrets must be defined in a ``.env`` file or in the process environment
(Replit Secrets are automatically injected as environment variables).
Never hardcode credentials.

Module-level helpers ``get_target_date()``, ``get_trading_day_bounds()``, and
``get_remaining_day_fraction()`` are importable without instantiating Settings.
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


def get_trading_day_bounds() -> tuple[datetime, datetime]:
    """Return UTC start and end datetimes for the current trading target date.

    The "trading day" runs from midnight to midnight Eastern, expressed as UTC.

    Args:
        None

    Returns:
        Tuple of (day_start_utc, day_end_utc) as timezone-aware UTC datetimes.

    Raises:
        Nothing.
    """
    target = get_target_date()
    et_midnight = _EASTERN.localize(
        datetime(target.year, target.month, target.day, 0, 0, 0)
    )
    utc_start = et_midnight.astimezone(pytz.utc)
    utc_end = utc_start + timedelta(days=1)
    return utc_start, utc_end


def get_remaining_day_fraction() -> float:
    """Return the fraction of the trading day that remains as a float in [0, 1].

    Used by Monte Carlo to scale the number of simulation steps.

    Args:
        None

    Returns:
        Float in ``[0.0, 1.0]`` representing remaining fraction of the day.

    Raises:
        Nothing.
    """
    now_et = datetime.now(_EASTERN)
    target = get_target_date()
    # Day runs from midnight to midnight Eastern
    day_end_et = _EASTERN.localize(
        datetime(target.year, target.month, target.day, 23, 59, 59)
    )
    day_start_et = _EASTERN.localize(
        datetime(target.year, target.month, target.day, 0, 0, 0)
    )
    total_seconds = (day_end_et - day_start_et).total_seconds()
    elapsed_seconds = max(0.0, (now_et - day_start_et).total_seconds())
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
        default="https://trading-api.kalshi.com/trade-api/v2",
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
        default=0.3,
        gt=0.0,
        description="Observation noise variance for ASOS readings",
    )

    # ------------------------------------------------------------------
    # Ornstein-Uhlenbeck process parameters
    # ------------------------------------------------------------------
    ou_theta: float = Field(
        default=0.1,
        gt=0.0,
        description="Mean-reversion speed (per hour)",
    )
    ou_sigma: float = Field(
        default=2.0,
        gt=0.0,
        description="Volatility (degrees F per sqrt-hour)",
    )
    mc_n_paths: int = Field(
        default=10_000,
        ge=1_000,
        description="Number of Monte Carlo simulation paths",
    )

    # ------------------------------------------------------------------
    # Scheduler intervals
    # ------------------------------------------------------------------
    asos_fetch_interval_minutes: int = Field(default=5, ge=1)
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
        description="IEM Mesonet API base URL (ASOS fallback)",
    )
    asos_staleness_minutes: int = Field(
        default=30,
        ge=1,
        description="Max age of NWS observation before falling back to IEM",
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
