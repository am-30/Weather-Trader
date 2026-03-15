"""
Application configuration loaded from environment variables via pydantic-settings.

All secrets must be defined in a ``.env`` file or in the process environment.
Never hardcode credentials. Use ``settings`` as a singleton after import:

    from src.config import settings
"""

from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated, typed application settings sourced from environment variables.

    All fields are required unless a default is provided. Fields with no
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
    # Kalshi
    # ------------------------------------------------------------------
    kalshi_api_key: str = Field(..., description="Kalshi REST API key")
    kalshi_api_base_url: str = Field(
        default="https://trading-api.kalshi.com/trade-api/v2",
        description="Base URL for the Kalshi trading API",
    )
    kalshi_env: str = Field(
        default="demo",
        description="'demo' or 'prod' — which Kalshi environment to target",
    )

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------
    database_url: str = Field(..., description="PostgreSQL connection string")

    # ------------------------------------------------------------------
    # NWS weather
    # ------------------------------------------------------------------
    nws_station: str = Field(
        default="KBOS",
        description="ICAO station code for weather observations",
    )
    nws_api_base_url: str = Field(
        default="https://api.weather.gov",
        description="NWS API base URL",
    )

    # ------------------------------------------------------------------
    # Trading parameters
    # ------------------------------------------------------------------
    max_trade_size_usd: float = Field(
        default=50.0,
        ge=1.0,
        description="Maximum USD amount to risk on a single trade",
    )
    min_edge_cents: float = Field(
        default=3.0,
        ge=0.0,
        description="Minimum expected-value edge (in cents) required to trade",
    )
    max_contracts_per_market: int = Field(
        default=10,
        ge=1,
        description="Maximum open contracts allowed per market",
    )

    # ------------------------------------------------------------------
    # Scheduler intervals
    # ------------------------------------------------------------------
    weather_fetch_interval_minutes: int = Field(default=15, ge=1)
    market_fetch_interval_minutes: int = Field(default=5, ge=1)
    forecast_interval_minutes: int = Field(default=30, ge=1)
    trade_eval_interval_minutes: int = Field(default=10, ge=1)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_level: str = Field(default="INFO")

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
