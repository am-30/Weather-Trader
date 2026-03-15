"""
Pydantic v2 models for NWS weather observations and gridpoint forecasts.

All timestamps are stored as UTC-aware datetimes. Temperatures are always
Fahrenheit floats rounded to one decimal place.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class WeatherObservation(BaseModel):
    """A single hourly observation record from a NWS station.

    Attributes:
        station_id:    ICAO station identifier (e.g. ``"KBOS"``).
        observed_at:   UTC timestamp of the observation.
        temp_f:        Air temperature in Fahrenheit (1 decimal place).
        dew_point_f:   Dew-point temperature in Fahrenheit, if available.
        wind_speed_mph: Wind speed in miles per hour, if available.
        wind_dir_deg:  Wind direction in degrees (0–360), if available.
        precip_in:     Precipitation in inches over the observation period.
        sky_cover:     Sky coverage percentage (0–100), if available.
        raw_text:      Raw METAR text from the API response.
    """

    station_id: str = Field(..., min_length=3, max_length=5)
    observed_at: datetime = Field(..., description="UTC observation timestamp")
    temp_f: float = Field(..., description="Temperature in Fahrenheit")
    dew_point_f: Optional[float] = Field(None)
    wind_speed_mph: Optional[float] = Field(None, ge=0.0)
    wind_dir_deg: Optional[float] = Field(None, ge=0.0, le=360.0)
    precip_in: float = Field(default=0.0, ge=0.0)
    sky_cover: Optional[float] = Field(None, ge=0.0, le=100.0)
    raw_text: Optional[str] = Field(None)

    @field_validator("temp_f", "dew_point_f", mode="before")
    @classmethod
    def round_temperature(cls, v: Optional[float]) -> Optional[float]:
        """Round a temperature value to one decimal place.

        Args:
            v: Raw temperature float or None.

        Returns:
            Temperature rounded to 1 d.p., or None.

        Raises:
            Nothing — None values are passed through unchanged.
        """
        if v is None:
            return None
        return round(float(v), 1)

    @field_validator("observed_at", mode="before")
    @classmethod
    def ensure_utc(cls, v: datetime | str) -> datetime:
        """Ensure the observation timestamp is timezone-aware UTC.

        Args:
            v: Datetime or ISO-8601 string.

        Returns:
            UTC-aware datetime.

        Raises:
            ValueError: If the value cannot be parsed or is not UTC.
        """
        from datetime import timezone

        if isinstance(v, str):
            v = datetime.fromisoformat(v)
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        return v


class DailyMaxObservation(BaseModel):
    """The daily maximum temperature derived from hourly observations.

    Attributes:
        station_id:  ICAO station code.
        date_utc:    The UTC calendar date for which the max was computed.
        max_temp_f:  Highest observed temperature for the day in Fahrenheit.
        observation_count: Number of hourly readings used for the calculation.
        computed_at: UTC timestamp of when this record was computed.
    """

    station_id: str
    date_utc: datetime = Field(..., description="UTC calendar date (time is midnight)")
    max_temp_f: float
    observation_count: int = Field(..., ge=1)
    computed_at: datetime

    @field_validator("max_temp_f", mode="before")
    @classmethod
    def round_temperature(cls, v: float) -> float:
        """Round max temperature to one decimal place.

        Args:
            v: Raw float value.

        Returns:
            Float rounded to 1 d.p.

        Raises:
            TypeError: If ``v`` is not numeric.
        """
        return round(float(v), 1)


class NWSGridForecastPeriod(BaseModel):
    """One period in an NWS gridpoint hourly forecast.

    Attributes:
        start_time: UTC start of the forecast period.
        end_time:   UTC end of the forecast period.
        temp_f:     Forecast temperature in Fahrenheit.
        is_daytime: Whether this is a daytime period.
        short_forecast: Short text description (e.g. ``"Partly Sunny"``).
    """

    start_time: datetime
    end_time: datetime
    temp_f: float
    is_daytime: bool
    short_forecast: str = ""

    @field_validator("temp_f", mode="before")
    @classmethod
    def round_temperature(cls, v: float) -> float:
        """Round forecast temperature to one decimal place.

        Args:
            v: Raw float value.

        Returns:
            Float rounded to 1 d.p.

        Raises:
            TypeError: If ``v`` is not numeric.
        """
        return round(float(v), 1)
