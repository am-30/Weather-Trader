"""
Pydantic v2 models for internal probabilistic temperature forecasts.

A ``TemperatureForecast`` is the output of the forecasting module and is
consumed by the trading strategy. All temperatures in Fahrenheit.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator, computed_field
import math


class ForecastDistribution(BaseModel):
    """A Gaussian probability distribution over daily maximum temperature.

    Attributes:
        mean_f:  Point estimate of the daily maximum temperature (Fahrenheit).
        std_f:   Standard deviation of the estimate in degrees Fahrenheit.
    """

    mean_f: float = Field(..., description="Mean forecast temperature (°F)")
    std_f: float = Field(..., gt=0.0, description="Std deviation of forecast (°F)")

    @field_validator("mean_f", "std_f", mode="before")
    @classmethod
    def round_temperature(cls, v: float) -> float:
        """Round to one decimal place.

        Args:
            v: Raw float.

        Returns:
            Float rounded to 1 d.p.
        """
        return round(float(v), 1)

    def prob_above(self, threshold_f: float) -> float:
        """Probability that the actual max temperature exceeds a threshold.

        Uses the complementary error function (erfc) of the normal CDF.

        Args:
            threshold_f: Temperature threshold in Fahrenheit.

        Returns:
            Float in [0, 1] representing P(max > threshold).
        """
        z = (threshold_f - self.mean_f) / (self.std_f * math.sqrt(2))
        return 0.5 * math.erfc(z)

    def prob_below(self, threshold_f: float) -> float:
        """Probability that the actual max temperature stays below a threshold.

        Args:
            threshold_f: Temperature threshold in Fahrenheit.

        Returns:
            Float in [0, 1] representing P(max < threshold).
        """
        return 1.0 - self.prob_above(threshold_f)

    def prob_in_range(self, low_f: float, high_f: float) -> float:
        """Probability that the actual max falls in the half-open range [low, high).

        Args:
            low_f:  Lower bound (inclusive) in Fahrenheit.
            high_f: Upper bound (exclusive) in Fahrenheit.

        Returns:
            Float in [0, 1].
        """
        return self.prob_above(low_f) - self.prob_above(high_f)


class TemperatureForecast(BaseModel):
    """Complete forecast for a single target date.

    Attributes:
        station_id:      ICAO station code (e.g. ``"KBOS"``).
        target_date_utc: UTC calendar date being forecast.
        distribution:    Gaussian distribution over the daily max.
        nws_point_forecast_f: NWS gridpoint forecast value (reference anchor).
        historical_bias_f:    Systematic bias correction applied (°F).
        model_version:   Identifier for the forecasting model version.
        generated_at:    UTC timestamp of forecast generation.
        observation_count: Number of historical readings used.
    """

    station_id: str
    target_date_utc: datetime
    distribution: ForecastDistribution
    nws_point_forecast_f: Optional[float] = None
    historical_bias_f: float = 0.0
    model_version: str = "v1.0"
    generated_at: datetime
    observation_count: int = 0

    @computed_field
    @property
    def mean_f(self) -> float:
        """Convenience alias for distribution.mean_f.

        Returns:
            Mean forecast temperature in Fahrenheit.
        """
        return self.distribution.mean_f

    @computed_field
    @property
    def std_f(self) -> float:
        """Convenience alias for distribution.std_f.

        Returns:
            Standard deviation of the forecast in degrees Fahrenheit.
        """
        return self.distribution.std_f
