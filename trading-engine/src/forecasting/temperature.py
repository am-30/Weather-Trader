"""
Probabilistic daily maximum temperature forecasting for KBOS.

Strategy
--------
1. Fetch the NWS gridpoint hourly forecast for the target date and extract
   the maximum predicted temperature as the primary point estimate.
2. Pull the last N days of observed daily maxima from the database to compute
   a historical bias (mean NWS error) and a residual standard deviation.
3. Combine these into a Gaussian ``ForecastDistribution`` centred on the
   bias-corrected NWS point estimate.
4. Persist the forecast record to ``temperature_forecasts``.

All temperatures in Fahrenheit. All timestamps in UTC.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import structlog

from src.config import settings
from src.data_feeds.nws import fetch_gridpoint_forecast
from src.db.connection import get_connection
from src.db.schema import log_system_event
from src.models.forecast import ForecastDistribution, TemperatureForecast

logger = structlog.get_logger(__name__)

_HISTORY_DAYS = 30
_MIN_STD_F = 2.5
_DEFAULT_STD_F = 5.0
_MODEL_VERSION = "v1.0"


def _fetch_historical_maxima(
    station_id: str,
    days: int = _HISTORY_DAYS,
) -> list[float]:
    """Read the last N daily max temperatures from the database.

    Args:
        station_id: ICAO station code.
        days:       Number of days of history to retrieve.

    Returns:
        List of max temperatures (Fahrenheit floats), oldest-first.

    Raises:
        psycopg2.Error: On database read failure.
    """
    since = datetime.now(timezone.utc) - timedelta(days=days)
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT max_temp_f
                    FROM daily_max_observations
                    WHERE station_id = %s AND date_utc >= %s
                    ORDER BY date_utc ASC
                    """,
                    (station_id, since.date()),
                )
                rows = cur.fetchall()
        return [float(row[0]) for row in rows]
    except Exception as exc:
        logger.error("forecast.fetch_history.failed", error=str(exc), exc_info=True)
        return []


def _compute_bias_and_std(
    historical_maxima: list[float],
    nws_forecast_f: float,
) -> tuple[float, float]:
    """Compute a simple bias correction and residual std from history.

    Uses the distribution of recent daily maxima as a proxy for forecast
    uncertainty. In a production system this would regress historical NWS
    errors against truth; here we use the historical std as a conservative
    uncertainty estimate.

    Args:
        historical_maxima: List of observed daily max temps.
        nws_forecast_f:    NWS point estimate for the target day.

    Returns:
        Tuple of (bias_correction_f, std_f).
    """
    if len(historical_maxima) < 3:
        logger.warning("forecast.bias.insufficient_history", count=len(historical_maxima))
        return 0.0, _DEFAULT_STD_F

    arr = np.array(historical_maxima, dtype=float)
    hist_mean = float(np.mean(arr))
    hist_std = float(np.std(arr, ddof=1))

    bias = round(hist_mean - nws_forecast_f, 1)
    std = max(hist_std, _MIN_STD_F)

    logger.debug(
        "forecast.bias_std",
        hist_mean=round(hist_mean, 1),
        nws_f=round(nws_forecast_f, 1),
        bias=bias,
        std=round(std, 1),
        n=len(historical_maxima),
    )
    return bias, round(std, 1)


def _nws_max_for_date(
    periods: list,
    target_date_utc: datetime,
    station_tz_offset: int = -5,
) -> Optional[float]:
    """Extract the maximum forecast temperature for a given UTC calendar date.

    NWS periods use local time internally; we convert the target UTC date to
    a local window using a fixed offset.

    Args:
        periods:          List of ``NWSGridForecastPeriod`` objects.
        target_date_utc:  UTC calendar date to extract the max for.
        station_tz_offset: UTC offset for the station's local timezone.

    Returns:
        Max Fahrenheit temperature for that day, or None if no periods found.
    """
    local_offset = timedelta(hours=station_tz_offset)
    day_start_local = (target_date_utc + local_offset).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    day_end_local = day_start_local + timedelta(days=1)

    temps = []
    for period in periods:
        period_local = period.start_time + local_offset
        if day_start_local <= period_local < day_end_local:
            temps.append(period.temp_f)

    return max(temps) if temps else None


def generate_forecast(
    station_id: str | None = None,
    target_date_utc: datetime | None = None,
) -> TemperatureForecast:
    """Generate a probabilistic daily max temperature forecast.

    Fetches the NWS gridpoint hourly forecast, determines the NWS point
    estimate for ``target_date_utc``, applies a bias correction derived from
    historical observations, and returns a ``TemperatureForecast`` with a
    Gaussian distribution.

    Args:
        station_id:      ICAO station code. Defaults to ``settings.nws_station``.
        target_date_utc: UTC calendar date to forecast. Defaults to today UTC.

    Returns:
        ``TemperatureForecast`` with populated distribution.

    Raises:
        RuntimeError: If the NWS gridpoint forecast cannot be retrieved.
    """
    station = station_id or settings.nws_station
    now_utc = datetime.now(timezone.utc)
    target = target_date_utc or now_utc.replace(hour=0, minute=0, second=0, microsecond=0)

    logger.info("forecast.generate.start", station=station, target_date=target.date().isoformat())

    try:
        periods = fetch_gridpoint_forecast(station)
    except Exception as exc:
        logger.error("forecast.generate.nws_failed", error=str(exc))
        raise RuntimeError(f"Cannot generate forecast: NWS API failed — {exc}") from exc

    nws_point_f = _nws_max_for_date(periods, target)
    if nws_point_f is None:
        logger.warning(
            "forecast.generate.no_nws_periods",
            date=target.date().isoformat(),
            total_periods=len(periods),
        )
        nws_point_f = 60.0

    historical = _fetch_historical_maxima(station)
    bias_f, std_f = _compute_bias_and_std(historical, nws_point_f)

    corrected_mean_f = round(nws_point_f + bias_f, 1)
    distribution = ForecastDistribution(mean_f=corrected_mean_f, std_f=std_f)

    forecast = TemperatureForecast(
        station_id=station,
        target_date_utc=target,
        distribution=distribution,
        nws_point_forecast_f=nws_point_f,
        historical_bias_f=bias_f,
        model_version=_MODEL_VERSION,
        generated_at=now_utc,
        observation_count=len(historical),
    )

    logger.info(
        "forecast.generate.complete",
        station=station,
        date=target.date().isoformat(),
        mean_f=corrected_mean_f,
        std_f=std_f,
        nws_point_f=nws_point_f,
        bias_f=bias_f,
    )

    _persist_forecast(forecast)
    return forecast


def _persist_forecast(forecast: TemperatureForecast) -> None:
    """Persist a TemperatureForecast record to the database.

    Args:
        forecast: The forecast to persist.

    Returns:
        None

    Raises:
        Nothing — errors are logged but not re-raised.
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO temperature_forecasts
                        (station_id, target_date_utc, mean_f, std_f,
                         nws_point_forecast_f, historical_bias_f,
                         model_version, observation_count, generated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        forecast.station_id,
                        forecast.target_date_utc.date(),
                        forecast.mean_f,
                        forecast.std_f,
                        forecast.nws_point_forecast_f,
                        forecast.historical_bias_f,
                        forecast.model_version,
                        forecast.observation_count,
                        forecast.generated_at,
                    ),
                )
        logger.info("forecast.persisted", date=forecast.target_date_utc.date().isoformat())
    except Exception as exc:
        logger.error("forecast.persist.failed", error=str(exc), exc_info=True)
        log_system_event("forecast.persist.error", str(exc), level="error")


def get_latest_forecast(
    station_id: str | None = None,
    target_date_utc: datetime | None = None,
) -> Optional[TemperatureForecast]:
    """Retrieve the most recently generated forecast from the database.

    Args:
        station_id:      ICAO station code.
        target_date_utc: UTC calendar date. Defaults to today UTC.

    Returns:
        The latest ``TemperatureForecast`` for the date, or None if not found.

    Raises:
        psycopg2.Error: On database read failure.
    """
    station = station_id or settings.nws_station
    now_utc = datetime.now(timezone.utc)
    target = target_date_utc or now_utc.replace(hour=0, minute=0, second=0, microsecond=0)

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT mean_f, std_f, nws_point_forecast_f, historical_bias_f,
                           model_version, observation_count, generated_at
                    FROM temperature_forecasts
                    WHERE station_id = %s AND target_date_utc = %s
                    ORDER BY generated_at DESC
                    LIMIT 1
                    """,
                    (station, target.date()),
                )
                row = cur.fetchone()
    except Exception as exc:
        logger.error("forecast.get_latest.failed", error=str(exc), exc_info=True)
        return None

    if not row:
        return None

    mean_f, std_f, nws_f, bias_f, model_v, obs_count, gen_at = row
    return TemperatureForecast(
        station_id=station,
        target_date_utc=target,
        distribution=ForecastDistribution(mean_f=float(mean_f), std_f=float(std_f)),
        nws_point_forecast_f=float(nws_f) if nws_f is not None else None,
        historical_bias_f=float(bias_f),
        model_version=model_v,
        generated_at=gen_at,
        observation_count=int(obs_count),
    )
