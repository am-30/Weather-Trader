"""
National Weather Service (weather.gov) API client.

Fetches hourly observations from a given ICAO station and retrieves
gridpoint hourly forecast data. All API calls are wrapped with
tenacity exponential-backoff retry logic (max 3 attempts).

NWS API documentation: https://www.weather.gov/documentation/services-web-api
"""

from __future__ import annotations

import math
from datetime import datetime, timezone, timedelta
from typing import Optional

import httpx
import structlog
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from src.config import settings
from src.models.weather import WeatherObservation, DailyMaxObservation, NWSGridForecastPeriod
from src.db.connection import get_connection
from src.db.schema import log_system_event

logger = structlog.get_logger(__name__)

_RETRY_EXCEPTIONS = (httpx.HTTPError, httpx.TimeoutException)
_DEFAULT_TIMEOUT = httpx.Timeout(15.0, connect=5.0)


def _celsius_to_fahrenheit(celsius: Optional[float]) -> Optional[float]:
    """Convert Celsius to Fahrenheit, returning None for None input.

    Args:
        celsius: Temperature in degrees Celsius, or None.

    Returns:
        Temperature in Fahrenheit rounded to 1 d.p., or None.
    """
    if celsius is None:
        return None
    return round((celsius * 9.0 / 5.0) + 32.0, 1)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(_RETRY_EXCEPTIONS),
    before_sleep=before_sleep_log(logger, "warning"),
    reraise=True,
)
def _get(url: str, params: dict | None = None) -> dict:
    """Perform a GET request to the NWS API with retry logic.

    Args:
        url:    Full URL to request.
        params: Optional query parameters.

    Returns:
        Parsed JSON response as a dict.

    Raises:
        httpx.HTTPStatusError: If the server returns a 4xx/5xx response.
        httpx.TimeoutException: If the request times out after retries.
    """
    headers = {
        "User-Agent": "KalshiWeatherTrader/1.0 (contact@example.com)",
        "Accept": "application/geo+json",
    }
    with httpx.Client(timeout=_DEFAULT_TIMEOUT) as client:
        response = client.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()


def fetch_recent_observations(
    station_id: str | None = None,
    hours: int = 25,
) -> list[WeatherObservation]:
    """Fetch the most recent hourly observations from a NWS station.

    Args:
        station_id: ICAO station code. Defaults to ``settings.nws_station``.
        hours:      How many hours of history to request. Defaults to 25.

    Returns:
        List of ``WeatherObservation`` objects sorted oldest-first.

    Raises:
        httpx.HTTPError: If the NWS API returns an error after retries.
        ValueError: If the API response cannot be parsed.
    """
    station = station_id or settings.nws_station
    url = f"{settings.nws_api_base_url}/stations/{station}/observations"
    params = {"limit": hours}

    logger.info("nws.fetch_observations.start", station=station, hours=hours)

    try:
        data = _get(url, params=params)
    except Exception as exc:
        logger.error("nws.fetch_observations.failed", station=station, error=str(exc))
        log_system_event("nws.fetch.error", str(exc), level="error")
        raise

    observations: list[WeatherObservation] = []
    for feature in data.get("features", []):
        props = feature.get("properties", {})
        try:
            temp_c = props.get("temperature", {}).get("value")
            dew_c = props.get("dewpoint", {}).get("value")
            wind_ms = props.get("windSpeed", {}).get("value")
            wind_dir = props.get("windDirection", {}).get("value")
            timestamp_raw = props.get("timestamp")
            if not timestamp_raw or temp_c is None:
                continue

            wind_mph = round(wind_ms * 2.23694, 1) if wind_ms is not None else None

            obs = WeatherObservation(
                station_id=station,
                observed_at=datetime.fromisoformat(timestamp_raw),
                temp_f=_celsius_to_fahrenheit(temp_c),
                dew_point_f=_celsius_to_fahrenheit(dew_c),
                wind_speed_mph=wind_mph,
                wind_dir_deg=float(wind_dir) if wind_dir is not None else None,
                raw_text=props.get("rawMessage"),
            )
            observations.append(obs)
        except Exception as exc:
            logger.warning("nws.observation.parse_error", error=str(exc))
            continue

    observations.sort(key=lambda o: o.observed_at)
    logger.info("nws.fetch_observations.done", count=len(observations))
    return observations


def persist_observations(observations: list[WeatherObservation]) -> int:
    """Upsert weather observations into the database.

    Uses ``ON CONFLICT DO NOTHING`` so duplicate fetches are idempotent.

    Args:
        observations: List of ``WeatherObservation`` records to persist.

    Returns:
        Number of rows actually inserted (excludes duplicates).

    Raises:
        psycopg2.Error: On database write failure.
    """
    if not observations:
        return 0

    inserted = 0
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                for obs in observations:
                    cur.execute(
                        """
                        INSERT INTO weather_observations
                            (station_id, observed_at, temp_f, dew_point_f,
                             wind_speed_mph, wind_dir_deg, precip_in, raw_text)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (station_id, observed_at) DO NOTHING
                        """,
                        (
                            obs.station_id,
                            obs.observed_at,
                            obs.temp_f,
                            obs.dew_point_f,
                            obs.wind_speed_mph,
                            obs.wind_dir_deg,
                            obs.precip_in,
                            obs.raw_text,
                        ),
                    )
                    inserted += cur.rowcount
        logger.info("nws.persist_observations.done", inserted=inserted, total=len(observations))
    except Exception as exc:
        logger.error("nws.persist_observations.failed", error=str(exc), exc_info=True)
        log_system_event("nws.persist.error", str(exc), level="error")
        raise

    return inserted


def compute_daily_max(
    station_id: str | None = None,
    date_utc: datetime | None = None,
) -> Optional[DailyMaxObservation]:
    """Compute and persist the daily maximum temperature from stored observations.

    Reads observations from the database for the given UTC calendar day and
    computes the maximum temperature.

    Args:
        station_id: ICAO station code. Defaults to ``settings.nws_station``.
        date_utc:   UTC date to compute for. Defaults to today UTC.

    Returns:
        ``DailyMaxObservation`` if at least one observation exists, else None.

    Raises:
        psycopg2.Error: On database error.
    """
    station = station_id or settings.nws_station
    now_utc = datetime.now(timezone.utc)
    target = date_utc or now_utc.replace(hour=0, minute=0, second=0, microsecond=0)

    day_start = target.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
    day_end = day_start + timedelta(days=1)

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT MAX(temp_f), COUNT(*)
                    FROM weather_observations
                    WHERE station_id = %s
                      AND observed_at >= %s
                      AND observed_at < %s
                    """,
                    (station, day_start, day_end),
                )
                row = cur.fetchone()
    except Exception as exc:
        logger.error("nws.compute_daily_max.read_failed", error=str(exc), exc_info=True)
        raise

    if not row or row[0] is None:
        logger.warning("nws.compute_daily_max.no_data", station=station, date=target.date())
        return None

    max_temp_f, count = float(row[0]), int(row[1])
    result = DailyMaxObservation(
        station_id=station,
        date_utc=day_start,
        max_temp_f=max_temp_f,
        observation_count=count,
        computed_at=now_utc,
    )

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO daily_max_observations
                        (station_id, date_utc, max_temp_f, observation_count, computed_at)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (station_id, date_utc)
                    DO UPDATE SET
                        max_temp_f        = EXCLUDED.max_temp_f,
                        observation_count = EXCLUDED.observation_count,
                        computed_at       = EXCLUDED.computed_at
                    """,
                    (
                        result.station_id,
                        result.date_utc.date(),
                        result.max_temp_f,
                        result.observation_count,
                        result.computed_at,
                    ),
                )
        logger.info(
            "nws.compute_daily_max.done",
            station=station,
            date=target.date(),
            max_temp_f=max_temp_f,
            observations=count,
        )
    except Exception as exc:
        logger.error("nws.compute_daily_max.write_failed", error=str(exc), exc_info=True)
        log_system_event("nws.daily_max.write_error", str(exc), level="error")
        raise

    return result


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(_RETRY_EXCEPTIONS),
    before_sleep=before_sleep_log(logger, "warning"),
    reraise=True,
)
def fetch_gridpoint_forecast(
    station_id: str | None = None,
) -> list[NWSGridForecastPeriod]:
    """Fetch hourly gridpoint forecast from NWS for KBOS.

    This two-step call first resolves the station's grid coordinates, then
    fetches the hourly forecast for the next ~156 hours.

    Args:
        station_id: ICAO station code. Defaults to ``settings.nws_station``.

    Returns:
        List of ``NWSGridForecastPeriod`` objects, oldest-first.

    Raises:
        httpx.HTTPError: If any NWS API call fails after retries.
        ValueError: If the grid metadata response is malformed.
    """
    station = station_id or settings.nws_station
    logger.info("nws.fetch_gridpoint_forecast.start", station=station)

    points_url = f"{settings.nws_api_base_url}/stations/{station}"
    try:
        station_meta = _get(points_url)
    except Exception as exc:
        logger.error("nws.gridpoint.station_meta_failed", error=str(exc))
        raise

    forecast_url = (
        f"{settings.nws_api_base_url}/gridpoints/"
        f"{station_meta['properties']['gridId']}/"
        f"{station_meta['properties']['gridX']},"
        f"{station_meta['properties']['gridY']}/forecast/hourly"
    )

    try:
        forecast_data = _get(forecast_url)
    except Exception as exc:
        logger.error("nws.gridpoint.forecast_failed", error=str(exc))
        raise

    periods: list[NWSGridForecastPeriod] = []
    for period in forecast_data.get("properties", {}).get("periods", []):
        try:
            temp_raw = period.get("temperature")
            unit = period.get("temperatureUnit", "F")
            if temp_raw is None:
                continue
            temp_f = float(temp_raw) if unit == "F" else _celsius_to_fahrenheit(float(temp_raw))
            if temp_f is None:
                continue

            periods.append(
                NWSGridForecastPeriod(
                    start_time=datetime.fromisoformat(period["startTime"]),
                    end_time=datetime.fromisoformat(period["endTime"]),
                    temp_f=temp_f,
                    is_daytime=period.get("isDaytime", True),
                    short_forecast=period.get("shortForecast", ""),
                )
            )
        except Exception as exc:
            logger.warning("nws.gridpoint.period_parse_error", error=str(exc))
            continue

    logger.info("nws.fetch_gridpoint_forecast.done", periods=len(periods))
    return periods


def run_weather_cycle(station_id: str | None = None) -> dict:
    """Run a complete weather data cycle: fetch, persist, compute daily max.

    This is the top-level function called by the scheduler on each weather
    fetch interval.

    Args:
        station_id: ICAO station code. Defaults to ``settings.nws_station``.

    Returns:
        Dict with keys ``observations_fetched``, ``observations_inserted``,
        and ``daily_max_f``.

    Raises:
        Nothing — errors are logged and returned in the result dict.
    """
    station = station_id or settings.nws_station
    result: dict = {"observations_fetched": 0, "observations_inserted": 0, "daily_max_f": None}

    try:
        observations = fetch_recent_observations(station, hours=25)
        result["observations_fetched"] = len(observations)
        result["observations_inserted"] = persist_observations(observations)
    except Exception as exc:
        logger.error("nws.run_cycle.fetch_failed", error=str(exc))
        return result

    try:
        daily_max = compute_daily_max(station)
        if daily_max:
            result["daily_max_f"] = daily_max.max_temp_f
    except Exception as exc:
        logger.error("nws.run_cycle.daily_max_failed", error=str(exc))

    log_system_event("nws.cycle.complete", f"Fetched {result['observations_fetched']} obs", details=result)
    return result
