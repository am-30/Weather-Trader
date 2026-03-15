"""
ASOS weather data fetcher for KBOS (Boston Logan Airport).

Primary source: NWS API ``/stations/KBOS/observations/latest``
Fallback source: IEM Mesonet JSON API (used when NWS data is stale > 15 min)

Also provides ``fetch_last_n_hours()`` for calibration use.
All observations are persisted via db_manager and the hard floor is updated.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from kalshi_weather_trader.config.settings import settings
from kalshi_weather_trader.db import db_manager
from kalshi_weather_trader.db.schemas import ASOSReadingDocument

logger = structlog.get_logger(__name__)

_RETRY_EXCEPTIONS = (httpx.HTTPError, httpx.TimeoutException)
_DEFAULT_TIMEOUT = httpx.Timeout(15.0, connect=5.0)
_NWS_HEADERS = {
    "User-Agent": "KalshiWeatherTrader/1.0 (automated trading system - contact@example.com)",
    "Accept": "application/geo+json",
}


# ---------------------------------------------------------------------------
# Internal HTTP helpers
# ---------------------------------------------------------------------------


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(_RETRY_EXCEPTIONS),
    reraise=True,
)
def _get_nws(url: str, params: Optional[dict] = None) -> dict:
    """Execute a GET request to the NWS API with retry logic.

    Args:
        url:    Full URL.
        params: Optional query parameters.

    Returns:
        Parsed JSON dict.

    Raises:
        httpx.HTTPStatusError: On 4xx/5xx after retries.
        httpx.TimeoutException: On timeout after retries.
    """
    with httpx.Client(timeout=_DEFAULT_TIMEOUT) as client:
        response = client.get(url, params=params, headers=_NWS_HEADERS)
        response.raise_for_status()
        return response.json()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(_RETRY_EXCEPTIONS),
    reraise=True,
)
def _get_iem(url: str, params: Optional[dict] = None) -> dict:
    """Execute a GET request to the IEM Mesonet API with retry logic.

    Args:
        url:    Full URL.
        params: Optional query parameters.

    Returns:
        Parsed JSON dict.

    Raises:
        httpx.HTTPStatusError: On 4xx/5xx after retries.
        httpx.TimeoutException: On timeout after retries.
    """
    with httpx.Client(timeout=_DEFAULT_TIMEOUT) as client:
        response = client.get(url, params=params)
        response.raise_for_status()
        return response.json()


# ---------------------------------------------------------------------------
# Unit converters
# ---------------------------------------------------------------------------


def _c_to_f(celsius: Optional[float]) -> Optional[float]:
    """Convert Celsius to Fahrenheit, rounded to 1 d.p.

    Args:
        celsius: Temperature in °C, or None.

    Returns:
        Temperature in °F rounded to 1 d.p., or None.

    Raises:
        Nothing.
    """
    if celsius is None:
        return None
    return round((celsius * 9.0 / 5.0) + 32.0, 1)


def _mph(ms: Optional[float]) -> Optional[float]:
    """Convert metres-per-second to miles-per-hour.

    Args:
        ms: Speed in m/s, or None.

    Returns:
        Speed in mph rounded to 1 d.p., or None.

    Raises:
        Nothing.
    """
    if ms is None:
        return None
    return round(ms * 2.23694, 1)


# ---------------------------------------------------------------------------
# NWS primary fetch
# ---------------------------------------------------------------------------


def _fetch_nws_latest() -> Optional[ASOSReadingDocument]:
    """Fetch the latest ASOS observation from the NWS API.

    Args:
        None

    Returns:
        ``ASOSReadingDocument`` if a valid, fresh reading was retrieved.
        None if the NWS response is unavailable or stale.

    Raises:
        Nothing — all exceptions are caught and logged.
    """
    url = f"{settings.nws_api_base_url}/stations/{settings.nws_station}/observations/latest"
    try:
        data = _get_nws(url)
    except Exception as exc:
        logger.warning("asos.nws_fetch.failed", error=str(exc))
        return None

    props = data.get("properties", {})
    timestamp_raw = props.get("timestamp")
    if not timestamp_raw:
        logger.warning("asos.nws_fetch.no_timestamp")
        return None

    try:
        obs_time = datetime.fromisoformat(timestamp_raw.replace("Z", "+00:00"))
    except ValueError as exc:
        logger.warning("asos.nws_fetch.bad_timestamp", raw=timestamp_raw, error=str(exc))
        return None

    # Staleness check
    age_minutes = (datetime.now(timezone.utc) - obs_time).total_seconds() / 60
    if age_minutes > settings.asos_staleness_minutes:
        logger.info(
            "asos.nws_fetch.stale",
            age_minutes=round(age_minutes, 1),
            threshold=settings.asos_staleness_minutes,
        )
        return None

    temp_c = (props.get("temperature") or {}).get("value")
    if temp_c is None:
        logger.warning("asos.nws_fetch.no_temperature")
        return None

    dew_c = (props.get("dewpoint") or {}).get("value")
    wind_ms = (props.get("windSpeed") or {}).get("value")

    return ASOSReadingDocument(
        station_id=settings.nws_station,
        observation_time_utc=obs_time,
        temperature_f=_c_to_f(temp_c),  # type: ignore[arg-type]
        dew_point_f=_c_to_f(dew_c),
        wind_speed_mph=_mph(wind_ms),
        raw_metar=props.get("rawMessage"),
    )


# ---------------------------------------------------------------------------
# IEM Mesonet fallback fetch
# ---------------------------------------------------------------------------


def _fetch_iem_current() -> Optional[ASOSReadingDocument]:
    """Fetch the current ASOS observation from the IEM Mesonet CSV endpoint.

    Uses a 60-minute look-back window and returns the most recent reading
    that has a valid temperature.  IEM's ``/cgi-bin/request/asos.py`` is
    used because the JSON endpoint has been deprecated/moved.

    Args:
        None

    Returns:
        ``ASOSReadingDocument`` if the IEM API returns a valid reading.
        None on failure.

    Raises:
        Nothing — all exceptions are caught and logged.
    """
    url = f"{settings.iem_api_base_url}/cgi-bin/request/asos.py"
    now_utc = datetime.now(timezone.utc)
    since_utc = now_utc - timedelta(hours=1)
    params = {
        "station": settings.nws_station,
        "data": "tmpf,dwpf,sknt",
        "year1": since_utc.year,
        "month1": since_utc.month,
        "day1": since_utc.day,
        "hour1": since_utc.hour,
        "minute1": since_utc.minute,
        "year2": now_utc.year,
        "month2": now_utc.month,
        "day2": now_utc.day,
        "hour2": now_utc.hour,
        "minute2": now_utc.minute,
        "tz": "UTC",
        "format": "onlycomma",
        "latlon": "no",
        "direct": "no",
    }

    try:
        with httpx.Client(timeout=_DEFAULT_TIMEOUT) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            csv_text = response.text
    except Exception as exc:
        logger.warning("asos.iem_fetch.failed", error=str(exc))
        return None

    try:
        lines = [ln for ln in csv_text.splitlines() if ln and not ln.startswith("#")]
        if len(lines) < 2:
            # First line is header; need at least one data row
            logger.warning("asos.iem_fetch.no_data")
            return None

        header = [h.strip() for h in lines[0].split(",")]

        # Walk rows newest-first to find one with a valid temperature
        for raw_row in reversed(lines[1:]):
            row = dict(zip(header, [v.strip() for v in raw_row.split(",")]))

            valid_raw = row.get("valid")
            if not valid_raw:
                continue

            obs_time = datetime.fromisoformat(valid_raw.replace(" ", "T") + "+00:00")

            tmpf = row.get("tmpf")
            if not tmpf or tmpf == "M":
                continue

            temp_f = round(float(tmpf), 1)

            dwpf = row.get("dwpf")
            dew_f = round(float(dwpf), 1) if dwpf and dwpf != "M" else None

            sknt = row.get("sknt")
            wind_mph = round(float(sknt) * 1.15078, 1) if sknt and sknt != "M" else None

            return ASOSReadingDocument(
                station_id=settings.nws_station,
                observation_time_utc=obs_time,
                temperature_f=temp_f,
                dew_point_f=dew_f,
                wind_speed_mph=wind_mph,
            )

        logger.warning("asos.iem_fetch.no_valid_temperature_row")
        return None
    except Exception as exc:
        logger.error("asos.iem_fetch.parse_error", error=str(exc), exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def fetch_current_observation() -> Optional[ASOSReadingDocument]:
    """Fetch the latest KBOS ASOS observation, persisting it to the database.

    Tries NWS API first; falls back to IEM Mesonet if NWS is stale or fails.
    Updates the hard floor in the markets table on success.

    Args:
        None

    Returns:
        ``ASOSReadingDocument`` with the latest reading, or None on total failure.

    Raises:
        Nothing — all exceptions are logged.
    """
    from kalshi_weather_trader.config.settings import get_target_date

    reading = _fetch_nws_latest()
    source = "NWS"

    if reading is None:
        logger.info("asos.fetch.falling_back_to_iem")
        reading = _fetch_iem_current()
        source = "IEM"

    if reading is None:
        logger.error("asos.fetch.all_sources_failed")
        return None

    logger.info(
        "asos.fetch.success",
        source=source,
        temp_f=reading.temperature_f,
        time=str(reading.observation_time_utc),
    )

    # Persist to DB
    try:
        db_manager.upsert_asos_reading(reading)
    except Exception as exc:
        logger.error("asos.fetch.persist_failed", error=str(exc))
        # Don't return None — we still have the reading even if DB write failed

    # Update hard floor
    try:
        target_date = get_target_date()
        db_manager.update_hard_floor(target_date, reading.temperature_f)
    except Exception as exc:
        logger.error("asos.fetch.hard_floor_update_failed", error=str(exc))

    return reading


def fetch_last_n_hours(hours: int = 24) -> list[ASOSReadingDocument]:
    """Fetch up to N hours of historical ASOS data from IEM Mesonet.

    Used by the calibrator to compute sigma and theta from recent data.

    Args:
        hours: Number of hours of history to retrieve. Defaults to 24.

    Returns:
        List of ``ASOSReadingDocument`` sorted oldest-first.

    Raises:
        Nothing — errors are logged and an empty list is returned on failure.
    """
    url = f"{settings.iem_api_base_url}/cgi-bin/request/asos.py"
    now_utc = datetime.now(timezone.utc)
    since_utc = now_utc - timedelta(hours=hours)

    params = {
        "station": settings.nws_station,
        "data": "tmpf,dwpf,sknt",
        "year1": since_utc.year,
        "month1": since_utc.month,
        "day1": since_utc.day,
        "hour1": since_utc.hour,
        "minute1": since_utc.minute,
        "year2": now_utc.year,
        "month2": now_utc.month,
        "day2": now_utc.day,
        "hour2": now_utc.hour,
        "minute2": now_utc.minute,
        "tz": "UTC",
        "format": "onlycomma",
        "latlon": "no",
        "direct": "no",
    }

    try:
        with httpx.Client(timeout=httpx.Timeout(30.0, connect=5.0)) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            csv_text = response.text
    except Exception as exc:
        logger.error("asos.fetch_history.failed", hours=hours, error=str(exc))
        return []

    readings: list[ASOSReadingDocument] = []
    lines = [ln for ln in csv_text.splitlines() if ln and not ln.startswith("#")]
    if len(lines) < 2:
        logger.info("asos.fetch_history.done", count=0, hours=hours)
        return []

    header = [h.strip() for h in lines[0].split(",")]
    for raw_row in lines[1:]:
        try:
            row = dict(zip(header, [v.strip() for v in raw_row.split(",")]))
            valid_raw = row.get("valid")
            if not valid_raw:
                continue
            obs_time = datetime.fromisoformat(valid_raw.replace(" ", "T") + "+00:00")

            tmpf = row.get("tmpf")
            if not tmpf or tmpf == "M":
                continue

            dwpf = row.get("dwpf")
            dew_f = round(float(dwpf), 1) if dwpf and dwpf != "M" else None

            sknt = row.get("sknt")
            wind_mph = round(float(sknt) * 1.15078, 1) if sknt and sknt != "M" else None

            readings.append(
                ASOSReadingDocument(
                    station_id=settings.nws_station,
                    observation_time_utc=obs_time,
                    temperature_f=round(float(tmpf), 1),
                    dew_point_f=dew_f,
                    wind_speed_mph=wind_mph,
                )
            )
        except Exception as exc:
            logger.warning("asos.fetch_history.parse_error", row=raw_row, error=str(exc))
            continue

    readings.sort(key=lambda r: r.observation_time_utc)
    logger.info("asos.fetch_history.done", count=len(readings), hours=hours)
    return readings
