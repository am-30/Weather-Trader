"""
ASOS weather data fetcher for KBOS (Boston Logan Airport).

Fetch priority on each scheduler tick:
  1. IEM Mesonet bulk gap-fill (primary) — all readings since last stored
  2. Aviation Weather Center METAR JSON (secondary) — if IEM returns nothing
  3. NWS /observations/latest (last resort) — also provides max6h_f for hard floor

A module-level rate-limit guard prevents calling any external API more often
than ``settings.asos_min_fetch_interval_minutes`` (default 4 min), so reducing
the scheduler interval to 2 min does not increase server load.

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
def _get_iem(url: str, params: Optional[dict] = None) -> str:
    """Execute a GET request to the IEM Mesonet CSV endpoint with retry logic.

    Args:
        url:    Full URL.
        params: Optional query parameters.

    Returns:
        Response body as text (IEM returns CSV, not JSON).

    Raises:
        httpx.HTTPStatusError: On 4xx/5xx after retries.
        httpx.TimeoutException: On timeout after retries.
    """
    with httpx.Client(timeout=_DEFAULT_TIMEOUT) as client:
        response = client.get(url, params=params)
        response.raise_for_status()
        return response.text


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(_RETRY_EXCEPTIONS),
    reraise=True,
)
def _get_avwx(url: str, params: Optional[dict] = None) -> list:
    """Execute a GET request to the Aviation Weather Center API with retry logic.

    Args:
        url:    Full URL.
        params: Optional query parameters.

    Returns:
        Parsed JSON list (AVWX METAR endpoint returns a JSON array).

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


def _fetch_nws_latest() -> tuple[Optional[ASOSReadingDocument], Optional[float]]:
    """Fetch the latest ASOS observation from the NWS API.

    Also extracts the 6-hour maximum temperature field (``maxTemperatureLast6Hours``)
    from the NWS GeoJSON response.  The ASOS sensor's 0.5°C persistence filter means
    the tabular temperature can be 0.2–0.4°F below the true intraday peak near bucket
    boundaries; the 6-hour max captures sub-threshold spikes and is the more reliable
    hard-floor source.

    Args:
        None

    Returns:
        Tuple of (``ASOSReadingDocument``, max6h_temp_f).  The reading is None if the
        NWS response is unavailable or stale.  max6h_temp_f is None if the field is
        absent from the NWS response (common outside the 6-hour update window).

    Raises:
        Nothing — all exceptions are caught and logged.
    """
    url = f"{settings.nws_api_base_url}/stations/{settings.nws_station}/observations/latest"
    try:
        data = _get_nws(url)
    except Exception as exc:
        logger.warning("asos.nws_fetch.failed", error=str(exc))
        return None, None

    props = data.get("properties", {})
    timestamp_raw = props.get("timestamp")
    if not timestamp_raw:
        logger.warning("asos.nws_fetch.no_timestamp")
        return None, None

    try:
        obs_time = datetime.fromisoformat(timestamp_raw.replace("Z", "+00:00"))
    except ValueError as exc:
        logger.warning("asos.nws_fetch.bad_timestamp", raw=timestamp_raw, error=str(exc))
        return None, None

    # Staleness check
    age_minutes = (datetime.now(timezone.utc) - obs_time).total_seconds() / 60
    if age_minutes > settings.asos_staleness_minutes:
        logger.info(
            "asos.nws_fetch.stale",
            age_minutes=round(age_minutes, 1),
            threshold=settings.asos_staleness_minutes,
        )
        return None, None

    temp_c = (props.get("temperature") or {}).get("value")
    if temp_c is None:
        logger.warning("asos.nws_fetch.no_temperature")
        return None, None

    dew_c = (props.get("dewpoint") or {}).get("value")
    wind_ms = (props.get("windSpeed") or {}).get("value")

    # 6-hour maximum temperature — captures sub-threshold intraday peaks that the
    # persistence filter suppresses in the tabular temperature field.
    max6h_c = (props.get("maxTemperatureLast6Hours") or {}).get("value")
    max6h_f = _c_to_f(max6h_c)

    reading = ASOSReadingDocument(
        station_id=settings.nws_station,
        observation_time_utc=obs_time,
        temperature_f=_c_to_f(temp_c),  # type: ignore[arg-type]
        dew_point_f=_c_to_f(dew_c),
        wind_speed_mph=_mph(wind_ms),
        raw_metar=props.get("rawMessage"),
    )
    return reading, max6h_f


# ---------------------------------------------------------------------------
# IEM Mesonet bulk fetch (primary source)
# ---------------------------------------------------------------------------


def _parse_iem_csv_rows(csv_text: str, skip_at_or_before: Optional[datetime] = None) -> list[ASOSReadingDocument]:
    """Parse IEM ASOS CSV response into a list of readings.

    Args:
        csv_text:            Raw CSV text from the IEM endpoint.
        skip_at_or_before:   If provided, skip any row whose observation_time_utc
                             is <= this value (used to avoid re-storing observations
                             we already have).

    Returns:
        List of ``ASOSReadingDocument`` sorted oldest-first. Empty on parse failure.

    Raises:
        Nothing — row-level errors are logged and skipped.
    """
    readings: list[ASOSReadingDocument] = []
    lines = [ln for ln in csv_text.splitlines() if ln and not ln.startswith("#")]
    if len(lines) < 2:
        return readings

    header = [h.strip() for h in lines[0].split(",")]
    for raw_row in lines[1:]:
        try:
            row = dict(zip(header, [v.strip() for v in raw_row.split(",")]))
            valid_raw = row.get("valid")
            if not valid_raw:
                continue

            valid_norm = " ".join(valid_raw.strip().split())
            try:
                obs_time = datetime.fromisoformat(valid_norm.replace(" ", "T") + "+00:00")
            except ValueError:
                try:
                    obs_time = datetime.strptime(valid_norm, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
                except ValueError:
                    logger.warning("asos.iem_parse.bad_timestamp", raw=valid_raw)
                    continue

            if skip_at_or_before is not None and obs_time <= skip_at_or_before:
                continue

            tmpf = row.get("tmpf")
            if not tmpf or tmpf == "M":
                continue

            dwpf = row.get("dwpf")
            dew_f = round(float(dwpf), 1) if dwpf and dwpf != "M" else None

            sknt = row.get("sknt")
            wind_mph = round(float(sknt) * 1.15078, 1) if sknt and sknt != "M" else None

            readings.append(ASOSReadingDocument(
                station_id=settings.nws_station,
                observation_time_utc=obs_time,
                temperature_f=round(float(tmpf), 1),
                dew_point_f=dew_f,
                wind_speed_mph=wind_mph,
            ))
        except Exception as exc:
            logger.warning("asos.iem_parse.row_error", row=raw_row, error=str(exc))
            continue

    readings.sort(key=lambda r: r.observation_time_utc)
    return readings


def _fetch_iem_since(since_utc: datetime) -> list[ASOSReadingDocument]:
    """Fetch all ASOS readings from IEM Mesonet since a given UTC timestamp.

    Makes a single CSV request covering the window [since_utc, now].  Returns
    every reading newer than ``since_utc``.  IEM's ``on_conflict_do_nothing``
    upsert means duplicate timestamps are safe to re-submit if they sneak in.

    Args:
        since_utc: Fetch readings with observation_time_utc strictly after
                   this value.  Pass ``now - 1h`` when no prior reading exists.

    Returns:
        List of ``ASOSReadingDocument`` sorted oldest-first.
        Empty list on failure or when no new data is available.

    Raises:
        Nothing — all exceptions are caught and logged.
    """
    url = f"{settings.iem_api_base_url}/cgi-bin/request/asos.py"
    now_utc = datetime.now(timezone.utc)
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
        csv_text = _get_iem(url, params)
    except Exception as exc:
        logger.warning("asos.iem_fetch.failed", error=str(exc))
        return []

    readings = _parse_iem_csv_rows(csv_text, skip_at_or_before=since_utc)
    logger.debug("asos.iem_fetch.done", n_new=len(readings), since=str(since_utc))
    return readings


# ---------------------------------------------------------------------------
# Aviation Weather Center METAR fetch (secondary source)
# ---------------------------------------------------------------------------


def _fetch_aviationweather_metar() -> Optional[ASOSReadingDocument]:
    """Fetch the latest KBOS METAR from the Aviation Weather Center API.

    Uses the public ``/api/data/metar`` endpoint which returns decoded JSON.
    KBOS issues routine METARs at :25 and :55 each hour, plus special METARs
    (SPECI) whenever significant conditions change — making this a valuable
    secondary source that can capture readings between IEM update cycles.

    Args:
        None

    Returns:
        ``ASOSReadingDocument`` for the most recent METAR if available and
        not stale.  None on failure or when the newest reading exceeds
        ``settings.asos_staleness_minutes``.

    Raises:
        Nothing — all exceptions are caught and logged.
    """
    url = f"{settings.aviationweather_api_base_url}/api/data/metar"
    params = {
        "ids": settings.nws_station,
        "format": "json",
        "hours": 2,
    }

    try:
        records = _get_avwx(url, params)
    except Exception as exc:
        logger.warning("asos.avwx_fetch.failed", error=str(exc))
        return None

    if not records:
        logger.debug("asos.avwx_fetch.empty")
        return None

    try:
        # Sort newest-first by obsTime (Unix epoch integer)
        records_sorted = sorted(records, key=lambda r: r.get("obsTime", 0), reverse=True)
        latest = records_sorted[0]

        obs_unix = latest.get("obsTime")
        if obs_unix is None:
            logger.warning("asos.avwx_fetch.no_obstime")
            return None

        obs_time = datetime.fromtimestamp(int(obs_unix), tz=timezone.utc)

        age_minutes = (datetime.now(timezone.utc) - obs_time).total_seconds() / 60
        if age_minutes > settings.asos_staleness_minutes:
            logger.info("asos.avwx_fetch.stale", age_minutes=round(age_minutes, 1))
            return None

        temp_c = latest.get("temp")
        if temp_c is None:
            logger.warning("asos.avwx_fetch.no_temperature")
            return None

        dew_c = latest.get("dewp")
        wind_kts = latest.get("wspd")
        wind_mph = round(float(wind_kts) * 1.15078, 1) if wind_kts is not None else None

        return ASOSReadingDocument(
            station_id=settings.nws_station,
            observation_time_utc=obs_time,
            temperature_f=_c_to_f(float(temp_c)),  # type: ignore[arg-type]
            dew_point_f=_c_to_f(float(dew_c)) if dew_c is not None else None,
            wind_speed_mph=wind_mph,
            raw_metar=latest.get("rawOb"),
        )
    except Exception as exc:
        logger.error("asos.avwx_fetch.parse_error", error=str(exc), exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

# Module-level timestamp of the last real API call.  Resets to None on process
# restart, which is intentional — the first call after startup always fetches.
_last_asos_fetch_utc: Optional[datetime] = None


def fetch_current_observation() -> Optional[ASOSReadingDocument]:
    """Fetch new KBOS ASOS readings, persisting all of them to the database.

    Fetch strategy (tried in order until new readings are found):
      1. IEM Mesonet bulk gap-fill — all readings since the last stored timestamp
      2. Aviation Weather Center METAR JSON — single latest reading
      3. NWS /observations/latest — last resort; also provides max6h_f for the
         hard floor to capture sub-threshold peaks the tabular field misses

    A module-level rate-limit guard (``_last_asos_fetch_utc``) prevents making
    any external API call more often than ``settings.asos_min_fetch_interval_minutes``
    (default 4 min).  When the guard fires, the last stored DB reading is returned
    immediately without touching any API.  This allows the scheduler interval to
    be shortened to 2 min for faster reaction to new METARs, while keeping the
    actual API call rate at ≤15/hr — the same as the previous 5-min schedule.

    Args:
        None

    Returns:
        The most recent ``ASOSReadingDocument`` (either newly fetched or the
        last one in the DB), or None if the DB is empty and all sources fail.

    Raises:
        Nothing — all exceptions are logged.
    """
    import math
    from kalshi_weather_trader.config.settings import get_target_date, get_nws_day_bounds

    global _last_asos_fetch_utc

    now_utc = datetime.now(timezone.utc)

    # ---- Rate-limit guard ----
    if _last_asos_fetch_utc is not None:
        elapsed_min = (now_utc - _last_asos_fetch_utc).total_seconds() / 60
        if elapsed_min < settings.asos_min_fetch_interval_minutes:
            logger.debug(
                "asos.fetch.rate_limited",
                elapsed_min=round(elapsed_min, 1),
                threshold=settings.asos_min_fetch_interval_minutes,
            )
            return db_manager.get_latest_asos_reading()

    _last_asos_fetch_utc = now_utc

    # ---- Determine gap-fill window ----
    last_stored = db_manager.get_latest_asos_reading()
    since_utc = (
        last_stored.observation_time_utc
        if last_stored is not None
        else now_utc - timedelta(hours=1)
    )

    # ---- Strategy 1: IEM bulk gap-fill (primary) ----
    new_readings: list[ASOSReadingDocument] = _fetch_iem_since(since_utc)
    max6h_f: Optional[float] = None
    source = "IEM"

    # ---- Strategy 2: Aviation Weather Center METAR (secondary) ----
    if not new_readings:
        avwx = _fetch_aviationweather_metar()
        if avwx is not None:
            new_readings = [avwx]
            source = "AVWX"

    # ---- Strategy 3: NWS /latest (last resort — also provides max6h_f) ----
    if not new_readings:
        nws_reading, max6h_f = _fetch_nws_latest()
        if nws_reading is not None:
            new_readings = [nws_reading]
            source = "NWS"

    if not new_readings:
        logger.error("asos.fetch.all_sources_failed")
        return last_stored  # return cached reading rather than None

    most_recent = new_readings[-1]
    logger.info(
        "asos.fetch.success",
        source=source,
        n_new=len(new_readings),
        latest_temp_f=most_recent.temperature_f,
        max6h_f=max6h_f,
        latest_time=str(most_recent.observation_time_utc),
    )

    # ---- Persist all new readings ----
    for r in new_readings:
        try:
            db_manager.upsert_asos_reading(r)
        except Exception as exc:
            logger.error("asos.fetch.persist_failed", error=str(exc))

    # ---- Update hard floor for each new reading ----
    # Only apply readings that fall within the target date's NWS observation
    # window.  After the 6 PM rollover, get_target_date() returns tomorrow;
    # readings from today must not corrupt tomorrow's hard floor.
    try:
        target_date = get_target_date()
        day_start, _ = get_nws_day_bounds(target_date)

        for r in new_readings:
            if r.observation_time_utc < day_start:
                continue
            floor_temp = r.temperature_f
            # On NWS path, max6h_f captures sub-threshold peaks suppressed by
            # the ASOS 0.5°C persistence filter.  Near a strike boundary the
            # difference (typically 0.2–0.4°F) can shift YES/NO probability by
            # 2–5%.  Only available on the NWS fallback path.
            if max6h_f is not None and max6h_f > floor_temp:
                logger.debug(
                    "asos.fetch.max6h_raised_floor",
                    tabular_f=floor_temp,
                    max6h_f=max6h_f,
                )
                floor_temp = max6h_f
            db_manager.update_hard_floor(target_date, float(math.floor(floor_temp)))
    except Exception as exc:
        logger.error("asos.fetch.hard_floor_update_failed", error=str(exc))

    return most_recent


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
            valid_norm = " ".join(valid_raw.strip().split())
            try:
                obs_time = datetime.fromisoformat(valid_norm.replace(" ", "T") + "+00:00")
            except ValueError:
                try:
                    obs_time = datetime.strptime(valid_norm, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
                except ValueError:
                    logger.warning("asos.fetch_history.bad_timestamp", raw=valid_raw)
                    continue

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
