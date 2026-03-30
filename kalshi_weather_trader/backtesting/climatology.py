"""
Historical climatology baseline for KBOS daily maximum temperature.

Fetches 10 years of daily maximum temperature data from IEM Mesonet CLImate
API and stores in the local database. Provides P(max >= strike) for any
date range — the "no model" baseline that model edge must beat.

Data source: Iowa Environmental Mesonet
  https://mesonet.agron.iastate.edu/cgi-bin/request/daily.py
  Format: CSV with columns station, day, max_tmpf
  Coverage: KBOS data available from ~1940 to present.

Usage:
    refresh_climatology_data("KBOS", years=10)
    prob = climatological_prob("KBOS", strike=45.5, target_month=3, window_days=15)
"""

from __future__ import annotations

import csv
import io
from datetime import date, timedelta
from typing import Optional

import httpx
import structlog

from kalshi_weather_trader.db import db_manager

logger = structlog.get_logger(__name__)

_IEM_DAILY_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/daily.py"
_DEFAULT_TIMEOUT = httpx.Timeout(60.0, connect=15.0)


def refresh_climatology_data(station_id: str = "KBOS", years: int = 10) -> int:
    """Fetch and store historical daily maximum temperatures from IEM.

    Retrieves the past ``years`` years of daily max temperature data and
    upserts into the ``historical_daily_highs`` table.

    Args:
        station_id: ICAO station code (default 'KBOS').
        years:      Number of years of history to fetch (default 10).

    Returns:
        Number of records upserted.

    Raises:
        Nothing — errors are logged; returns 0 on failure.
    """
    end = date.today()
    start = date(end.year - years, end.month, end.day)

    params = {
        "station": station_id,
        "data": "max_tmpf",
        "year1": start.year,
        "month1": start.month,
        "day1": start.day,
        "year2": end.year,
        "month2": end.month,
        "day2": end.day,
        "format": "comma",
        "missing": "M",
        "trace": "trace",
        "tz": "America/New_York",
    }

    try:
        with httpx.Client(timeout=_DEFAULT_TIMEOUT) as client:
            response = client.get(_IEM_DAILY_URL, params=params)
            response.raise_for_status()
            content = response.text
    except Exception as exc:
        logger.error("climatology.fetch.failed", station=station_id, error=str(exc))
        return 0

    count = 0
    reader = csv.DictReader(io.StringIO(content))
    for row in reader:
        try:
            day_str = row.get("day", "").strip()
            high_str = row.get("max_tmpf", "").strip()
            if not day_str or not high_str or high_str in ("M", "T", ""):
                continue
            obs_date = date.fromisoformat(day_str)
            high_f = float(high_str)
            db_manager.upsert_historical_daily_high(station_id, obs_date, high_f, source="IEM")
            count += 1
        except (ValueError, KeyError):
            continue

    logger.info(
        "climatology.fetch.done",
        station=station_id,
        records=count,
        start=str(start),
        end=str(end),
    )
    return count


def climatological_prob(
    station_id: str,
    strike: float,
    target_date: date,
    window_days: int = 15,
) -> Optional[float]:
    """Compute the climatological P(daily max >= strike) for a calendar date.

    Uses a ±window_days window around the target date's day-of-year to
    assemble a sample of historical outcomes, then returns the fraction
    that exceeded the strike.

    Args:
        station_id:  ICAO station code.
        strike:      Temperature threshold (°F). Uses Kalshi semantics: strike
                     boundary is at strike - 0.5°F.
        target_date: The calendar date being priced.
        window_days: Half-width of the day-of-year window (default ±15 days).

    Returns:
        Fraction of historical days where daily max >= strike - 0.5°F,
        or None if fewer than 10 historical records are available.

    Raises:
        Nothing — errors are logged.
    """
    # Build a multi-year date range spanning the same calendar window
    # Query all historical records within ±window_days of the target day-of-year.
    # We scan ±1 year around each historical year's equivalent date.
    try:
        all_records = db_manager.get_historical_daily_highs(
            station_id=station_id,
            start_date=date(target_date.year - 15, 1, 1),
            end_date=target_date - timedelta(days=1),
        )
    except Exception as exc:
        logger.error("climatology.prob.db_failed", error=str(exc))
        return None

    if not all_records:
        return None

    # Filter to same calendar window (day-of-year ± window_days)
    target_doy = target_date.timetuple().tm_yday
    boundary = strike - 0.5  # Kalshi half-integer boundary

    in_window: list[float] = []
    for obs_date, high_f in all_records:
        obs_doy = obs_date.timetuple().tm_yday
        # Circular day-of-year difference (handles year boundary)
        diff = abs(obs_doy - target_doy)
        if diff > 182:
            diff = 365 - diff
        if diff <= window_days:
            in_window.append(high_f)

    if len(in_window) < 10:
        logger.warning(
            "climatology.prob.insufficient_data",
            station=station_id,
            n_records=len(in_window),
            target_date=str(target_date),
            window_days=window_days,
        )
        return None

    prob = sum(1 for h in in_window if h >= boundary) / len(in_window)
    logger.debug(
        "climatology.prob.computed",
        station=station_id,
        strike=strike,
        boundary=boundary,
        n_records=len(in_window),
        prob=round(prob, 4),
    )
    return round(prob, 4)
