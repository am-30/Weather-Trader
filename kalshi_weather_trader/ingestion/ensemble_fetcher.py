"""
NWP ensemble forecast fetcher via Open-Meteo Ensemble API.

Fetches ensemble temperature forecasts for KBOS from:
- GFS 0.25° (31 members): model = 'gfs025'
- ECMWF IFS 0.25° (51 members): model = 'ecmwf_ifs025'

The ensemble spread (std of per-member daily highs) is stored alongside the
ensemble mean curve in nwp_forecasts with model names 'GFS_ENS' / 'ECMWF_ENS'.

When ensemble spread exceeds the configured threshold, the MC simulation inflates
sigma to reflect genuine atmospheric uncertainty (frontal days, model divergence).
"""

from __future__ import annotations

import statistics
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import httpx
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from kalshi_weather_trader.config.settings import get_target_date, settings
from kalshi_weather_trader.db import db_manager
from kalshi_weather_trader.db.schemas import NWPForecastDocument

logger = structlog.get_logger(__name__)

_RETRY_EXCEPTIONS = (httpx.HTTPError, httpx.TimeoutException)
_DEFAULT_TIMEOUT = httpx.Timeout(60.0, connect=15.0)

_KBOS_LAT = 42.3606
_KBOS_LON = -71.0097

_ENSEMBLE_API_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"

# Mapping from our internal model name to Open-Meteo ensemble model identifier
_ENSEMBLE_MODEL_MAP = {
    "GFS_ENS": "gfs025",
    "ECMWF_ENS": "ecmwf_ifs025",
}


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(_RETRY_EXCEPTIONS),
    reraise=True,
)
def _get_ensemble_api(params: dict) -> dict:
    """Execute a GET request to the Open-Meteo Ensemble API with retry logic.

    Args:
        params: Query parameters dict.

    Returns:
        Parsed JSON response dict.

    Raises:
        httpx.HTTPStatusError: On 4xx/5xx after retries.
        httpx.TimeoutException: On timeout after retries.
    """
    with httpx.Client(timeout=_DEFAULT_TIMEOUT) as client:
        response = client.get(_ENSEMBLE_API_URL, params=params)
        response.raise_for_status()
        return response.json()


def _fetch_ensemble_model(model_name: str, target_date: date) -> Optional[NWPForecastDocument]:
    """Fetch ensemble forecast for one model and compute spread of daily highs.

    For each ensemble member, extracts the hourly temperature curve for the
    target date and computes the predicted daily maximum. The spread (std) of
    these per-member daily maxes quantifies atmospheric uncertainty on that day.

    The ensemble mean curve (averaged across members per hour) is stored as
    hourly_temps so the existing NWP curve blending infrastructure can use it
    if desired. predicted_daily_high is the ensemble mean daily high.

    Args:
        model_name:  'GFS_ENS' or 'ECMWF_ENS'.
        target_date: The calendar date to retrieve forecasts for.

    Returns:
        NWPForecastDocument with ensemble_highs and ensemble_spread populated,
        or None if the API call fails.

    Raises:
        Nothing — exceptions are caught and logged.
    """
    om_model = _ENSEMBLE_MODEL_MAP.get(model_name)
    if not om_model:
        logger.error("ensemble.fetch.unknown_model", model=model_name)
        return None

    date_str = target_date.isoformat()
    next_date_str = (target_date + timedelta(days=1)).isoformat()

    try:
        params = {
            "latitude": _KBOS_LAT,
            "longitude": _KBOS_LON,
            "hourly": "temperature_2m",
            "temperature_unit": "fahrenheit",
            "timezone": "America/New_York",
            "models": om_model,
            "start_date": date_str,
            "end_date": next_date_str,
        }
        data = _get_ensemble_api(params)
    except Exception as exc:
        logger.warning("ensemble.fetch.api_failed", model=model_name, error=str(exc))
        return None

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        logger.warning("ensemble.fetch.empty_response", model=model_name)
        return None

    # Collect member keys: keys starting with 'temperature_2m_member'
    member_keys = sorted(k for k in hourly if k.startswith("temperature_2m_member"))
    if not member_keys:
        logger.warning("ensemble.fetch.no_member_keys", model=model_name, keys=list(hourly.keys())[:5])
        return None

    # For each member, extract today's hourly temps and compute daily max
    member_highs: list[float] = []
    member_curves: list[list[float]] = []
    for key in member_keys:
        raw = hourly.get(key, [])
        member_temps_today: list[float] = []
        for t, val in zip(times, raw):
            if val is None:
                continue
            if t.startswith(date_str):
                member_temps_today.append(float(val))
        if member_temps_today:
            member_curves.append(member_temps_today)
            member_highs.append(max(member_temps_today))

    if not member_highs:
        logger.warning("ensemble.fetch.no_member_data", model=model_name, date=date_str)
        return None

    # Ensemble mean curve (per-hour average across members)
    n_hours = max(len(c) for c in member_curves)
    mean_curve: list[float] = []
    for h in range(n_hours):
        vals = [c[h] for c in member_curves if h < len(c)]
        if vals:
            mean_curve.append(round(sum(vals) / len(vals), 1))

    mean_high = round(sum(member_highs) / len(member_highs), 1)
    spread = round(statistics.stdev(member_highs), 2) if len(member_highs) > 1 else 0.0

    doc = NWPForecastDocument(
        target_date=target_date,
        model_name=model_name,
        fetched_at_utc=datetime.now(timezone.utc),
        hourly_temps=mean_curve if mean_curve else [mean_high],
        predicted_daily_high=mean_high,
        ensemble_highs=member_highs,
        ensemble_spread=spread,
    )

    logger.info(
        "ensemble.fetch.success",
        model=model_name,
        n_members=len(member_highs),
        mean_high=mean_high,
        spread=spread,
        date=date_str,
    )
    return doc


def fetch_all_ensemble_models(target_date: Optional[date] = None) -> dict[str, NWPForecastDocument]:
    """Fetch and persist ensemble forecasts from GFS and ECMWF.

    Args:
        target_date: The date to fetch for. Defaults to today's trading date.

    Returns:
        Dict mapping model_name -> NWPForecastDocument for models that succeeded.

    Raises:
        Nothing — per-model failures are logged; partial results are returned.
    """
    if target_date is None:
        target_date = get_target_date()

    results: dict[str, NWPForecastDocument] = {}
    for model_name in _ENSEMBLE_MODEL_MAP:
        doc = _fetch_ensemble_model(model_name, target_date)
        if doc is not None:
            results[model_name] = doc
            try:
                db_manager.upsert_nwp_forecast(doc)
            except Exception as exc:
                logger.error("ensemble.persist.failed", model=model_name, error=str(exc))

    spread_by_model = {m: d.ensemble_spread for m, d in results.items()}
    logger.info(
        "ensemble.fetch_all.done",
        models_ok=list(results.keys()),
        spreads=spread_by_model,
        date=str(target_date),
    )
    return results
