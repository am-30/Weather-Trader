"""
Numerical Weather Prediction (NWP) forecast fetcher via Open-Meteo API.

Fetches hourly temperature forecasts for the KBOS grid point from three models:
- HRRR  (High-Resolution Rapid Refresh — NOAA, ~3 km, US-only)
- GFS   (Global Forecast System — NOAA, ~13 km, global)
- ECMWF (IFS 0.25° — European Centre, global)

The Open-Meteo free API provides all three in a single call per model.
Forecasts are stored via db_manager and a blended prediction is computed
using model weights from system_state.
"""

from __future__ import annotations

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
_DEFAULT_TIMEOUT = httpx.Timeout(30.0, connect=10.0)

# KBOS coordinates
_KBOS_LAT = 42.3606
_KBOS_LON = -71.0097

# Open-Meteo model identifiers.
# NOTE: gfs_seamless was previously used for GFS but it is a near-term blend
# of GFS + HRRR, producing identical results to ncep_hrrr_conus for same-day
# forecasts.  gfs_global is the pure GFS model (~25 km, 4x daily) and gives
# genuinely independent data.
_MODEL_MAP = {
    "HRRR": "ncep_hrrr_conus",
    "GFS": "gfs_global",
    "ECMWF": "ecmwf_ifs025",
}

# Fallback model identifiers to try if the primary name fails
_MODEL_FALLBACKS: dict[str, list[str]] = {
    "GFS": ["gfs_seamless"],
    "HRRR": [],
    "ECMWF": ["ecmwf_ifs04"],
}

_OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(_RETRY_EXCEPTIONS),
    reraise=True,
)
def _get_open_meteo(params: dict) -> dict:
    """Execute a GET request to the Open-Meteo API with retry logic.

    Args:
        params: Query parameters dict.

    Returns:
        Parsed JSON response dict.

    Raises:
        httpx.HTTPStatusError: On 4xx/5xx after retries.
        httpx.TimeoutException: On timeout after retries.
    """
    with httpx.Client(timeout=_DEFAULT_TIMEOUT) as client:
        response = client.get(_OPEN_METEO_URL, params=params)
        response.raise_for_status()
        return response.json()


# ---------------------------------------------------------------------------
# Per-model fetch
# ---------------------------------------------------------------------------


def _fetch_model(model_name: str, target_date: date) -> Optional[NWPForecastDocument]:
    """Fetch an hourly forecast for one NWP model for the target date.

    Fetches 26 hours of data: the 24 ET hours of the target date plus the first
    2 hours of the next calendar day (midnight and 1 AM ET), which correspond to
    the end of the NWS observation window during EDT (UTC-4).  The MC engine
    needs these extra values to correctly price the final ~2 hours of the day.

    Args:
        model_name:  One of 'HRRR', 'GFS', 'ECMWF'.
        target_date: The calendar date to retrieve forecasts for.

    Returns:
        ``NWPForecastDocument`` with a 26-element hourly_temps array, or None
        if the API call fails or no data covers the target date.

    Raises:
        Nothing — exceptions are caught and logged.
    """
    primary = _MODEL_MAP.get(model_name)
    if not primary:
        logger.error("nwp.fetch.unknown_model", model=model_name)
        return None

    candidates = [primary] + _MODEL_FALLBACKS.get(model_name, [])
    date_str = target_date.isoformat()
    next_date_str = (target_date + timedelta(days=1)).isoformat()
    hourly_temps: list[float] = []

    for om_model in candidates:
        params = {
            "latitude": _KBOS_LAT,
            "longitude": _KBOS_LON,
            "hourly": "temperature_2m",
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "timezone": "America/New_York",
            "models": om_model,
            "start_date": date_str,
            "end_date": next_date_str,
        }
        try:
            data = _get_open_meteo(params)
        except Exception as exc:
            logger.warning(
                "nwp.fetch.candidate_failed",
                model=model_name,
                om_model=om_model,
                error=str(exc),
            )
            continue

        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        temps_raw = hourly.get("temperature_2m", [])

        if not times or not temps_raw:
            logger.warning(
                "nwp.fetch.empty_response",
                model=model_name,
                om_model=om_model,
            )
            continue

        candidate_temps: list[float] = []
        for t, temp in zip(times, temps_raw):
            if temp is None:
                continue
            if t.startswith(date_str):
                candidate_temps.append(round(float(temp), 1))
            elif t.startswith(next_date_str) and len(candidate_temps) >= 24:
                hour = int(t[11:13])  # extract HH from "YYYY-MM-DDTHH:MM"
                if hour <= 1:
                    candidate_temps.append(round(float(temp), 1))

        if not candidate_temps:
            logger.warning(
                "nwp.fetch.no_temps_for_date",
                model=model_name,
                om_model=om_model,
                date=date_str,
            )
            continue

        if len(candidate_temps) < 26:
            logger.warning(
                "nwp.fetch.partial_data",
                model=model_name,
                om_model=om_model,
                hours=len(candidate_temps),
            )

        hourly_temps = candidate_temps
        logger.info(
            "nwp.fetch.candidate_ok",
            model=model_name,
            om_model=om_model,
            hours=len(hourly_temps),
        )
        break

    if not hourly_temps:
        logger.error(
            "nwp.fetch.all_candidates_failed",
            model=model_name,
            candidates=candidates,
        )
        return None

    predicted_high = round(max(hourly_temps), 1)

    doc = NWPForecastDocument(
        target_date=target_date,
        model_name=model_name,
        fetched_at_utc=datetime.now(timezone.utc),
        hourly_temps=hourly_temps,
        predicted_daily_high=predicted_high,
    )

    logger.info(
        "nwp.fetch.success",
        model=model_name,
        date=date_str,
        predicted_high=predicted_high,
        hours=len(hourly_temps),
    )
    return doc


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def fetch_all_models(target_date: Optional[date] = None) -> dict[str, NWPForecastDocument]:
    """Fetch and persist NWP forecasts from all three models.

    Args:
        target_date: The date to fetch for. Defaults to today's trading date.

    Returns:
        Dict mapping model_name → ``NWPForecastDocument`` for models that
        succeeded. Missing models are excluded (not set to None).

    Raises:
        Nothing — per-model failures are logged; partial results are returned.
    """
    if target_date is None:
        target_date = get_target_date()

    results: dict[str, NWPForecastDocument] = {}
    for model_name in _MODEL_MAP:
        doc = _fetch_model(model_name, target_date)
        if doc is not None:
            results[model_name] = doc
            try:
                db_manager.upsert_nwp_forecast(doc)
            except Exception as exc:
                logger.error("nwp.persist.failed", model=model_name, error=str(exc))

    logger.info(
        "nwp.fetch_all.done",
        models_ok=list(results.keys()),
        date=str(target_date),
    )
    return results


def get_blended_forecast(target_date: Optional[date] = None) -> Optional[float]:
    """Compute a weight-blended daily high forecast from all available models.

    Reads model weights from ``system_state`` and applies them to the latest
    ``nwp_forecasts`` for the target date.  If a model is missing, its weight
    is redistributed proportionally among the remaining models.

    Args:
        target_date: Trading date. Defaults to today's trading date.

    Returns:
        Blended predicted high temperature in °F, or None if no forecasts exist.

    Raises:
        Nothing — errors are logged.
    """
    if target_date is None:
        target_date = get_target_date()

    try:
        forecasts = db_manager.get_latest_nwp_forecasts(target_date)
    except Exception as exc:
        logger.error("nwp.blended.db_read_failed", error=str(exc))
        return None

    if not forecasts:
        logger.warning("nwp.blended.no_forecasts", date=str(target_date))
        return None

    # Read model weights from system_state
    try:
        state = db_manager.get_system_state(target_date)
        weights: dict[str, float] = (
            state.model_weights if state else {"HRRR": 0.5, "GFS": 0.3, "ECMWF": 0.2}
        )
    except Exception as exc:
        logger.warning("nwp.blended.weights_read_failed", error=str(exc))
        weights = {"HRRR": 0.5, "GFS": 0.3, "ECMWF": 0.2}

    # Only use weights for models that have forecasts
    available_weights = {m: weights.get(m, 0.0) for m in forecasts}
    total_weight = sum(available_weights.values())

    if total_weight == 0.0:
        # Fallback to equal weights
        available_weights = {m: 1.0 for m in forecasts}
        total_weight = float(len(forecasts))

    blended = sum(
        (w / total_weight) * forecasts[m].predicted_daily_high
        for m, w in available_weights.items()
    )
    blended = round(blended, 1)

    logger.info(
        "nwp.blended.done",
        blended=blended,
        models=list(forecasts.keys()),
        date=str(target_date),
    )
    return blended


def get_nwp_curve(target_date: Optional[date] = None) -> list[float]:
    """Return the blended hourly temperature curve for the target date.

    Computes a weight-blended hourly temperature array across all available
    models.  Returns a list of up to 24 Fahrenheit values.

    Args:
        target_date: Trading date. Defaults to today's trading date.

    Returns:
        List of blended hourly temperatures (°F), up to 26 values.  Empty list if no forecasts.

    Raises:
        Nothing — errors are logged.
    """
    if target_date is None:
        target_date = get_target_date()

    try:
        forecasts = db_manager.get_latest_nwp_forecasts(target_date)
    except Exception as exc:
        logger.error("nwp.curve.db_read_failed", error=str(exc))
        return []

    if not forecasts:
        return []

    try:
        state = db_manager.get_system_state(target_date)
        weights: dict[str, float] = (
            state.model_weights if state else {"HRRR": 0.5, "GFS": 0.3, "ECMWF": 0.2}
        )
    except Exception:
        weights = {"HRRR": 0.5, "GFS": 0.3, "ECMWF": 0.2}

    # Determine curve length (longest model array so no data is hidden)
    n_hours = max(len(f.hourly_temps) for f in forecasts.values())
    if n_hours == 0:
        return []

    available_weights = {m: weights.get(m, 0.0) for m in forecasts}

    curve: list[float] = []
    for hour in range(n_hours):
        # Only include models that have data for this hour
        hour_contributions = {
            m: forecasts[m].hourly_temps[hour]
            for m in available_weights
            if hour < len(forecasts[m].hourly_temps)
        }
        if not hour_contributions:
            break
        hour_weights = {m: available_weights[m] for m in hour_contributions}
        hour_total_weight = sum(hour_weights.values())
        if hour_total_weight == 0.0:
            hour_weights = {m: 1.0 for m in hour_contributions}
            hour_total_weight = float(len(hour_contributions))
        blended_hour = sum(
            (w / hour_total_weight) * hour_contributions[m]
            for m, w in hour_weights.items()
        )
        curve.append(round(blended_hour, 1))

    return curve
