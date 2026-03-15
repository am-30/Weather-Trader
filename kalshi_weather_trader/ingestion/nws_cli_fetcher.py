"""
NWS CLI (Climate Report) fetcher for Boston (BOS).

Fetches the official NWS daily Climate Summary product published at:
  https://forecast.weather.gov/product.php?site=BOX&product=CLI&issuedby=BOS

The CLI product is typically posted ~9:30 AM ET the following morning and
contains the OFFICIAL daily maximum temperature used for Kalshi settlement.

Public API:
    fetch_official_daily_high(target_date) -> Optional[float]
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Optional

import httpx
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = structlog.get_logger(__name__)

_BASE_URL = "https://forecast.weather.gov/product.php"
_PARAMS_BASE = {"site": "BOX", "product": "CLI", "issuedby": "BOS", "format": "txt"}
_RETRY_EXCEPTIONS = (httpx.HTTPError, httpx.TimeoutException)
_DEFAULT_TIMEOUT = httpx.Timeout(20.0, connect=10.0)
_NWS_HEADERS = {
    "User-Agent": "KalshiWeatherTrader/1.0 (automated trading system - contact@example.com)",
    "Accept": "text/html,text/plain",
}
_MAX_VERSIONS = 5


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(_RETRY_EXCEPTIONS),
    reraise=True,
)
def _fetch_cli_text(version: int) -> str:
    """Fetch the raw CLI product text for a given version number.

    Args:
        version: NWS product version (1 = most recent, higher = older).

    Returns:
        Raw page text as a string.

    Raises:
        httpx.HTTPStatusError: On non-2xx response.
        httpx.HTTPError: On network-level failures (retriable).
    """
    params = {**_PARAMS_BASE, "version": str(version)}
    with httpx.Client(timeout=_DEFAULT_TIMEOUT, headers=_NWS_HEADERS, follow_redirects=True) as client:
        resp = client.get(_BASE_URL, params=params)
        resp.raise_for_status()
        return resp.text


def _parse_report_date(text: str) -> Optional[date]:
    """Extract the climate summary date from the CLI product text.

    Matches the header line:
        CLIMATE SUMMARY FOR SUNDAY MARCH 15 2026
        CLIMATE SUMMARY FOR MARCH 15 2026

    Args:
        text: Raw CLI product text.

    Returns:
        Parsed date, or None if the header is not found or unparseable.

    Raises:
        Nothing.
    """
    # Match optional day-of-week, then month day year
    match = re.search(
        r'CLIMATE SUMMARY FOR (?:\w+ )?(\w+ \d{1,2} \d{4})',
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1).upper(), "%B %d %Y").date()
    except ValueError:
        return None


def _parse_maximum_temp(text: str) -> Optional[float]:
    """Extract the MAXIMUM TODAY temperature from the CLI product text.

    The relevant section looks like:
        TEMPERATURE (F)
                       TODAY    NORMAL    RECORD     YEAR
        MAXIMUM         45        44        72       1945
        MINIMUM         32        29        -2       1957

    Rejects 'M' (missing) and any non-numeric token in the TODAY column.

    Args:
        text: Raw CLI product text (date already validated by caller).

    Returns:
        Maximum temperature as float, or None if missing/non-numeric.

    Raises:
        Nothing.
    """
    match = re.search(r'MAXIMUM\s+([\dM.]+)', text, re.IGNORECASE)
    if not match:
        return None
    raw = match.group(1).strip()
    if raw.upper() == 'M' or not re.fullmatch(r'[\d.]+', raw):
        logger.warning("nws_cli.maximum_missing_or_nonnumeric", raw_value=raw)
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def fetch_official_daily_high(target_date: date) -> Optional[float]:
    """Fetch the NWS official daily maximum temperature for *target_date*.

    Tries CLI product versions 1 through _MAX_VERSIONS (most-recent first),
    validates each version's report date against *target_date* before
    accepting the value.  Returns None if:
    - No version matches *target_date* (report not yet posted or date mismatch)
    - MAXIMUM field shows 'M' (missing — intraday partial report)
    - Any network error that exhausts retries

    Args:
        target_date: The calendar date for which to fetch the official high.

    Returns:
        Official daily maximum temperature in °F, or None.

    Raises:
        Nothing — all errors are caught and logged.
    """
    log = logger.bind(target_date=str(target_date))
    log.info("nws_cli.fetch.start", max_versions=_MAX_VERSIONS)

    for version in range(1, _MAX_VERSIONS + 1):
        try:
            text = _fetch_cli_text(version)
        except Exception as exc:
            log.warning("nws_cli.fetch.http_error", version=version, error=str(exc))
            return None  # Network failure — don't continue cycling versions

        report_date = _parse_report_date(text)
        log.info(
            "nws_cli.version_checked",
            version=version,
            report_date=str(report_date),
            requested_date=str(target_date),
        )

        if report_date is None:
            log.warning("nws_cli.date_parse_failed", version=version)
            continue

        if report_date == target_date:
            temp = _parse_maximum_temp(text)
            if temp is None:
                log.warning(
                    "nws_cli.maximum_not_available",
                    version=version,
                    report_date=str(report_date),
                )
                return None  # Report is for the right date but reading isn't final
            log.info(
                "nws_cli.fetch.success",
                version=version,
                report_date=str(report_date),
                official_high=temp,
            )
            return temp

        if report_date < target_date:
            # Versions are newest-first; if we've gone past the target there's
            # nothing to find in older versions.
            log.info(
                "nws_cli.fetch.date_passed",
                version=version,
                report_date=str(report_date),
                reason="report is older than target; stopping version scan",
            )
            return None

        # report_date > target_date: this version is for a future date (unlikely
        # but guard against it); continue to the next version.
        log.debug(
            "nws_cli.fetch.future_report",
            version=version,
            report_date=str(report_date),
        )

    log.info("nws_cli.fetch.not_found", reason="no matching version in range")
    return None
