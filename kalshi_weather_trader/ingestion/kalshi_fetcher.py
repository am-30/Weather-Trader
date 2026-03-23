"""
Kalshi REST API v2 client with RSA authentication.

Handles:
- RSA PKCS1v15 / SHA-256 request signing from PEM key stored in env var
- Market data polling (tickers, bid/ask, implied probabilities)
- Order submission (used by execution/trader.py)

Authentication spec:
    timestamp_ms = str(int(time.time() * 1000))
    message = (timestamp_ms + METHOD.upper() + path).encode("utf-8")
    sig = private_key.sign(message, PSS(mgf=MGF1(SHA256()), salt_length=PSS.DIGEST_LENGTH), SHA256())
    headers["KALSHI-ACCESS-KEY"]       = key_id
    headers["KALSHI-ACCESS-TIMESTAMP"] = timestamp_ms
    headers["KALSHI-ACCESS-SIGNATURE"] = base64.b64encode(sig).decode()

Note: trader.py imports _get_auth_headers directly to avoid duplicating auth logic.
"""

from __future__ import annotations

import base64
import time
from datetime import date, datetime, timezone
from typing import Any, Optional
from urllib.parse import urlparse

import httpx
import structlog
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from kalshi_weather_trader.config.settings import settings

logger = structlog.get_logger(__name__)

_DEFAULT_TIMEOUT = httpx.Timeout(15.0, connect=5.0)


def _is_retryable(exc: Exception) -> bool:
    """Return True only for transient errors (5xx, network, timeout).

    4xx client errors (including 401 auth failures) are never retried —
    retrying them just hammers the API and masks the real problem.

    Args:
        exc: The exception raised by the HTTP call.

    Returns:
        True if tenacity should retry, False if it should propagate immediately.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return isinstance(exc, (httpx.HTTPError, httpx.TimeoutException))


class KalshiFetcher:
    """RSA-authenticated Kalshi REST API v2 client.

    Instantiate once and reuse. The private key is loaded from the
    ``KALSHI_PRIVATE_KEY`` environment variable at construction time.

    Args:
        None (reads all config from settings singleton).
    """

    def __init__(self) -> None:
        """Initialise the fetcher, loading and validating the PEM private key.

        Args:
            None

        Returns:
            None

        Raises:
            ValueError: If the PEM key cannot be loaded.
        """
        if not settings.kalshi_access_key or not settings.kalshi_private_key:
            raise ValueError(
                "KALSHI_ACCESS_KEY and KALSHI_PRIVATE_KEY must be set before "
                "initialising KalshiFetcher."
            )

        self._base_url = settings.kalshi_api_base_url.rstrip("/")
        self._access_key = settings.kalshi_access_key
        # Kalshi v2 signing requires the full path including version prefix
        # e.g. /trade-api/v2/markets, not just /markets
        self._base_path = urlparse(self._base_url).path.rstrip("/")

        # Normalise \\n sequences that Replit Secrets may inject
        pem = settings.kalshi_private_key.replace("\\n", "\n")
        try:
            self._private_key = serialization.load_pem_private_key(
                pem.encode("utf-8"),
                password=None,
            )
        except Exception as exc:
            logger.error("kalshi.init.key_load_failed", error=str(exc))
            raise ValueError(f"Failed to load Kalshi private key: {exc}") from exc

        logger.info("kalshi.init.done", base_url=self._base_url)

    # ------------------------------------------------------------------
    # Auth helpers
    # ------------------------------------------------------------------

    def _get_auth_headers(self, method: str, path: str) -> dict[str, str]:
        """Build RSA-signed authentication headers for a Kalshi API request.

        This method is intentionally public so trader.py can import and call it
        without duplicating authentication logic.

        Args:
            method: HTTP method string (e.g. 'GET', 'POST').
            path:   API path including leading slash (e.g. '/markets').

        Returns:
            Dict of HTTP headers including KALSHI-ACCESS-KEY,
            KALSHI-ACCESS-TIMESTAMP, KALSHI-ACCESS-SIGNATURE, and
            Content-Type.

        Raises:
            cryptography.exceptions.InvalidSignature: Should never occur with
                a valid private key — logged and re-raised if it does.
        """
        timestamp_ms = str(int(time.time() * 1000))
        message = (timestamp_ms + method.upper() + path).encode("utf-8")
        sig = self._private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self._access_key,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    # ------------------------------------------------------------------
    # Internal HTTP wrappers
    # ------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception(_is_retryable),
        reraise=True,
    )
    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        """Authenticated GET request to the Kalshi API.

        Args:
            path:   API path with leading slash.
            params: Optional query parameters.

        Returns:
            Parsed JSON dict.

        Raises:
            httpx.HTTPStatusError: On 4xx/5xx after retries.
            httpx.TimeoutException: On timeout after retries.
        """
        url = self._base_url + path
        headers = self._get_auth_headers("GET", self._base_path + path)
        with httpx.Client(timeout=_DEFAULT_TIMEOUT) as client:
            response = client.get(url, params=params, headers=headers)
            if response.is_error:
                logger.error(
                    "kalshi.http.error_response",
                    method="GET",
                    url=url,
                    status=response.status_code,
                    body=response.text[:1000],
                )
            response.raise_for_status()
            return response.json()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception(_is_retryable),
        reraise=True,
    )
    def _post(self, path: str, body: dict) -> dict:
        """Authenticated POST request to the Kalshi API.

        Args:
            path: API path with leading slash.
            body: JSON-serialisable request body.

        Returns:
            Parsed JSON dict.

        Raises:
            httpx.HTTPStatusError: On 4xx/5xx after retries.
            httpx.TimeoutException: On timeout after retries.
        """
        url = self._base_url + path
        headers = self._get_auth_headers("POST", self._base_path + path)
        with httpx.Client(timeout=_DEFAULT_TIMEOUT) as client:
            response = client.post(url, json=body, headers=headers)
            if response.is_error:
                logger.error(
                    "kalshi.http.error_response",
                    method="POST",
                    url=url,
                    status=response.status_code,
                    body=response.text[:1000],
                )
            response.raise_for_status()
            return response.json()

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_market(m: dict) -> dict:
        """Normalize Kalshi market price fields to a consistent cents representation.

        The Kalshi API may return prices as ``yes_bid_dollars`` / ``yes_ask_dollars``
        (floats in [0, 1]) or as ``yes_bid`` / ``yes_ask`` (integers in [0, 100]).
        This method ensures downstream code always sees integer cents fields.

        Args:
            m: Raw market dict from the Kalshi API.

        Returns:
            Same dict with ``yes_bid`` and ``yes_ask`` set to integer cents (0–100),
            falling back to ``last_price`` if spread is unavailable.
        """
        m = dict(m)  # shallow copy — do not mutate the original

        # Prefer _dollars fields (new API); fall back to plain int fields
        bid_raw = m.get("yes_bid_dollars")
        ask_raw = m.get("yes_ask_dollars")
        last_raw = m.get("last_price_dollars") or m.get("last_price")

        if bid_raw is not None:
            bid_f = float(bid_raw)
            if bid_f > 1.0:
                # API returned integer cents (0–100) instead of fractional dollars
                logger.warning("kalshi.normalize_market.bid_cents_detected", bid_raw=bid_f)
                bid_f /= 100.0
            m["yes_bid"] = round(bid_f * 100)
        if ask_raw is not None:
            ask_f = float(ask_raw)
            if ask_f > 1.0:
                # API returned integer cents (0–100) instead of fractional dollars
                logger.warning("kalshi.normalize_market.ask_cents_detected", ask_raw=ask_f)
                ask_f /= 100.0
            m["yes_ask"] = round(ask_f * 100)

        # If one side is missing, fall back to last_price for both
        if not m.get("yes_bid") and not m.get("yes_ask") and last_raw is not None:
            last_cents = round(float(last_raw) * 100) if float(last_raw) <= 1.0 else round(float(last_raw))
            m.setdefault("yes_bid", last_cents)
            m.setdefault("yes_ask", last_cents)

        # Populate strike from floor_strike if ticker parsing would be needed later
        if "floor_strike" in m and "yes_strike" not in m:
            m["yes_strike"] = m["floor_strike"]

        return m

    def get_temperature_markets(
        self, target_date: Optional[date] = None
    ) -> list[dict]:
        """Fetch all open KBOS maximum temperature markets for a given date.

        Filters by the Kalshi event ticker prefix for KBOS daily max temp.
        Tries multiple query strategies and normalises price fields before returning
        so that downstream code always sees ``yes_bid`` / ``yes_ask`` in cents.

        Args:
            target_date: Date to search for (used to filter by ticker pattern).
                         Defaults to today's trading target date.

        Returns:
            List of normalised market dicts.

        Raises:
            Nothing — errors are caught and logged.
        """
        from kalshi_weather_trader.config.settings import get_target_date

        if target_date is None:
            target_date = get_target_date()

        # Kalshi ticker pattern: KXHIGHTBOS-26MAR15 (Boston max temp series)
        # Date format is %y%b%d: e.g. 26MAR15 for March 15 2026
        date_str = target_date.strftime("%y%b%d").upper()
        event_ticker = f"KXHIGHTBOS-{date_str}"
        valid_statuses = {"active", "initialized"}

        def _filter_and_normalize(raw: list[dict]) -> list[dict]:
            """Keep only today's markets in a tradeable status and normalise prices."""
            return [
                self._normalize_market(m)
                for m in raw
                if m.get("ticker", "").startswith(event_ticker)
                and m.get("status", "active") in valid_statuses
            ]

        # Strategy 1: series_ticker — the authoritative approach per API docs
        try:
            data = self._get(
                "/markets",
                params={"series_ticker": "KXHIGHTBOS", "limit": 100},
            )
            markets = _filter_and_normalize(data.get("markets", []))
            logger.info(
                "kalshi.get_markets.found",
                event_ticker=event_ticker,
                strategy="series_ticker",
                raw_count=len(data.get("markets", [])),
                filtered_count=len(markets),
            )
            if markets:
                return markets
        except Exception as exc:
            logger.warning("kalshi.get_markets.strategy1_failed", error=str(exc))

        # Strategy 2: event_ticker param directly
        try:
            data = self._get(
                "/markets",
                params={"event_ticker": event_ticker, "limit": 100},
            )
            markets = _filter_and_normalize(data.get("markets", []))
            logger.info(
                "kalshi.get_markets.found",
                event_ticker=event_ticker,
                strategy="event_ticker",
                filtered_count=len(markets),
            )
            if markets:
                return markets
        except Exception as exc:
            logger.warning("kalshi.get_markets.strategy2_failed", error=str(exc))

        # Strategy 3: GET /events/{event_ticker}/markets (nested resource pattern)
        try:
            data = self._get(f"/events/{event_ticker}/markets", params={"limit": 100})
            markets = _filter_and_normalize(data.get("markets", []))
            logger.info(
                "kalshi.get_markets.found",
                event_ticker=event_ticker,
                strategy="events_nested_markets",
                filtered_count=len(markets),
            )
            return markets
        except Exception as exc:
            logger.error(
                "kalshi.get_markets.all_strategies_failed",
                event_ticker=event_ticker,
                error=str(exc),
            )
            return []

    def get_market_by_ticker(self, ticker: str) -> Optional[dict]:
        """Fetch a single market by its full ticker.

        Args:
            ticker: Full Kalshi market ticker string.

        Returns:
            Raw market dict, or None if not found.

        Raises:
            httpx.HTTPError: On network failure after retries.
        """
        path = f"/markets/{ticker}"
        try:
            data = self._get(path)
            return data.get("market", data)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                logger.warning("kalshi.get_market.not_found", ticker=ticker)
                return None
            raise
        except Exception as exc:
            logger.error("kalshi.get_market.failed", ticker=ticker, error=str(exc))
            raise

    @staticmethod
    def extract_strike_from_market(market: dict) -> Optional[float]:
        """Extract the strike temperature from a normalised market dict.

        Prefers the ``floor_strike`` field returned directly by the API.
        Falls back to parsing the ticker string.

        Args:
            market: Normalised market dict (output of ``_normalize_market``).

        Returns:
            Strike temperature as float °F, or None if unavailable.

        Raises:
            Nothing.
        """
        # floor_strike is set by Kalshi directly — most reliable
        fs = market.get("floor_strike")
        if fs is not None:
            try:
                return float(fs)
            except (TypeError, ValueError):
                pass
        # Fall back to ticker parsing
        ticker = market.get("ticker", "")
        return KalshiFetcher.extract_strike_from_ticker_str(ticker)

    def extract_strike_from_ticker(self, ticker: str) -> Optional[float]:
        """Parse the strike temperature from a Kalshi market ticker string.

        Actual KXHIGHTBOS ticker examples (as observed from the API):
          - ``KXHIGHTBOS-26MAR15-T38``   → above-threshold market, strike = 38.0
          - ``KXHIGHTBOS-26MAR15-T45``   → above-threshold market, strike = 45.0
          - ``KXHIGHTBOS-26MAR15-B38.5`` → below-threshold market, strike = 38.5
          - ``KXHIGHTBOS-26MAR15-B44.5`` → below-threshold market, strike = 44.5

        Args:
            ticker: Full Kalshi market ticker string.

        Returns:
            Strike temperature as float °F, or None if parsing fails.

        Raises:
            Nothing — returns None on parse failure.
        """
        return KalshiFetcher.extract_strike_from_ticker_str(ticker)

    @staticmethod
    def extract_strike_from_ticker_str(ticker: str) -> Optional[float]:
        """Static version of strike extraction for use without an instance.

        Args:
            ticker: Full Kalshi market ticker string.

        Returns:
            Strike temperature as float °F, or None if parsing fails.

        Raises:
            Nothing.
        """
        import re

        # Match -T<number> or -B<number> at the end, supporting decimals
        m = re.search(r"-[TB](\d+(?:\.\d+)?)$", ticker)
        if m:
            return float(m.group(1))

        logger.warning("kalshi.extract_strike.failed", ticker=ticker)
        return None

    @staticmethod
    def get_strike_label(market: dict) -> str:
        """Return a human-readable temperature range label for a market.

        Uses ``floor_strike`` and ``cap_strike`` from the API when available,
        falling back to the ticker prefix (T = below threshold, B = range bucket).

        Examples:
          - T38  (floor=None, cap=38)  → "<38°F"
          - B38.5 (floor=38, cap=39)  → "38–39°F"
          - B54  (floor=54, cap=None) → ">54°F"

        Args:
            market: Normalised market dict (output of ``_normalize_market``).

        Returns:
            Human-readable strike label string.

        Raises:
            Nothing.
        """
        import re

        floor_raw = market.get("floor_strike")
        cap_raw = market.get("cap_strike")
        ticker = market.get("ticker", "")

        def _fmt(v: object) -> str:
            f = float(v)  # type: ignore[arg-type]
            return str(int(f)) if f == int(f) else str(f)

        # Use API-provided bounds when present
        if floor_raw is not None and cap_raw is not None:
            return f"{_fmt(floor_raw)}–{_fmt(cap_raw)}°F"
        if floor_raw is not None:
            return f">{_fmt(floor_raw)}°F"
        if cap_raw is not None:
            return f"<{_fmt(cap_raw)}°F"

        # Fall back to ticker prefix
        m = re.search(r"-([TB])(\d+(?:\.\d+)?)$", ticker)
        if m:
            prefix, num = m.group(1), m.group(2)
            val = float(num)
            fmt_val = str(int(val)) if val == int(val) else num
            if prefix == "T":
                return f"<{fmt_val}°F"
            # B = range bucket; estimate cap as floor + 1
            cap_est = int(val) + 1
            return f"{fmt_val}–{cap_est}°F"

        return ticker  # last resort: raw ticker

    def get_best_market_for_date(self, target_date: Optional[date] = None) -> Optional[dict]:
        """Find the single best open temperature market for the target date.

        Returns the market closest to the blended NWP prediction.
        Falls back to returning the first market if no prediction is available.

        Args:
            target_date: Trading date. Defaults to today's target.

        Returns:
            Raw market dict with the best strike, or None if no markets found.

        Raises:
            Nothing — errors are logged.
        """
        markets = self.get_temperature_markets(target_date)
        if not markets:
            logger.warning("kalshi.best_market.no_markets", date=str(target_date))
            return None

        if len(markets) == 1:
            return markets[0]

        # Prefer markets near the blended NWP forecast
        try:
            from kalshi_weather_trader.ingestion.nwp_fetcher import get_blended_forecast

            blended = get_blended_forecast(target_date)
            if blended is not None:
                best = min(
                    markets,
                    key=lambda m: abs(
                        (self.extract_strike_from_ticker(m.get("ticker", "")) or 0) - blended
                    ),
                )
                return best
        except Exception:
            pass

        return markets[0]

    def get_positions(self, ticker: Optional[str] = None) -> list[dict]:
        """Fetch current portfolio positions from Kalshi.

        Args:
            ticker: Optional market ticker to filter positions. If None, returns all.

        Returns:
            List of position dicts with keys: ticker, position, market_exposure, etc.
            Returns empty list on error.

        Raises:
            Nothing — errors are caught and logged.
        """
        log = structlog.get_logger()
        try:
            params: dict[str, Any] = {"limit": 100}
            if ticker:
                params["ticker"] = ticker
            data = self._get("/portfolio/positions", params=params)
            positions = data.get("market_positions", [])
            log.info("kalshi.positions.fetched", count=len(positions))
            return positions
        except Exception as e:
            log.warning("kalshi.positions.fetch_failed", error=str(e))
            return []

    def get_balance(self) -> float:
        """Fetch the current account balance in USD.

        Args:
            None

        Returns:
            Account balance in USD as a float.

        Raises:
            httpx.HTTPError: On API failure.
        """
        try:
            data = self._get("/portfolio/balance")
            raw_balance = data.get("balance", 0)
            return round(int(raw_balance) / 100.0, 2)
        except Exception as exc:
            logger.error("kalshi.get_balance.failed", error=str(exc))
            raise

    def submit_order(
        self,
        ticker: str,
        side: str,
        action: str,
        count: int,
        yes_price_cents: int,
    ) -> dict:
        """Submit a limit order to Kalshi.

        Args:
            ticker:          Market ticker string.
            side:            'yes' or 'no'.
            action:          'buy' or 'sell'.
            count:           Number of contracts.
            yes_price_cents: Limit price in cents (1–99).

        Returns:
            Parsed order response dict from the Kalshi API.

        Raises:
            httpx.HTTPStatusError: On API rejection (e.g. insufficient funds).
            httpx.TimeoutException: On timeout.
        """
        body = {
            "ticker": ticker,
            "side": side.lower(),
            "action": action.lower(),
            "count": count,
            "type": "limit",
            "yes_price": yes_price_cents,
        }
        try:
            data = self._post("/portfolio/orders", body)
            logger.info(
                "kalshi.submit_order.done",
                ticker=ticker,
                side=side,
                count=count,
                price_cents=yes_price_cents,
            )
            return data
        except httpx.HTTPStatusError as exc:
            logger.error(
                "kalshi.submit_order.http_error",
                status=exc.response.status_code,
                body=exc.response.text[:500],
            )
            raise
        except Exception as exc:
            logger.error("kalshi.submit_order.failed", error=str(exc))
            raise


# Module-level singleton — constructed lazily to avoid startup failures
# when env vars aren't loaded yet.  Use get_kalshi_fetcher() instead of
# importing this directly.
_fetcher: Optional[KalshiFetcher] = None


def get_kalshi_fetcher() -> KalshiFetcher:
    """Return the module-level KalshiFetcher singleton, constructing it if needed.

    Args:
        None

    Returns:
        Shared ``KalshiFetcher`` instance.

    Raises:
        ValueError: If the Kalshi private key is invalid.
    """
    global _fetcher
    if _fetcher is None:
        _fetcher = KalshiFetcher()
    return _fetcher
