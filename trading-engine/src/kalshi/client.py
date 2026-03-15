"""
Authenticated Kalshi REST API v2 client.

Wraps all HTTP calls with tenacity exponential-backoff retry logic (3 attempts).
All methods return typed Pydantic models. Never exposes raw dicts to callers.

API docs: https://trading-api.kalshi.com/trade-api/v2/openapi.json
"""

from __future__ import annotations

from datetime import datetime, timezone
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
from src.models.market import (
    KalshiMarket,
    MarketStatus,
    OrderBook,
    OrderRequest,
    OrderResponse,
    OrderSide,
    OrderAction,
)

logger = structlog.get_logger(__name__)

_RETRY_EXCEPTIONS = (httpx.HTTPError, httpx.TimeoutException)
_DEFAULT_TIMEOUT = httpx.Timeout(15.0, connect=5.0)


class KalshiClient:
    """Authenticated Kalshi REST API v2 client.

    Instantiate once and reuse. The API key is read from ``settings`` and
    passed as a header on every request.

    Args:
        api_key: Kalshi API key. Defaults to ``settings.kalshi_api_key``.
        base_url: Base URL. Defaults to ``settings.kalshi_api_base_url``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._api_key = api_key or settings.kalshi_api_key
        self._base_url = (base_url or settings.kalshi_api_base_url).rstrip("/")
        self._headers = {
            "Authorization": f"Token {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(_RETRY_EXCEPTIONS),
        before_sleep=before_sleep_log(logger, "warning"),
        reraise=True,
    )
    def _get(self, path: str, params: dict | None = None) -> dict:
        """Execute an authenticated GET request.

        Args:
            path:   API path (e.g. ``"/markets"``). Leading slash required.
            params: Optional query parameters.

        Returns:
            Parsed JSON response as a dict.

        Raises:
            httpx.HTTPStatusError: On 4xx/5xx after retries.
            httpx.TimeoutException: On timeout after retries.
        """
        url = self._base_url + path
        with httpx.Client(timeout=_DEFAULT_TIMEOUT) as client:
            response = client.get(url, params=params, headers=self._headers)
            response.raise_for_status()
            return response.json()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(_RETRY_EXCEPTIONS),
        before_sleep=before_sleep_log(logger, "warning"),
        reraise=True,
    )
    def _post(self, path: str, body: dict) -> dict:
        """Execute an authenticated POST request.

        Args:
            path: API path. Leading slash required.
            body: JSON-serialisable request body.

        Returns:
            Parsed JSON response as a dict.

        Raises:
            httpx.HTTPStatusError: On 4xx/5xx after retries.
            httpx.TimeoutException: On timeout after retries.
        """
        url = self._base_url + path
        with httpx.Client(timeout=_DEFAULT_TIMEOUT) as client:
            response = client.post(url, json=body, headers=self._headers)
            response.raise_for_status()
            return response.json()

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def get_markets(
        self,
        event_ticker: str | None = None,
        status: str = "open",
        limit: int = 100,
    ) -> list[KalshiMarket]:
        """Fetch a list of markets, optionally filtered by event ticker and status.

        Args:
            event_ticker: Parent event ticker prefix to filter by (e.g. ``"KXMAXTEMP"``).
            status:       Market status filter. Defaults to ``"open"``.
            limit:        Maximum number of results. Defaults to 100.

        Returns:
            List of ``KalshiMarket`` objects.

        Raises:
            httpx.HTTPError: On API failure after retries.
        """
        params: dict = {"limit": limit, "status": status}
        if event_ticker:
            params["event_ticker"] = event_ticker

        logger.info("kalshi.get_markets", params=params)
        try:
            data = self._get("/markets", params=params)
        except Exception as exc:
            logger.error("kalshi.get_markets.failed", error=str(exc), exc_info=True)
            raise

        markets = []
        for raw in data.get("markets", []):
            try:
                market = KalshiMarket(
                    ticker=raw.get("ticker", ""),
                    event_ticker=raw.get("event_ticker", ""),
                    title=raw.get("title", ""),
                    status=MarketStatus(raw.get("status", "unknown")),
                    yes_bid=int(raw.get("yes_bid", 0)),
                    yes_ask=int(raw.get("yes_ask", 0)),
                    no_bid=int(raw.get("no_bid", 0)),
                    no_ask=int(raw.get("no_ask", 0)),
                    last_price=int(raw.get("last_price", 0)),
                    volume=int(raw.get("volume", 0)),
                    open_interest=int(raw.get("open_interest", 0)),
                    close_time=_parse_dt(raw.get("close_time")),
                    expiration_time=_parse_dt(raw.get("expiration_time")),
                    result=raw.get("result"),
                )
                markets.append(market)
            except Exception as exc:
                logger.warning("kalshi.market.parse_error", ticker=raw.get("ticker"), error=str(exc))

        logger.info("kalshi.get_markets.done", count=len(markets))
        return markets

    def get_market(self, ticker: str) -> KalshiMarket:
        """Fetch a single market by its ticker.

        Args:
            ticker: Unique market ticker.

        Returns:
            ``KalshiMarket`` for the requested ticker.

        Raises:
            httpx.HTTPError: On API failure after retries.
            KeyError: If the API response does not contain market data.
        """
        logger.info("kalshi.get_market", ticker=ticker)
        try:
            data = self._get(f"/markets/{ticker}")
        except Exception as exc:
            logger.error("kalshi.get_market.failed", ticker=ticker, error=str(exc))
            raise

        raw = data.get("market", data)
        return KalshiMarket(
            ticker=raw.get("ticker", ticker),
            event_ticker=raw.get("event_ticker", ""),
            title=raw.get("title", ""),
            status=MarketStatus(raw.get("status", "unknown")),
            yes_bid=int(raw.get("yes_bid", 0)),
            yes_ask=int(raw.get("yes_ask", 0)),
            no_bid=int(raw.get("no_bid", 0)),
            no_ask=int(raw.get("no_ask", 0)),
            last_price=int(raw.get("last_price", 0)),
            volume=int(raw.get("volume", 0)),
            open_interest=int(raw.get("open_interest", 0)),
            close_time=_parse_dt(raw.get("close_time")),
            expiration_time=_parse_dt(raw.get("expiration_time")),
            result=raw.get("result"),
        )

    def get_order_book(self, ticker: str) -> OrderBook:
        """Fetch the order book for a specific market.

        Args:
            ticker: Unique market ticker.

        Returns:
            ``OrderBook`` with YES bid and ask levels.

        Raises:
            httpx.HTTPError: On API failure after retries.
        """
        logger.info("kalshi.get_order_book", ticker=ticker)
        try:
            data = self._get(f"/markets/{ticker}/orderbook")
        except Exception as exc:
            logger.error("kalshi.get_order_book.failed", ticker=ticker, error=str(exc))
            raise

        book_data = data.get("orderbook", {})
        yes_bids = [(int(p), int(q)) for p, q in book_data.get("yes", [])]
        yes_asks = [(int(p), int(q)) for p, q in book_data.get("no", [])]

        return OrderBook(
            ticker=ticker,
            yes_bids=sorted(yes_bids, key=lambda x: -x[0]),
            yes_asks=sorted(yes_asks, key=lambda x: x[0]),
        )

    # ------------------------------------------------------------------
    # Portfolio / account
    # ------------------------------------------------------------------

    def get_balance(self) -> float:
        """Fetch the current account balance in USD.

        Returns:
            Account balance in USD (float, 2 d.p.).

        Raises:
            httpx.HTTPError: On API failure after retries.
        """
        logger.info("kalshi.get_balance")
        try:
            data = self._get("/portfolio/balance")
        except Exception as exc:
            logger.error("kalshi.get_balance.failed", error=str(exc))
            raise

        balance_cents = data.get("balance", 0)
        return round(balance_cents / 100.0, 2)

    def get_positions(self) -> list[dict]:
        """Fetch all open positions from the Kalshi account.

        Returns:
            Raw list of position dicts from the API.

        Raises:
            httpx.HTTPError: On API failure after retries.
        """
        logger.info("kalshi.get_positions")
        try:
            data = self._get("/portfolio/positions")
        except Exception as exc:
            logger.error("kalshi.get_positions.failed", error=str(exc))
            raise
        return data.get("market_positions", [])

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    def submit_order(self, order: OrderRequest) -> OrderResponse:
        """Submit a limit or market order to Kalshi.

        Args:
            order: ``OrderRequest`` model with all required fields.

        Returns:
            ``OrderResponse`` with the Kalshi-assigned order ID and status.

        Raises:
            httpx.HTTPStatusError: On 4xx (e.g. insufficient balance) / 5xx.
            httpx.TimeoutException: On timeout after retries.
        """
        body = {
            "action": order.action.value,
            "side": order.side.value,
            "ticker": order.ticker,
            "count": order.count,
            "type": order.type,
        }
        if order.yes_price is not None:
            body["yes_price"] = order.yes_price
        if order.client_order_id:
            body["client_order_id"] = order.client_order_id

        logger.info(
            "kalshi.submit_order",
            ticker=order.ticker,
            action=order.action.value,
            side=order.side.value,
            count=order.count,
            yes_price=order.yes_price,
        )

        try:
            data = self._post("/portfolio/orders", body)
        except httpx.HTTPStatusError as exc:
            logger.error(
                "kalshi.submit_order.http_error",
                status_code=exc.response.status_code,
                response_text=exc.response.text,
            )
            raise
        except Exception as exc:
            logger.error("kalshi.submit_order.failed", error=str(exc))
            raise

        raw = data.get("order", data)
        return OrderResponse(
            order_id=raw["order_id"],
            status=raw.get("status", "unknown"),
            created_time=_parse_dt(raw.get("created_time")) or datetime.now(timezone.utc),
            ticker=raw.get("ticker", order.ticker),
            side=OrderSide(raw.get("side", order.side.value)),
            action=OrderAction(raw.get("action", order.action.value)),
            count=int(raw.get("count", order.count)),
            yes_price=raw.get("yes_price"),
        )

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order by its Kalshi order ID.

        Args:
            order_id: Kalshi-assigned order identifier.

        Returns:
            True if the cancellation was accepted, False otherwise.

        Raises:
            httpx.HTTPError: On network failure after retries.
        """
        logger.info("kalshi.cancel_order", order_id=order_id)
        try:
            self._post(f"/portfolio/orders/{order_id}/decrease", {"reduce_by": 999999})
            return True
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "kalshi.cancel_order.http_error",
                order_id=order_id,
                status_code=exc.response.status_code,
            )
            return False
        except Exception as exc:
            logger.error("kalshi.cancel_order.failed", order_id=order_id, error=str(exc))
            return False


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _parse_dt(value: str | None) -> Optional[datetime]:
    """Parse an ISO-8601 datetime string to a UTC-aware datetime.

    Args:
        value: ISO-8601 string or None.

    Returns:
        UTC-aware datetime or None.
    """
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


# Module-level singleton
kalshi_client = KalshiClient()
