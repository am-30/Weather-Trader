"""
Pydantic v2 models representing Kalshi market structures.

These models map directly to the Kalshi REST API v2 response schemas.
All monetary values are stored in cents (integer) as returned by the API.
All timestamps are UTC-aware datetimes.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class MarketStatus(str, Enum):
    """Lifecycle state of a Kalshi market."""

    OPEN = "open"
    CLOSED = "closed"
    SETTLED = "settled"
    UNKNOWN = "unknown"


class OrderSide(str, Enum):
    """Which side of a binary market the order is on."""

    YES = "yes"
    NO = "no"


class OrderAction(str, Enum):
    """Buy or sell action for an order."""

    BUY = "buy"
    SELL = "sell"


class KalshiMarket(BaseModel):
    """A Kalshi binary market contract.

    Attributes:
        ticker:        Unique market ticker (e.g. ``"KXMAXTEMP-KBOS-25MAR15-B72"``).
        event_ticker: Ticker of the parent event.
        title:         Human-readable market title.
        status:        Current lifecycle status.
        yes_bid:       Best bid for YES contracts, in cents.
        yes_ask:       Best ask for YES contracts, in cents.
        no_bid:        Best bid for NO contracts, in cents.
        no_ask:        Best ask for NO contracts, in cents.
        last_price:    Last traded price in cents.
        volume:        Total volume (number of contracts traded).
        open_interest: Outstanding open contracts.
        close_time:    UTC time when trading closes.
        expiration_time: UTC expiration / settlement time.
        result:        Settlement outcome (``"yes"`` / ``"no"``), or ``None``.
    """

    ticker: str
    event_ticker: str
    title: str
    status: MarketStatus = MarketStatus.UNKNOWN
    yes_bid: int = Field(default=0, ge=0, le=100)
    yes_ask: int = Field(default=0, ge=0, le=100)
    no_bid: int = Field(default=0, ge=0, le=100)
    no_ask: int = Field(default=0, ge=0, le=100)
    last_price: int = Field(default=0, ge=0, le=100)
    volume: int = Field(default=0, ge=0)
    open_interest: int = Field(default=0, ge=0)
    close_time: Optional[datetime] = None
    expiration_time: Optional[datetime] = None
    result: Optional[str] = None
    fetched_at: datetime = Field(
        default_factory=lambda: __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        )
    )

    @property
    def yes_mid(self) -> float:
        """Mid-price of the YES contract, in cents.

        Returns:
            Average of yes_bid and yes_ask, or last_price if spread is zero.
        """
        if self.yes_bid == 0 and self.yes_ask == 0:
            return float(self.last_price)
        return (self.yes_bid + self.yes_ask) / 2.0

    @property
    def implied_probability(self) -> float:
        """YES probability implied by the mid-price (0.0 – 1.0).

        Returns:
            Probability between 0 and 1.
        """
        return self.yes_mid / 100.0


class OrderBook(BaseModel):
    """Snapshot of a market's order book at a point in time.

    Attributes:
        ticker:     Market ticker this book belongs to.
        yes_bids:   List of (price_cents, quantity) tuples on the YES bid side.
        yes_asks:   List of (price_cents, quantity) tuples on the YES ask side.
        fetched_at: UTC timestamp of the snapshot.
    """

    ticker: str
    yes_bids: list[tuple[int, int]] = Field(default_factory=list)
    yes_asks: list[tuple[int, int]] = Field(default_factory=list)
    fetched_at: datetime = Field(
        default_factory=lambda: __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        )
    )


class OrderRequest(BaseModel):
    """Parameters for submitting an order to Kalshi.

    Attributes:
        ticker:    Market ticker to trade.
        action:    Buy or sell.
        side:      YES or NO.
        count:     Number of contracts.
        type:      Order type — ``"limit"`` or ``"market"``.
        yes_price: Limit price in cents (required for limit orders).
        client_order_id: Optional idempotency key.
    """

    ticker: str
    action: OrderAction
    side: OrderSide
    count: int = Field(..., ge=1)
    type: str = Field(default="limit")
    yes_price: Optional[int] = Field(None, ge=1, le=99)
    client_order_id: Optional[str] = None

    @field_validator("yes_price")
    @classmethod
    def price_required_for_limit(
        cls, v: Optional[int], info: object
    ) -> Optional[int]:
        """Require yes_price when order type is 'limit'.

        Args:
            v: The yes_price value.
            info: Pydantic validation info carrying other field values.

        Returns:
            The validated yes_price.

        Raises:
            ValueError: If type is 'limit' and yes_price is not set.
        """
        values = getattr(info, "data", {})
        if values.get("type") == "limit" and v is None:
            raise ValueError("yes_price is required for limit orders")
        return v


class OrderResponse(BaseModel):
    """API response after order submission.

    Attributes:
        order_id:   Kalshi-assigned order identifier.
        status:     Order status (e.g. ``"resting"`` / ``"filled"`` / ``"canceled"``).
        created_time: UTC creation timestamp.
        ticker:     Market the order was placed in.
        side:       Which side was ordered.
        action:     Buy or sell.
        count:      Contracts requested.
        yes_price:  Limit price in cents.
    """

    order_id: str
    status: str
    created_time: datetime
    ticker: str
    side: OrderSide
    action: OrderAction
    count: int
    yes_price: Optional[int] = None
