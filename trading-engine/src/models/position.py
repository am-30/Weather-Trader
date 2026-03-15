"""
Pydantic v2 models for internal trade records, positions, and P&L.

These are persisted to the PostgreSQL database and are distinct from
the Kalshi API response schemas in ``market.py``.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, computed_field


class TradeStatus(str, Enum):
    """Lifecycle of a submitted trade."""

    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELED = "canceled"
    REJECTED = "rejected"


class TradeRecord(BaseModel):
    """An immutable record of a single trade that was sent to Kalshi.

    Attributes:
        id:              Internal UUID (set by DB on insert).
        kalshi_order_id: Kalshi-assigned order identifier.
        ticker:          Market ticker.
        side:            ``"yes"`` or ``"no"``.
        action:          ``"buy"`` or ``"sell"``.
        contracts:       Number of contracts.
        limit_price_cents: Limit price in cents.
        fill_price_cents:  Actual fill price in cents (None until filled).
        status:          Current order status.
        created_at:      UTC timestamp of order creation.
        filled_at:       UTC timestamp of fill, if filled.
        strategy_signal: Name of the strategy that generated the order.
        forecast_temp_f: The model-forecast max temperature that drove the trade.
    """

    id: Optional[str] = None
    kalshi_order_id: str
    ticker: str
    side: str
    action: str
    contracts: int = Field(..., ge=1)
    limit_price_cents: int = Field(..., ge=1, le=99)
    fill_price_cents: Optional[int] = None
    status: TradeStatus = TradeStatus.PENDING
    created_at: datetime
    filled_at: Optional[datetime] = None
    strategy_signal: str = ""
    forecast_temp_f: Optional[float] = None

    @computed_field
    @property
    def cost_cents(self) -> int:
        """Total cost of the order in cents.

        For YES buys: contracts * limit_price_cents.
        For NO buys: contracts * (100 - limit_price_cents).

        Returns:
            Integer cost in cents.
        """
        if self.side == "yes":
            return self.contracts * self.limit_price_cents
        return self.contracts * (100 - self.limit_price_cents)

    @computed_field
    @property
    def max_payout_cents(self) -> int:
        """Maximum payout if the contract settles in our favour.

        Returns:
            Contracts * 100 cents.
        """
        return self.contracts * 100


class Position(BaseModel):
    """Aggregated position for a single Kalshi market.

    A position is computed by aggregating all filled TradeRecords for a ticker.

    Attributes:
        ticker:          Market ticker.
        net_yes_contracts: Net YES contracts held (negative = short YES / long NO).
        avg_yes_price_cents: Volume-weighted average price of YES contracts.
        realised_pnl_cents: P&L from closed legs, in cents.
        unrealised_pnl_cents: Mark-to-market P&L from current mid-price.
        last_updated:    UTC timestamp of last position recalculation.
    """

    ticker: str
    net_yes_contracts: int = 0
    avg_yes_price_cents: float = 0.0
    realised_pnl_cents: int = 0
    unrealised_pnl_cents: int = 0
    last_updated: datetime

    @computed_field
    @property
    def total_pnl_cents(self) -> int:
        """Sum of realised and unrealised P&L in cents.

        Returns:
            Total P&L in cents.
        """
        return self.realised_pnl_cents + self.unrealised_pnl_cents

    @computed_field
    @property
    def total_pnl_usd(self) -> float:
        """Total P&L converted to USD (100 cents = $1).

        Returns:
            Float USD value rounded to 2 d.p.
        """
        return round(self.total_pnl_cents / 100.0, 2)


class SystemState(BaseModel):
    """Snapshot of overall system health and statistics.

    Attributes:
        last_weather_fetch:  UTC time of the most recent NWS data pull.
        last_market_fetch:   UTC time of the most recent Kalshi data pull.
        last_forecast_run:   UTC time of the most recent forecast computation.
        last_trade_eval:     UTC time of the most recent trade evaluation.
        total_trades_today:  Number of trades placed today.
        open_positions:      Number of markets with a non-zero position.
        total_pnl_cents:     Sum of all position P&Ls in cents.
        errors_last_hour:    Count of logged ERROR events in the last hour.
    """

    last_weather_fetch: Optional[datetime] = None
    last_market_fetch: Optional[datetime] = None
    last_forecast_run: Optional[datetime] = None
    last_trade_eval: Optional[datetime] = None
    total_trades_today: int = 0
    open_positions: int = 0
    total_pnl_cents: int = 0
    errors_last_hour: int = 0

    @computed_field
    @property
    def total_pnl_usd(self) -> float:
        """System-wide total P&L in USD.

        Returns:
            Float USD value rounded to 2 d.p.
        """
        return round(self.total_pnl_cents / 100.0, 2)
