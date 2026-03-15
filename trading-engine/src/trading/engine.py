"""
Trade execution engine.

Consumes ``TradeSignal`` objects from the strategy module, applies position
and sizing limits, submits orders via the Kalshi client, and persists every
trade attempt to the database.

Sizing logic
------------
Position size in contracts = floor(max_trade_size_usd / (limit_price / 100))
capped at ``settings.max_contracts_per_market`` minus current open contracts.

A conservative limit price is used: for YES buys, yes_ask; for NO buys, no_ask.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
import uuid

import structlog

from src.config import settings
from src.db.connection import get_connection
from src.db.schema import log_system_event
from src.kalshi.client import kalshi_client
from src.models.market import KalshiMarket, OrderRequest, OrderSide, OrderAction
from src.models.position import TradeRecord, TradeStatus
from src.trading.strategy import TradeSignal

logger = structlog.get_logger(__name__)


def _get_open_contract_count(ticker: str) -> int:
    """Read the number of open (unfilled or partially filled) contracts for a ticker.

    Args:
        ticker: Kalshi market ticker.

    Returns:
        Number of open contracts, or 0 on error.
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT COALESCE(SUM(contracts), 0)
                    FROM trades
                    WHERE ticker = %s AND status IN ('pending', 'partial', 'filled')
                    """,
                    (ticker,),
                )
                row = cur.fetchone()
                return int(row[0]) if row else 0
    except Exception as exc:
        logger.error("engine.get_open_contracts.failed", ticker=ticker, error=str(exc))
        return 0


def _compute_contract_count(
    limit_price_cents: int,
    current_open: int,
) -> int:
    """Compute the number of contracts to trade given sizing constraints.

    Args:
        limit_price_cents: Limit price in cents (1–99).
        current_open:      Currently open contracts for this ticker.

    Returns:
        Number of contracts to order (may be 0 if at limit).
    """
    if limit_price_cents <= 0:
        return 0
    max_by_budget = int((settings.max_trade_size_usd * 100) / limit_price_cents)
    remaining_capacity = max(0, settings.max_contracts_per_market - current_open)
    return min(max_by_budget, remaining_capacity)


def _persist_trade(record: TradeRecord) -> None:
    """Insert a trade record into the database.

    Args:
        record: ``TradeRecord`` to persist.

    Returns:
        None

    Raises:
        Nothing — errors are logged but not re-raised.
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO trades
                        (id, kalshi_order_id, ticker, side, action, contracts,
                         limit_price_cents, fill_price_cents, status,
                         strategy_signal, forecast_temp_f, created_at, filled_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (kalshi_order_id) DO NOTHING
                    """,
                    (
                        record.id or str(uuid.uuid4()),
                        record.kalshi_order_id,
                        record.ticker,
                        record.side,
                        record.action,
                        record.contracts,
                        record.limit_price_cents,
                        record.fill_price_cents,
                        record.status.value,
                        record.strategy_signal,
                        record.forecast_temp_f,
                        record.created_at,
                        record.filled_at,
                    ),
                )
        logger.info(
            "engine.trade.persisted",
            order_id=record.kalshi_order_id,
            ticker=record.ticker,
        )
    except Exception as exc:
        logger.error(
            "engine.trade.persist_failed",
            order_id=record.kalshi_order_id,
            error=str(exc),
            exc_info=True,
        )
        log_system_event(
            "trade.persist.error",
            f"Failed to persist trade {record.kalshi_order_id}: {exc}",
            level="error",
        )


def execute_signal(
    signal: TradeSignal,
    market: KalshiMarket,
    dry_run: bool = False,
) -> Optional[TradeRecord]:
    """Execute a single trade signal.

    Validates position limits, computes contract count, submits the order
    to Kalshi, and persists the trade record.

    Args:
        signal:  ``TradeSignal`` from the strategy.
        market:  Current market quote data (used for limit price).
        dry_run: If True, log the intended trade but do not submit to Kalshi.

    Returns:
        ``TradeRecord`` if the order was submitted, else None.
    """
    limit_price = market.yes_ask if signal.side == OrderSide.YES else market.no_ask
    if limit_price == 0:
        logger.warning("engine.skip.zero_ask", ticker=signal.ticker, side=signal.side.value)
        return None

    current_open = _get_open_contract_count(signal.ticker)
    contract_count = _compute_contract_count(limit_price, current_open)

    if contract_count <= 0:
        logger.info(
            "engine.skip.position_limit",
            ticker=signal.ticker,
            current_open=current_open,
            max_allowed=settings.max_contracts_per_market,
        )
        return None

    logger.info(
        "engine.execute_signal",
        ticker=signal.ticker,
        side=signal.side.value,
        contracts=contract_count,
        limit_price_cents=limit_price,
        edge_cents=signal.edge_cents,
        dry_run=dry_run,
    )

    if dry_run:
        log_system_event(
            "trade.dry_run",
            f"DRY RUN: would buy {contract_count}x {signal.side.value} @ {limit_price}¢ on {signal.ticker}",
        )
        return None

    order_request = OrderRequest(
        ticker=signal.ticker,
        action=signal.action,
        side=signal.side,
        count=contract_count,
        type="limit",
        yes_price=limit_price if signal.side == OrderSide.YES else (100 - limit_price),
        client_order_id=str(uuid.uuid4()),
    )

    try:
        order_response = kalshi_client.submit_order(order_request)
    except Exception as exc:
        logger.error(
            "engine.submit_order.failed",
            ticker=signal.ticker,
            error=str(exc),
            exc_info=True,
        )
        log_system_event("trade.submit.error", str(exc), level="error")
        return None

    record = TradeRecord(
        kalshi_order_id=order_response.order_id,
        ticker=signal.ticker,
        side=signal.side.value,
        action=signal.action.value,
        contracts=contract_count,
        limit_price_cents=limit_price,
        status=TradeStatus.PENDING,
        created_at=order_response.created_time,
        strategy_signal=signal.strategy_name,
        forecast_temp_f=signal.forecast_mean_f,
    )
    _persist_trade(record)

    log_system_event(
        "trade.placed",
        f"Placed {contract_count}x {signal.side.value} @ {limit_price}¢ on {signal.ticker}",
        details={
            "order_id": order_response.order_id,
            "ticker": signal.ticker,
            "edge_cents": signal.edge_cents,
        },
    )
    return record


def run_trade_evaluation(
    dry_run: bool = False,
) -> dict:
    """Run a full trade evaluation cycle: fetch markets, get forecast, scan, execute.

    This is the top-level function called by the scheduler on each trade
    evaluation interval.

    Args:
        dry_run: If True, signals are computed but no orders are placed.

    Returns:
        Dict with keys ``markets_fetched``, ``signals_generated``, ``trades_placed``.
    """
    from src.forecasting.temperature import get_latest_forecast, generate_forecast
    from src.trading.strategy import scan_markets

    result: dict = {"markets_fetched": 0, "signals_generated": 0, "trades_placed": 0}

    try:
        markets = kalshi_client.get_markets(event_ticker="KXMAXTEMP", status="open")
        kbos_markets = [m for m in markets if settings.nws_station in m.ticker.upper()]
        result["markets_fetched"] = len(kbos_markets)
    except Exception as exc:
        logger.error("engine.run_eval.markets_failed", error=str(exc))
        return result

    forecast = get_latest_forecast()
    if forecast is None:
        logger.info("engine.run_eval.no_forecast_cached_generating")
        try:
            forecast = generate_forecast()
        except Exception as exc:
            logger.error("engine.run_eval.forecast_failed", error=str(exc))
            return result

    signals = scan_markets(kbos_markets, forecast)
    result["signals_generated"] = len(signals)

    for signal in signals:
        try:
            market = kalshi_client.get_market(signal.ticker)
        except Exception as exc:
            logger.warning("engine.run_eval.market_refresh_failed", ticker=signal.ticker, error=str(exc))
            continue

        trade = execute_signal(signal, market, dry_run=dry_run)
        if trade:
            result["trades_placed"] += 1

    logger.info("engine.run_eval.complete", **result)
    log_system_event("trade_eval.complete", f"Eval done: {result}", details=result)
    return result
