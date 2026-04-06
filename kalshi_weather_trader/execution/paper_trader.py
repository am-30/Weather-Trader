"""
Paper trading simulator for the Kalshi weather trading system.

Mirrors the intended live trading rules without placing real orders:
- Entries at 10 AM ET when model has edge AND ask < 50¢
- Limit-sell exit when market bid reaches 75¢
- Settlement close based on official NWS daily high

Three scheduler entry points:
  run_paper_entry_10am()       — 10:00 AM ET daily (new job)
  check_limit_sell_exits()     — called from job_evaluate_trade (every 5 min)
  run_paper_settlement_close() — called from job_confirm_settlement (10:05 AM)
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Optional

import pytz
import structlog

from kalshi_weather_trader.config.settings import get_target_date, settings
from kalshi_weather_trader.db import db_manager
from kalshi_weather_trader.db.schemas import PaperTradeDocument

logger = structlog.get_logger(__name__)
_EASTERN = pytz.timezone("America/New_York")


def run_paper_entry_10am(target_date: Optional[date] = None) -> None:
    """Simulate 10 AM paper trade entries for the given target date.

    Fetches live market prices and model edge, then records simulated positions
    for any strike where model has positive edge AND ask < settings.paper_entry_max_ask_cents.
    Budget is split equally among qualifying signals (flat mode).

    Args:
        target_date: Trading date. Defaults to ``get_target_date()``.

    Returns:
        None

    Raises:
        Nothing — all errors are logged.
    """
    if target_date is None:
        target_date = get_target_date()

    log = logger.bind(date=str(target_date), fn="run_paper_entry_10am")

    try:
        from kalshi_weather_trader.ingestion.kalshi_fetcher import KalshiFetcher
        from kalshi_weather_trader.ingestion.nwp_fetcher import get_nwp_curve
        from kalshi_weather_trader.quant.edge_computer import compute_edge_table

        # Guard: skip if market already settled
        market_row = db_manager.get_market(target_date)
        if market_row and market_row.market_status == "settled":
            log.info("paper_entry.skipped_settled")
            return

        state = db_manager.get_system_state(target_date)
        if state is None:
            log.warning("paper_entry.no_system_state")
            return

        nwp_curve = get_nwp_curve(target_date)
        fetcher = KalshiFetcher()
        markets = fetcher.get_temperature_markets(target_date)

        if not markets:
            log.warning("paper_entry.no_markets")
            return

        edge_rows, diag, _ = compute_edge_table(
            target_date=target_date,
            state=state,
            market_row=market_row,
            nwp_curve=nwp_curve,
            markets=markets,
            edge_threshold=settings.edge_threshold,
        )
        log.debug("paper_entry.edge_computed", n_rows=len(edge_rows), diag_lines=len(diag))

        # Filter: positive edge AND ask < threshold AND signal is a buy
        qualifying = [
            r for r in edge_rows
            if r.action in ("BUY YES", "BUY NO")
            and r.edge > settings.edge_threshold
            and r.yes_ask > 0
            and r.yes_ask < settings.paper_entry_max_ask_cents
        ]

        if not qualifying:
            log.info("paper_entry.no_qualifying_signals",
                     total_rows=len(edge_rows),
                     max_ask=settings.paper_entry_max_ask_cents)
            return

        log.info("paper_entry.qualifying", count=len(qualifying))

        # Flat budget: split daily budget equally
        n = len(qualifying)
        per_trade_budget = settings.paper_daily_budget / n
        entry_time = datetime.now(timezone.utc)

        for row in qualifying:
            # Determine entry price: ask for BUY_YES, no-ask for BUY_NO
            if row.action == "BUY YES":
                entry_cents = row.yes_ask
                action_str = "BUY_YES"
            else:
                entry_cents = 100 - row.yes_bid  # cost of the NO side
                action_str = "BUY_NO"

            ask_dec = entry_cents / 100.0
            # How many contracts fit in the per-trade budget?
            # Cost per contract = ask_dec dollars (e.g. 0.33 for a 33¢ contract).
            contracts = max(1, int(per_trade_budget / ask_dec))
            cost_usd = round(ask_dec * contracts, 2)

            doc = PaperTradeDocument(
                target_date=target_date,
                market_ticker=row.ticker,
                action=action_str,
                kalshi_strike=row.kalshi_strike,
                entry_at_utc=entry_time,
                entry_price_cents=entry_cents,
                contracts=contracts,
                cost_usd=cost_usd,
                fair_value_prob=row.model_prob,
                edge_at_entry=row.edge,
                budget_mode=settings.paper_budget_mode,
                status="open",
            )
            db_manager.insert_paper_trade(doc)
            log.info(
                "paper_entry.inserted",
                ticker=row.ticker,
                action=action_str,
                entry_cents=entry_cents,
                contracts=contracts,
                edge=row.edge,
            )

    except Exception as exc:
        logger.error("paper_entry.failed", error=str(exc), exc_info=True)


def check_limit_sell_exits(target_date: Optional[date] = None) -> None:
    """Check open paper positions against current market prices.

    Closes any position whose limit-sell price has been reached
    (bid >= settings.paper_limit_sell_cents for BUY_YES positions;
    the NO side equivalent for BUY_NO).

    Called every 5 minutes by the trade evaluation scheduler job.

    Args:
        target_date: Trading date. Defaults to ``get_target_date()``.

    Returns:
        None

    Raises:
        Nothing — all errors are logged.
    """
    if target_date is None:
        target_date = get_target_date()

    try:
        open_positions = db_manager.get_open_paper_trades(target_date)
        if not open_positions:
            return

        from kalshi_weather_trader.ingestion.kalshi_fetcher import KalshiFetcher
        fetcher = KalshiFetcher()
        markets = fetcher.get_temperature_markets(target_date)
        if not markets:
            return

        # Build price lookup by ticker
        bid_by_ticker = {m["ticker"]: (m.get("yes_bid") or 0) for m in markets}
        ask_by_ticker = {m["ticker"]: (m.get("yes_ask") or 0) for m in markets}

        limit_cents = settings.paper_limit_sell_cents
        now_utc = datetime.now(timezone.utc)

        for pos in open_positions:
            bid = bid_by_ticker.get(pos.market_ticker, 0)
            ask = ask_by_ticker.get(pos.market_ticker, 0)

            triggered = False
            if pos.action == "BUY_YES" and bid >= limit_cents:
                triggered = True
            elif pos.action == "BUY_NO":
                # NO position value = 100 - yes_ask
                no_value = (100 - ask) if ask > 0 else 0
                if no_value >= limit_cents:
                    triggered = True

            if triggered:
                pnl_cents = (limit_cents - pos.entry_price_cents) * pos.contracts
                pnl_usd = pnl_cents / 100.0
                db_manager.update_paper_trade_exit(
                    position_id=pos.position_id,
                    status="limit_sell_closed",
                    exit_at_utc=now_utc,
                    exit_price_cents=limit_cents,
                    pnl_cents=pnl_cents,
                    pnl_usd=pnl_usd,
                )
                logger.info(
                    "paper_limit_sell.triggered",
                    position_id=pos.position_id,
                    ticker=pos.market_ticker,
                    entry_cents=pos.entry_price_cents,
                    exit_cents=limit_cents,
                    pnl_usd=round(pnl_usd, 2),
                )

    except Exception as exc:
        logger.error("paper_limit_sell.failed", error=str(exc), exc_info=True)


def run_paper_settlement_close(target_date: Optional[date] = None) -> None:
    """Close open paper positions at settlement using the official NWS daily high.

    Called after the NWS CLI product is confirmed at 10:05 AM the following day.
    Positions not already closed by limit-sell are resolved as win or loss.

    Args:
        target_date: The date to settle. Defaults to yesterday (since this
                     runs at 10:05 AM the *next* day).

    Returns:
        None

    Raises:
        Nothing — all errors are logged.
    """
    if target_date is None:
        # Settlement always refers to yesterday's contracts
        now_et = datetime.now(_EASTERN)
        target_date = (now_et - timedelta(days=1)).date()

    log = logger.bind(date=str(target_date), fn="run_paper_settlement_close")

    try:
        market = db_manager.get_market(target_date)
        if market is None or market.final_official_high is None:
            log.warning("paper_settlement.no_official_high")
            return

        official_high = market.final_official_high
        open_positions = db_manager.get_open_paper_trades(target_date)

        if not open_positions:
            log.debug("paper_settlement.no_open_positions")
            return

        now_utc = datetime.now(timezone.utc)

        for pos in open_positions:
            # Determine win/loss based on official high and position side
            if pos.action == "BUY_YES":
                win = official_high >= pos.kalshi_strike
            else:  # BUY_NO
                win = official_high < pos.kalshi_strike

            exit_price_cents = 100 if win else 0
            pnl_cents = (exit_price_cents - pos.entry_price_cents) * pos.contracts
            pnl_usd = pnl_cents / 100.0
            status = "settled_win" if win else "settled_loss"

            db_manager.update_paper_trade_exit(
                position_id=pos.position_id,
                status=status,
                exit_at_utc=now_utc,
                exit_price_cents=exit_price_cents,
                pnl_cents=pnl_cents,
                pnl_usd=pnl_usd,
                official_high_f=official_high,
                settlement_win=win,
            )
            log.info(
                "paper_settlement.closed",
                position_id=pos.position_id,
                ticker=pos.market_ticker,
                strike=pos.kalshi_strike,
                action=pos.action,
                official_high=official_high,
                win=win,
                pnl_usd=round(pnl_usd, 2),
            )

    except Exception as exc:
        logger.error("paper_settlement.failed", error=str(exc), exc_info=True)
