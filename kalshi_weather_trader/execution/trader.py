"""
Kalshi order execution engine with kill switch and fractional Kelly sizing.

Every trade decision follows this checklist:
1. Check ``markets.auto_trade_enabled`` (kill switch) — abort if False.
2. Check ``settings.dry_run`` — log but do not place real order if True.
3. Run Monte Carlo to compute fair-value probability.
4. Compute edge = fair_value - market_price.
5. Check edge vs. ``settings.edge_threshold`` — no trade if edge insufficient.
6. Size position using fractional Kelly (25% Kelly by default).
7. Submit order to Kalshi (or log dry-run).
8. Persist ``TradeLogDocument`` regardless of outcome.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from typing import Optional

import structlog

from kalshi_weather_trader.config.settings import get_target_date, settings
from kalshi_weather_trader.db import db_manager
from kalshi_weather_trader.db.schemas import TradeLogDocument

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Kelly sizing
# ---------------------------------------------------------------------------


def compute_kelly_contracts(
    p: float,
    ask_decimal: float,
    max_size_usd: float,
    kelly_fraction: float,
    max_contracts: int,
) -> Optional[int]:
    """Compute the number of contracts to trade using fractional Kelly.

    Formula:
        b = (1 / ask_decimal) - 1       (payout ratio)
        kelly = (p * b - (1 - p)) / b   (full Kelly fraction)
        contracts = floor(kelly_fraction * kelly * max_size_usd / (ask_decimal * 100))

    Returns None (no trade) if full Kelly is <= 0, indicating negative edge.

    Args:
        p:             Model probability of YES outcome (0.0–1.0).
        ask_decimal:   Market ask price as a decimal (e.g. 0.55 for 55 cents).
        max_size_usd:  Maximum dollar amount to risk.
        kelly_fraction: Fractional Kelly multiplier (e.g. 0.25).
        max_contracts:  Hard cap on contracts.

    Returns:
        Integer number of contracts (>= 1), or None if Kelly <= 0 or fractional
        sizing rounds below 1 contract.

    Raises:
        Nothing.
    """
    if ask_decimal <= 0.0 or ask_decimal >= 1.0:
        logger.warning("trader.kelly.invalid_ask", ask_decimal=ask_decimal)
        return None

    b = (1.0 / ask_decimal) - 1.0
    kelly = (p * b - (1.0 - p)) / b

    if kelly <= 0:
        logger.debug("trader.kelly.negative", p=p, ask=ask_decimal, kelly=round(kelly, 4))
        return None

    raw_contracts = kelly_fraction * kelly * max_size_usd / (ask_decimal * 100.0)
    contracts = min(int(raw_contracts), max_contracts)
    if contracts < 1:
        logger.debug(
            "trader.kelly.below_min",
            p=p,
            ask=ask_decimal,
            raw_contracts=round(raw_contracts, 3),
        )
        return None

    logger.debug(
        "trader.kelly.computed",
        p=p,
        ask=ask_decimal,
        b=round(b, 4),
        kelly=round(kelly, 4),
        raw_contracts=round(raw_contracts, 2),
        contracts=contracts,
    )
    return contracts


# ---------------------------------------------------------------------------
# Trade evaluation
# ---------------------------------------------------------------------------


def evaluate_and_trade(target_date: Optional[date] = None) -> None:
    """Evaluate market conditions and place a trade if edge is sufficient.

    This is the top-level function called every 5 minutes by the scheduler.

    Steps:
        1. Kill switch check
        2. Fetch ASOS + system state + NWP data
        3. Run Monte Carlo pricing
        4. Compute edge for each available market
        5. Size position with Kelly
        6. Place order (or dry-run log)
        7. Persist trade log

    Args:
        target_date: Active trading date. Defaults to today's target.

    Returns:
        None

    Raises:
        Nothing — all errors are caught and logged.
    """
    from kalshi_weather_trader.ingestion.asos_fetcher import fetch_current_observation
    from kalshi_weather_trader.ingestion.kalshi_fetcher import get_kalshi_fetcher
    from kalshi_weather_trader.ingestion.nwp_fetcher import get_nwp_curve
    from kalshi_weather_trader.quant.mc_params_builder import build_mc_params
    from kalshi_weather_trader.quant.monte_carlo import price_full_distribution

    if target_date is None:
        target_date = get_target_date()

    # ------------------------------------------------------------------
    # 1. Kill switch check
    # ------------------------------------------------------------------
    try:
        market_row = db_manager.get_market(target_date)
        if market_row is not None and not market_row.auto_trade_enabled:
            logger.info(
                "trader.evaluate.kill_switch_active",
                date=str(target_date),
            )
            return
    except Exception as exc:
        logger.error("trader.evaluate.kill_switch_check_failed", error=str(exc))
        return  # Fail safe — do not trade if we can't check the kill switch

    # ------------------------------------------------------------------
    # 2. Gather state
    # ------------------------------------------------------------------
    try:
        asos = fetch_current_observation()
        if asos is None:
            logger.warning("trader.evaluate.no_asos")
            return

        state = db_manager.get_system_state(target_date)
        nwp_curve = get_nwp_curve(target_date)

    except Exception as exc:
        logger.error("trader.evaluate.state_gather_failed", error=str(exc))
        return

    # ------------------------------------------------------------------
    # 3. Fetch Kalshi markets
    # ------------------------------------------------------------------
    try:
        fetcher = get_kalshi_fetcher()
        markets = fetcher.get_temperature_markets(target_date)
        if not markets:
            logger.info("trader.evaluate.no_kalshi_markets", date=str(target_date))
            return
    except Exception as exc:
        logger.error("trader.evaluate.kalshi_fetch_failed", error=str(exc))
        return

    # Fetch existing positions to avoid over-sizing
    existing_positions: dict[str, int] = {}
    try:
        raw_positions = fetcher.get_positions()
        for pos in raw_positions:
            t = pos.get("ticker", "")
            qty = pos.get("position", 0)
            if t and qty:
                existing_positions[t] = int(qty)
        logger.info("trader.positions.fetched", count=len(existing_positions))
    except Exception as exc:
        logger.warning("trader.positions.fetch_failed", error=str(exc))

    # ------------------------------------------------------------------
    # 4. Price all strikes from one Monte Carlo run
    # ------------------------------------------------------------------
    # Collect all temperature thresholds needed for bucket probability computation.
    # floor_strike and cap_strike come from the Kalshi API; both are needed so
    # compute_yes_prob can determine market type without relying on the ticker prefix.
    # Collect all temperature thresholds at the half-integer rounding boundaries.
    # NWS rounds to nearest integer, so the settlement boundary between bucket
    # {38,39} and bucket {40,41} is at 39.5°F, not 40.0°F.  Including these
    # half-integer values in the MC strike list means the CDF is computed directly
    # at each boundary rather than interpolated.
    all_thresholds: set[float] = set()
    market_by_extracted_strike: dict[float, dict] = {}
    for m in markets:
        floor_raw = m.get("floor_strike")
        cap_raw = m.get("cap_strike")
        if floor_raw is not None:
            f = float(floor_raw)
            all_thresholds.add(f)
            all_thresholds.add(f - 0.5)   # rounding lower boundary
        if cap_raw is not None:
            c = float(cap_raw)
            all_thresholds.add(c)
            all_thresholds.add(c + 0.5)   # rounding upper boundary
        extracted = fetcher.extract_strike_from_market(m)
        if extracted is not None:
            all_thresholds.add(extracted)
            market_by_extracted_strike[extracted] = m

    if not all_thresholds:
        logger.warning("trader.evaluate.no_valid_strikes")
        return

    all_strikes = sorted(all_thresholds)

    try:
        mc_params = build_mc_params(target_date, state, asos, market_row, nwp_curve)
        mc_context = json.dumps({
            "T0": round(mc_params.T0, 2),
            "bias": round(mc_params.bias, 4),
            "sigma": round(mc_params.sigma, 4),
            "theta": round(mc_params.theta, 4),
            "hour_offset": mc_params.hour_offset,
            "drift_adj": round(mc_params.drift_adj, 4),
            "n_paths": mc_params.n_paths,
            "hard_floor": round(mc_params.hard_floor, 2),
            "is_future_day": mc_params.is_future_day,
        })
        mc_result = price_full_distribution(mc_params, all_strikes, target_date)
    except Exception as exc:
        logger.error("trader.evaluate.mc_failed", error=str(exc))
        return

    # ------------------------------------------------------------------
    # 5. Evaluate edge for each strike
    # ------------------------------------------------------------------
    from kalshi_weather_trader.quant.monte_carlo import compute_normalized_market_probs

    prob_by_ticker, prob_sum_raw, partition_gaps = compute_normalized_market_probs(
        markets, mc_result.probabilities
    )
    if abs(prob_sum_raw - 1.0) > 0.01:
        logger.warning(
            "trader.evaluate.partition_sum_off",
            sum_raw=round(prob_sum_raw, 4),
            n_gaps=len(partition_gaps),
        )

    best_edge = 0.0
    best_strike = None
    best_action = None
    best_market = None

    # Separate tracking for no-trade audit log: best candidate regardless of threshold.
    # Initialized to -999 so even a fully negative edge is recorded.
    _nt_edge: float = -999.0
    _nt_strike: Optional[float] = None
    _nt_action: str = "BUY_YES"
    _nt_market: Optional[dict] = None

    for m in markets:
        extracted_strike = fetcher.extract_strike_from_market(m)
        if extracted_strike is None:
            continue

        ticker = m.get("ticker", "")
        fair_p = prob_by_ticker.get(ticker)
        if fair_p is None:
            continue

        yes_bid = m.get("yes_bid") or 0
        yes_ask = m.get("yes_ask") or 0
        if yes_bid == 0 or yes_ask == 0:
            continue

        bid_dec = yes_bid / 100.0
        ask_dec = yes_ask / 100.0

        edge_yes = fair_p - ask_dec  # edge for buying YES
        edge_no = (1.0 - fair_p) - (1.0 - bid_dec)  # = bid_dec - fair_p

        # Track the best-seen candidate for no-trade audit (regardless of threshold)
        candidate_edge = edge_yes if edge_yes >= edge_no else edge_no
        candidate_action = "BUY_YES" if edge_yes >= edge_no else "BUY_NO"
        if candidate_edge > _nt_edge:
            _nt_edge = candidate_edge
            _nt_strike = extracted_strike
            _nt_action = candidate_action
            _nt_market = m

        if edge_yes > settings.edge_threshold and edge_yes > best_edge:
            best_edge = edge_yes
            best_strike = extracted_strike
            best_action = "BUY_YES"
            best_market = m
        elif edge_no > settings.edge_threshold and edge_no > best_edge:
            best_edge = edge_no
            best_strike = extracted_strike
            best_action = "BUY_NO"
            best_market = m

    if best_strike is None:
        logger.info(
            "trader.evaluate.no_edge",
            date=str(target_date),
            n_markets=len(markets),
        )
        _log_no_trade(
            target_date,
            mc_context,
            _nt_market if _nt_market is not None else (markets[0] if markets else None),
            _nt_strike,
            _nt_action,
            _nt_edge if _nt_edge > -999.0 else None,
        )
        return

    # ------------------------------------------------------------------
    # 6. Size position with Kelly
    # ------------------------------------------------------------------
    m = best_market  # type: ignore[assignment]
    # Use the normalized probability already computed for the winning market
    fair_p = prob_by_ticker[m.get("ticker", "")]
    yes_ask = m.get("yes_ask") or 0
    yes_bid = m.get("yes_bid") or 0

    if best_action == "BUY_YES":
        ask_dec = yes_ask / 100.0
        price_cents = yes_ask
        side = "yes"
        p_kelly = fair_p
    else:
        ask_dec = (100 - yes_bid) / 100.0  # cost of buying NO
        price_cents = 100 - yes_bid
        side = "no"
        p_kelly = 1.0 - fair_p

    contracts = compute_kelly_contracts(
        p=p_kelly,
        ask_decimal=ask_dec,
        max_size_usd=settings.max_trade_size_usd,
        kelly_fraction=settings.kelly_fraction,
        max_contracts=settings.max_contracts_per_market,
    )

    if contracts is None:
        logger.info("trader.evaluate.kelly_no_trade", strike=best_strike, action=best_action)
        return

    kelly_frac_full = (p_kelly * ((1.0 / ask_dec) - 1.0) - (1.0 - p_kelly)) / ((1.0 / ask_dec) - 1.0)

    # ------------------------------------------------------------------
    # 7. Place order (or dry-run)
    # ------------------------------------------------------------------
    ticker = m.get("ticker", "")

    # Reduce sizing by existing position
    current_exposure = existing_positions.get(ticker, 0)
    contracts = max(0, contracts - current_exposure)
    if contracts == 0:
        logger.info("trader.position.already_full", ticker=ticker, exposure=current_exposure)
        return

    order_id = None

    if settings.dry_run:
        logger.info(
            "trader.execute.dry_run",
            ticker=ticker,
            action=best_action,
            strike=best_strike,
            contracts=contracts,
            price_cents=price_cents,
            edge=round(best_edge, 4),
        )
        status = "dry_run"
    else:
        try:
            # Re-check kill switch immediately before submitting
            market_row_fresh = db_manager.get_market(target_date)
            if market_row_fresh and not market_row_fresh.auto_trade_enabled:
                logger.warning("trader.execute.kill_switch_last_check_failed")
                return

            order_data = fetcher.submit_order(
                ticker=ticker,
                side=side,
                action="buy",
                count=contracts,
                yes_price_cents=price_cents,
            )
            order_id = (order_data.get("order") or {}).get("order_id")
            status = "filled"
            logger.info(
                "trader.execute.order_submitted",
                ticker=ticker,
                action=best_action,
                contracts=contracts,
                order_id=order_id,
            )
        except Exception as exc:
            logger.error("trader.execute.order_failed", error=str(exc), exc_info=True)
            status = "failed"

    # ------------------------------------------------------------------
    # 8. Persist trade log
    # ------------------------------------------------------------------
    kalshi_implied = (yes_bid + yes_ask) / 200.0
    try:
        log = TradeLogDocument(
            target_date=target_date,
            executed_at_utc=datetime.now(timezone.utc),
            market_ticker=ticker,
            action=best_action,
            kalshi_strike=best_strike,
            contracts=contracts,
            price_cents=price_cents,
            fair_value_prob=round(fair_p, 6),
            kalshi_implied_prob=round(kalshi_implied, 6),
            edge_at_execution=round(best_edge, 6),
            kelly_fraction=round(kelly_frac_full * settings.kelly_fraction, 6),
            dry_run=settings.dry_run,
            order_id=order_id,
            status=status,
            notes=mc_context,
        )
        db_manager.insert_trade_log(log)
    except Exception as exc:
        logger.error("trader.execute.log_failed", error=str(exc))


def _log_no_trade(
    target_date: date,
    mc_context: str,
    best_market: Optional[dict],
    best_strike: Optional[float],
    best_action: str,
    best_edge: Optional[float],
) -> None:
    """Log a no-trade decision to trade_logs for audit purposes.

    Records the market that had the highest edge (even if below threshold) so
    the audit trail accurately reflects the near-miss rather than an arbitrary
    first market.

    Args:
        target_date:  Active trading date.
        mc_context:   JSON string of MCParams fields for audit.
        best_market:  Market dict with the highest edge seen this cycle, or None.
        best_strike:  Strike for best_market, or None.
        best_action:  "BUY_YES" or "BUY_NO" for best_market.
        best_edge:    Highest edge seen (may be negative), or None if no markets
                      were parseable.

    Returns:
        None

    Raises:
        Nothing.
    """
    if best_market is None:
        return

    ticker = best_market.get("ticker", "unknown")
    strike = best_strike if best_strike is not None else 0.0
    fair_p = mc_result.probabilities.get(strike, 0.5)
    yes_bid = best_market.get("yes_bid") or 50
    yes_ask = best_market.get("yes_ask") or 50
    kalshi_implied = (yes_bid + yes_ask) / 200.0
    edge_val = best_edge if best_edge is not None else 0.0

    # Price cents depends on direction
    price_cents = yes_ask if best_action == "BUY_YES" else (100 - yes_bid)

    try:
        log = TradeLogDocument(
            target_date=target_date,
            executed_at_utc=datetime.now(timezone.utc),
            market_ticker=ticker,
            action=best_action,
            kalshi_strike=strike,
            contracts=0,
            price_cents=price_cents,
            fair_value_prob=round(fair_p, 6),
            kalshi_implied_prob=round(kalshi_implied, 6),
            edge_at_execution=round(edge_val, 6),
            dry_run=settings.dry_run,
            status="no_trade",
            notes=mc_context,
        )
        db_manager.insert_trade_log(log)
    except Exception as exc:
        logger.debug("trader.no_trade_log_failed", error=str(exc))
