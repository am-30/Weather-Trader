"""
Shared edge computation for the Kalshi weather trading system.

Extracts the Monte Carlo → probability normalization → per-ticker edge
calculation from the UI into a reusable function.  Callers: the Trading Desk
UI, the paper trading scheduler job, and (eventually) the live trader.

No Streamlit imports.  No database writes.  Pure computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Optional

import pytz
import structlog

logger = structlog.get_logger(__name__)

_EASTERN = pytz.timezone("America/New_York")


@dataclass
class EdgeRow:
    """Per-market edge computation result.

    Attributes:
        ticker:         Kalshi market ticker string.
        strike_label:   Human-readable range label (e.g. '38–39°F').
        kalshi_strike:  Floor strike temperature extracted from the market.
        action:         'BUY_YES', 'BUY_NO', '—', or 'NO_LIQUIDITY'.
        yes_bid:        Current YES bid in cents (0 if absent).
        yes_ask:        Current YES ask in cents (0 if absent).
        model_prob:     Model's estimated P(YES) for this market bucket.
        edge:           Signed edge — positive means the model favours the action.
                        For BUY_YES: model_prob - ask_decimal.
                        For BUY_NO:  (1 - model_prob) - no_ask_decimal.
        ask_decimal:    Cost to enter the signal side as a decimal (0–1).
    """

    ticker: str
    strike_label: str
    kalshi_strike: float
    action: str
    yes_bid: int
    yes_ask: int
    model_prob: float
    edge: float
    ask_decimal: float


def compute_edge_table(
    target_date: date,
    state,          # SystemStateDocument | None
    market_row,     # MarketDocument | None
    nwp_curve: Optional[list[float]],
    markets: list[dict],
    edge_threshold: float = 0.05,
) -> tuple[list[EdgeRow], list[str], float | None]:
    """Run Monte Carlo once and compute per-market edge rows.

    This is the canonical edge computation used by both the Trading Desk UI
    and the paper trading scheduler.  It does not write to the database and
    has no Streamlit dependencies.

    Args:
        target_date:    Active trading date.
        state:          SystemStateDocument from DB, or None if unavailable.
        market_row:     MarketDocument for hard-floor, or None.
        nwp_curve:      Blended NWP hourly temperature curve (24 values), or None.
        markets:        Normalized Kalshi market dicts (from KalshiFetcher).
        edge_threshold: Minimum edge to label a signal BUY_YES / BUY_NO.

    Returns:
        Tuple of:
          - list[EdgeRow]: One entry per market with valid strike.
          - list[str]: Diagnostic messages (for logging / UI expander).
          - float | None: Raw probability partition sum before normalization.

    Raises:
        Nothing — errors are appended to diagnostics and an empty list returned.
    """
    from kalshi_weather_trader.ingestion.kalshi_fetcher import KalshiFetcher
    from kalshi_weather_trader.quant.mc_params_builder import build_mc_params
    from kalshi_weather_trader.quant.monte_carlo import (
        compute_normalized_market_probs,
        price_full_distribution,
    )

    diag: list[str] = []
    edge_rows: list[EdgeRow] = []
    prob_sum_raw: float | None = None

    if not markets:
        diag.append("compute_edge_table: no markets provided")
        return edge_rows, diag, prob_sum_raw

    if state is None:
        diag.append("compute_edge_table: no system state — cannot run MC")
        return edge_rows, diag, prob_sum_raw

    try:
        now_et = datetime.now(timezone.utc).astimezone(_EASTERN)
        hour_et = now_et.hour
        is_future_day = target_date > now_et.date()
        is_dst = bool(now_et.dst())
        hour_offset = (1 if is_dst else 0) if is_future_day else hour_et

        hard_floor = (
            (market_row.current_max_observed if market_row else None)
            or state.kalman_temp_estimate
        )
        effective_curve = nwp_curve if nwp_curve else [state.kalman_temp_estimate] * 24

        # Collect all strike thresholds including half-integer rounding boundaries
        all_strikes_set: set[float] = set()
        for m in markets:
            floor_raw = m.get("floor_strike")
            cap_raw = m.get("cap_strike")
            if floor_raw is not None:
                f = float(floor_raw)
                all_strikes_set.add(f)
                all_strikes_set.add(f - 0.5)
            if cap_raw is not None:
                c = float(cap_raw)
                all_strikes_set.add(c)
                all_strikes_set.add(c + 0.5)
            extracted = KalshiFetcher.extract_strike_from_market(m)
            if extracted is not None:
                all_strikes_set.add(extracted)
        all_strikes = sorted(all_strikes_set)

        diag.append(f"T0={state.kalman_temp_estimate:.1f}°F, floor={hard_floor:.1f}°F, "
                    f"sigma={state.sigma_volatility:.3f}, theta={state.theta_decay:.4f}, "
                    f"hour_et={hour_et}, strikes={all_strikes}")

        params = build_mc_params(target_date, state, None, market_row, nwp_curve)
        mc_result = price_full_distribution(params, all_strikes, target_date)
        cumulative_probs = mc_result.probabilities

        diag.append(f"MC ok — p10={mc_result.percentile_10:.1f}°F, "
                    f"p50={mc_result.percentile_50:.1f}°F, "
                    f"p90={mc_result.percentile_90:.1f}°F, "
                    f"mean={mc_result.mean_max:.1f}°F")

        prob_by_ticker, prob_sum_raw, gaps = compute_normalized_market_probs(
            markets, cumulative_probs
        )
        diag.append(f"Partition sum (pre-norm): {prob_sum_raw:.4f}, gaps: {len(gaps)}")

        for m in sorted(markets, key=lambda x: KalshiFetcher.extract_strike_from_market(x) or 0):
            strike = KalshiFetcher.extract_strike_from_market(m)
            if strike is None:
                continue
            model_p = prob_by_ticker.get(m.get("ticker", ""), 0.5)
            yes_bid = m.get("yes_bid") or 0
            yes_ask = m.get("yes_ask") or 0

            if yes_ask > 0:
                ask_dec = yes_ask / 100.0
                edge_val = round(model_p - ask_dec, 4)
                action = "BUY YES" if edge_val > edge_threshold else "—"
                entry_ask_dec = ask_dec
            elif yes_bid > 0:
                no_ask_dec = (100 - yes_bid) / 100.0
                edge_val = round((1.0 - model_p) - no_ask_dec, 4)
                action = "BUY NO" if edge_val > edge_threshold else "—"
                entry_ask_dec = no_ask_dec
            else:
                edge_val = 0.0
                action = "NO LIQUIDITY"
                entry_ask_dec = 0.0

            edge_rows.append(EdgeRow(
                ticker=m["ticker"],
                strike_label=KalshiFetcher.get_strike_label(m),
                kalshi_strike=strike,
                action=action,
                yes_bid=yes_bid,
                yes_ask=yes_ask,
                model_prob=round(model_p, 4),
                edge=edge_val,
                ask_decimal=entry_ask_dec,
            ))

    except Exception as exc:
        import traceback
        diag.append(f"compute_edge_table EXCEPTION: {exc}")
        diag.append(traceback.format_exc())
        logger.error("edge_computer.failed", error=str(exc), exc_info=True)

    return edge_rows, diag, prob_sum_raw
