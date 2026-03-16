"""
Streamlit command center for the Kalshi weather trading system.

Four tabs:
  Tab 1 — Trading Desk:         Live metrics, kill switch, edge table, recent trades.
  Tab 2 — Visualizer:           Plotly chart with ASOS history, NWP curves, MC band,
                                and hard floor line.
  Tab 3 — Calibration:          Model weights bar chart, drift sliders, force snapshot,
                                recalibrate button, snapshot history table.
  Tab 4 — Model Transparency:   Data freshness panel + Kalman filter state audit trail.

Run as a separate process:
    streamlit run kalshi_weather_trader/ui/app.py

Reads PostgreSQL via db_manager.  Writes only:
  - markets.auto_trade_enabled  (kill switch)
  - system_state manual overrides (guarded by confirmation checkbox)
"""

from __future__ import annotations

import os
import threading
import time
import traceback as _traceback
from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.graph_objects as go
import pytz
import streamlit as st

# -----------------------------------------------------------------------
# Page config must be the first Streamlit call
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="Kalshi Weather Trader",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------------
# Guard: show setup page if required secrets are missing
# -----------------------------------------------------------------------
_REQUIRED = ["DATABASE_URL", "KALSHI_ACCESS_KEY", "KALSHI_PRIVATE_KEY"]
_missing = [k for k in _REQUIRED if not os.environ.get(k)]

if _missing:
    st.title("🌡️ Kalshi Weather Trader — Setup Required")
    st.error(f"Missing required secrets: **{', '.join(_missing)}**")
    st.markdown("""
### How to add Secrets in Replit

1. Look at the left sidebar and click the **lock icon** labelled **Secrets**
2. Add each of the following keys:

| Secret Key | Where to find the value |
|---|---|
| `DATABASE_URL` | Left sidebar → **Database** tab → copy the connection string |
| `KALSHI_ACCESS_KEY` | Your Kalshi API settings page → Key ID |
| `KALSHI_PRIVATE_KEY` | Paste the full PEM block including `-----BEGIN RSA PRIVATE KEY-----` |
| `DRY_RUN` | Set to `true` while testing (no real orders placed) |

3. After adding all secrets, **stop and restart** the Streamlit process in the shell.
""")
    st.stop()

try:
    from kalshi_weather_trader.config.settings import get_target_date, settings  # noqa: E402
    from kalshi_weather_trader.db import db_manager  # noqa: E402
except Exception as _import_err:
    st.title("⚠️ Startup Error")
    st.error(str(_import_err))
    st.code(_traceback.format_exc())
    st.stop()

# -----------------------------------------------------------------------
# Background scheduler — starts once per Streamlit server process.
# State is stored on the `sys` module, which is truly process-global and
# survives Streamlit's per-rerun script re-execution.  module-level globals
# and globals() checks both fail because Streamlit runs the script in a
# fresh namespace on every rerun.
# -----------------------------------------------------------------------
import sys as _sys

if not hasattr(_sys, "_kalshi_scheduler_lock"):
    _sys._kalshi_scheduler_lock = threading.Lock()
    _sys._kalshi_scheduler_started = False


def _maybe_start_scheduler() -> None:
    """Start the APScheduler background scheduler once, if not already running."""
    with _sys._kalshi_scheduler_lock:
        if _sys._kalshi_scheduler_started:
            return
        try:
            from kalshi_weather_trader.scheduler.orchestrator import (
                build_scheduler,
                startup_sequence,
            )

            startup_sequence()
            _sched = build_scheduler()
            _sched.start()
            _sys._kalshi_scheduler_started = True
        except Exception as exc:
            # Non-fatal: dashboard still works; log but don't crash the UI
            import structlog as _structlog
            _structlog.get_logger(__name__).error(
                "app.scheduler.start_failed", error=str(exc)
            )


_maybe_start_scheduler()

_EASTERN = pytz.timezone("America/New_York")


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _now_et_str() -> str:
    """Return the current time in Eastern as a human-readable string."""
    return datetime.now(timezone.utc).astimezone(_EASTERN).strftime("%Y-%m-%d %H:%M ET")


def _metric_or_na(label: str, value, fmt: str = "{:.1f}°F") -> None:
    """Render a Streamlit metric with N/A fallback."""
    if value is None:
        st.metric(label, "N/A")
    else:
        st.metric(label, fmt.format(value))


# -----------------------------------------------------------------------
# Tab 1 — Trading Desk
# -----------------------------------------------------------------------


def render_trading_desk(target_date) -> None:
    """Render the live trading desk tab.

    Args:
        target_date: Active trading date.

    Returns:
        None
    """
    st.header(f"Trading Desk — {target_date} (as of {_now_et_str()})")

    # Live metrics row
    col1, col2, col3, col4, col5 = st.columns(5)

    try:
        asos = db_manager.get_latest_asos_reading()
    except Exception:
        asos = None

    try:
        market = db_manager.get_market(target_date)
    except Exception:
        market = None

    try:
        state = db_manager.get_system_state(target_date)
    except Exception:
        state = None

    with col1:
        _metric_or_na("ASOS Temp (°F)", asos.temperature_f if asos else None)
    with col2:
        _metric_or_na("Max Observed (°F)", market.current_max_observed if market else None)
    with col3:
        _metric_or_na("Kalman Estimate (°F)", state.kalman_temp_estimate if state else None)
    with col4:
        _metric_or_na("Kalman Bias (°F)", state.kalman_bias_estimate if state else None, fmt="{:+.2f}°F")
    with col5:
        _metric_or_na("Sigma (°F/√hr)", state.sigma_volatility if state else None, fmt="{:.3f}")

    st.divider()

    # Kill switch
    col_ks, col_resume = st.columns(2)
    auto_trade = market.auto_trade_enabled if market else True

    with col_ks:
        if auto_trade:
            st.success("Auto-trading: ENABLED")
            if st.button("🛑 HALT TRADING (Kill Switch)", type="primary", use_container_width=True):
                try:
                    db_manager.set_kill_switch(target_date, enabled=False)
                    st.warning("Kill switch activated — trading halted.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to set kill switch: {exc}")
        else:
            st.error("Auto-trading: HALTED")

    with col_resume:
        if not auto_trade:
            if st.button("▶️ Resume Trading", use_container_width=True):
                try:
                    db_manager.set_kill_switch(target_date, enabled=True)
                    st.success("Trading resumed.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to resume trading: {exc}")

    st.divider()

    # Live multi-strike edge table
    col_edge_hdr, col_edge_btn = st.columns([3, 1])
    with col_edge_hdr:
        st.subheader("Edge Table (Live)")
    with col_edge_btn:
        refresh_edge = st.button("Refresh Edge Table", key="refresh_edge_table")

    # Run on first load OR when button is pressed
    if refresh_edge or "edge_table_rows" not in st.session_state:
        import traceback as _tb
        from kalshi_weather_trader.ingestion.kalshi_fetcher import KalshiFetcher
        from kalshi_weather_trader.ingestion.nwp_fetcher import get_nwp_curve
        from kalshi_weather_trader.quant.monte_carlo import MCParams, price_full_distribution

        edge_diag: list[str] = []
        edge_error: str | None = None
        edge_rows: list[dict] = []
        prob_sum_raw_ui: float | None = None
        gaps_ui: list = []

        try:
            fetcher = KalshiFetcher()

            # Step 1 — Kalshi markets
            date_str = target_date.strftime("%y%b%d").upper()
            event_ticker_queried = f"KXHIGHTBOS-{date_str}"
            edge_diag.append(f"Querying event ticker: {event_ticker_queried}")
            markets = fetcher.get_temperature_markets(target_date)
            if markets:
                edge_diag.append(f"Kalshi markets fetched: {len(markets)} — tickers: {[m.get('ticker') for m in markets]}")
                # Show price fields from first market to confirm field names
                first = markets[0]
                edge_diag.append(
                    f"Sample price fields — yes_bid={first.get('yes_bid')} yes_ask={first.get('yes_ask')} "
                    f"yes_bid_dollars={first.get('yes_bid_dollars')} yes_ask_dollars={first.get('yes_ask_dollars')} "
                    f"last_price_dollars={first.get('last_price_dollars')} floor_strike={first.get('floor_strike')}"
                )
            else:
                edge_diag.append("Kalshi markets: NONE returned by any strategy — see Calibration tab diagnostic for raw API response")

            # Step 2 — system state
            state = db_manager.get_system_state(target_date)
            edge_diag.append(f"System state: T={state.kalman_temp_estimate}°F, bias={state.kalman_bias_estimate:.2f}" if state else "System state: NOT IN DB (run ASOS fetch first)")

            # Step 3 — NWP curve
            nwp_curve = get_nwp_curve(target_date)
            edge_diag.append(f"NWP curve: {len(nwp_curve)} hours" if nwp_curve else "NWP curve: EMPTY (fetch NWP models in Calibration tab)")

            if not markets:
                edge_diag.append("Cannot build edge table: no markets.")
            elif not state:
                edge_diag.append("Cannot run MC: no system state. Use fallback temps.")
                # Fallback: show markets with bids/asks but no model prob
                for m in sorted(markets, key=lambda x: KalshiFetcher.extract_strike_from_market(x) or 0):
                    strike = KalshiFetcher.extract_strike_from_market(m)
                    if strike is None:
                        continue
                    yes_bid = m.get("yes_bid") or 0
                    yes_ask = m.get("yes_ask") or 0
                    edge_rows.append({
                        "Range": KalshiFetcher.get_strike_label(m),
                        "Ticker": m["ticker"],
                        "Bid": f"{yes_bid}¢" if yes_bid else "—",
                        "Ask": f"{yes_ask}¢" if yes_ask else "—",
                        "Model P(YES)": "N/A (no state)",
                        "Edge": "N/A",
                        "Signal": "—",
                    })
            else:
                # Step 4 — build MCParams
                from kalshi_weather_trader.quant.monte_carlo import compute_normalized_market_probs

                now_et_ui = datetime.now(timezone.utc).astimezone(_EASTERN)
                hour_et_ui = now_et_ui.hour
                # After 6 PM ET rollover target_date is tomorrow → start from NWS day
                # start (midnight EST = curve index 1 during EDT, index 0 during EST).
                is_future_day_ui = target_date > now_et_ui.date()
                is_dst_ui = bool(now_et_ui.dst())
                hour_offset_ui = (1 if is_dst_ui else 0) if is_future_day_ui else hour_et_ui
                drift_adj_ui = (
                    state.morning_drift_adjustment if hour_et_ui < 12
                    else state.afternoon_drift_adjustment
                )

                mkt = db_manager.get_market(target_date)
                hard_floor = (mkt.current_max_observed if mkt else None) or state.kalman_temp_estimate

                # Fall back to a flat curve at current temp if NWP is missing
                effective_curve = nwp_curve if nwp_curve else [state.kalman_temp_estimate] * 24

                # Collect ALL threshold values including half-integer rounding boundaries.
                # NWS rounds to nearest integer: the settlement boundary between {38,39}
                # and {40,41} is at 39.5°F, not 40.0°F.  Including ±0.5 values ensures
                # the MC CDF is evaluated at the exact rounding boundaries.
                all_strikes_set_ui: set[float] = set()
                for m in markets:
                    floor_raw_ui = m.get("floor_strike")
                    cap_raw_ui = m.get("cap_strike")
                    if floor_raw_ui is not None:
                        f_ui = float(floor_raw_ui)
                        all_strikes_set_ui.add(f_ui)
                        all_strikes_set_ui.add(f_ui - 0.5)
                    if cap_raw_ui is not None:
                        c_ui = float(cap_raw_ui)
                        all_strikes_set_ui.add(c_ui)
                        all_strikes_set_ui.add(c_ui + 0.5)
                    extracted_ui = KalshiFetcher.extract_strike_from_market(m)
                    if extracted_ui is not None:
                        all_strikes_set_ui.add(extracted_ui)
                all_strikes_ui = sorted(all_strikes_set_ui)
                edge_diag.append(f"Strikes for MC (floor+cap+extracted): {all_strikes_ui}")

                # MC input diagnostics
                edge_diag.append("--- MC Inputs ---")
                edge_diag.append(f"T0 (Kalman temp estimate): {state.kalman_temp_estimate:.1f}°F")
                edge_diag.append(f"hard_floor (current_max_observed): {hard_floor:.1f}°F")
                edge_diag.append(f"bias (Kalman bias estimate): {state.kalman_bias_estimate:.2f}°F")
                edge_diag.append(f"sigma (volatility): {state.sigma_volatility:.3f}")
                edge_diag.append(f"theta (mean reversion): {state.theta_decay:.4f}")
                edge_diag.append(f"hour_et: {hour_et_ui}, is_future_day: {is_future_day_ui}, hour_offset: {hour_offset_ui}")
                if effective_curve:
                    edge_diag.append(f"NWP curve: min={min(effective_curve):.1f}°F, max={max(effective_curve):.1f}°F, curve[{hour_offset_ui}]={effective_curve[min(hour_offset_ui, len(effective_curve)-1)]:.1f}°F")
                    edge_diag.append(f"NWP curve (first 8h): {[round(v,1) for v in effective_curve[:8]]}")
                else:
                    edge_diag.append("NWP curve: EMPTY — using flat fallback")
                mc_mean_target = (effective_curve[min(hour_offset_ui, len(effective_curve)-1)] + state.kalman_bias_estimate) if effective_curve else (state.kalman_temp_estimate + state.kalman_bias_estimate)
                edge_diag.append(f"MC mean-reversion target at step 0: {mc_mean_target:.1f}°F (NWP[offset]+bias)")

                params = MCParams(
                    T0=state.kalman_temp_estimate,
                    hard_floor=hard_floor,
                    nwp_curve=effective_curve,
                    bias=state.kalman_bias_estimate,
                    theta=state.theta_decay,
                    sigma=state.sigma_volatility,
                    drift_adj=drift_adj_ui,
                    hour_offset=hour_offset_ui,
                    n_paths=settings.mc_n_paths,
                )
                edge_diag.append(f"day_fraction_remaining: {params.day_fraction_remaining:.3f}")

                mc_result = price_full_distribution(params, all_strikes_ui, target_date)
                cumulative_probs = mc_result.probabilities
                edge_diag.append(f"MC ran OK — {len(cumulative_probs)} cumulative probs computed")
                edge_diag.append(f"MC output: p10={mc_result.percentile_10:.1f}°F, p50={mc_result.percentile_50:.1f}°F, p90={mc_result.percentile_90:.1f}°F, mean={mc_result.mean_max:.1f}°F")
                edge_diag.append(f"MC cumulative probs: { {k: round(v,3) for k,v in sorted(cumulative_probs.items())} }")

                prob_by_ticker_ui, prob_sum_raw_ui, gaps_ui = compute_normalized_market_probs(
                    markets, cumulative_probs
                )
                edge_diag.append(f"Partition sum (pre-normalization): {prob_sum_raw_ui:.4f}")
                edge_diag.append(f"Partition gaps: {[(f'{g[0]:.1f}–{g[1]:.1f}°F', f'{g[2]:.3f}') for g in gaps_ui]}")

                for m in sorted(markets, key=lambda x: KalshiFetcher.extract_strike_from_market(x) or 0):
                    if KalshiFetcher.extract_strike_from_market(m) is None:
                        continue
                    model_p = prob_by_ticker_ui.get(m.get("ticker", ""), 0.5)

                    yes_bid = m.get("yes_bid") or 0
                    yes_ask = m.get("yes_ask") or 0

                    if yes_ask > 0:
                        edge_yes = round(model_p - yes_ask / 100, 3)
                        signal = "BUY YES" if edge_yes > settings.edge_threshold else "—"
                    elif yes_bid > 0:
                        edge_no = round((1 - model_p) - (100 - yes_bid) / 100, 3)
                        edge_yes = -edge_no
                        signal = "BUY NO" if edge_no > settings.edge_threshold else "—"
                    else:
                        edge_yes = None
                        signal = "NO LIQUIDITY"

                    edge_rows.append({
                        "Range": KalshiFetcher.get_strike_label(m),
                        "Ticker": m["ticker"],
                        "Bid": f"{yes_bid}¢" if yes_bid else "—",
                        "Ask": f"{yes_ask}¢" if yes_ask else "—",
                        "Model P(YES)": f"{model_p:.1%}",
                        "Edge": f"{edge_yes:+.3f}" if edge_yes is not None else "N/A",
                        "Signal": signal,
                    })

                # Sum row
                edge_rows.append({
                    "Range": "─────────",
                    "Ticker": "",
                    "Bid": "",
                    "Ask": "",
                    "Model P(YES)": f"Σ = {prob_sum_raw_ui:.1%} (raw)",
                    "Edge": "",
                    "Signal": "✓ OK" if abs(prob_sum_raw_ui - 1.0) < 0.01 else "⚠ GAP",
                })

        except Exception as e:
            edge_error = _tb.format_exc()
            edge_diag.append(f"EXCEPTION: {e}")

        st.session_state["edge_table_rows"] = edge_rows
        st.session_state["edge_table_diag"] = edge_diag
        st.session_state["edge_table_error"] = edge_error
        st.session_state["edge_table_prob_sum"] = prob_sum_raw_ui
        st.session_state["edge_table_gaps"] = gaps_ui

    # Display
    rows = st.session_state.get("edge_table_rows", [])
    diag = st.session_state.get("edge_table_diag", [])
    err = st.session_state.get("edge_table_error")
    _prob_sum = st.session_state.get("edge_table_prob_sum")
    _gaps = st.session_state.get("edge_table_gaps", [])

    if _prob_sum is not None and abs(_prob_sum - 1.0) > 0.01:
        st.warning(
            f"⚠️ Partition sum = {_prob_sum:.1%} (expected 100%). "
            f"Partition has {len(_gaps)} gap(s). "
            f"Probabilities normalized for edge calculation."
        )

    if rows:
        st.dataframe(rows, use_container_width=True)
    else:
        st.info("No rows — see diagnostics below.")

    with st.expander("Diagnostics", expanded=not rows):
        for line in diag:
            st.text(line)
        if err:
            st.error("Traceback:")
            st.code(err)

    st.divider()

    # Recent trades
    st.subheader("Recent Trades")
    try:
        trades = db_manager.get_recent_trades(target_date, limit=10)
        if trades:
            trade_rows = [
                {
                    "Time (ET)": t.executed_at_utc.astimezone(_EASTERN).strftime("%H:%M:%S"),
                    "Action": t.action,
                    "Strike": t.kalshi_strike,
                    "Contracts": t.contracts,
                    "Price": f"{t.price_cents}¢",
                    "Edge": f"{t.edge_at_execution*100:+.1f}%",
                    "Status": t.status,
                    "Dry Run": "✓" if t.dry_run else "✗",
                }
                for t in trades
            ]
            st.dataframe(pd.DataFrame(trade_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No trades logged for today.")
    except Exception as exc:
        st.warning(f"Could not load trade history: {exc}")


# -----------------------------------------------------------------------
# Tab 2 — Visualizer
# -----------------------------------------------------------------------


def render_visualizer(target_date) -> None:
    """Render the Plotly temperature visualisation tab.

    Args:
        target_date: Active trading date.

    Returns:
        None
    """
    st.header(f"Temperature Visualizer — {target_date}")

    try:
        # ASOS history for today
        from kalshi_weather_trader.config.settings import get_trading_day_bounds
        day_start, day_end = get_trading_day_bounds()
        asos_readings = db_manager.get_asos_readings_since(day_start)
        market = db_manager.get_market(target_date)
        nwp_forecasts = db_manager.get_latest_nwp_forecasts(target_date)
        snapshots = db_manager.get_snapshots_for_date(target_date)

        fig = go.Figure()

        # ASOS temperature line (solid blue)
        if asos_readings:
            asos_times = [r.observation_time_utc.astimezone(_EASTERN) for r in asos_readings]
            asos_temps = [r.temperature_f for r in asos_readings]
            fig.add_trace(go.Scatter(
                x=asos_times, y=asos_temps,
                mode="lines", name="ASOS Observed",
                line=dict(color="royalblue", width=2),
            ))

        # NWP model curves (dashed)
        # Anchor NWP curves at midnight ET of target_date (Open-Meteo indexes
        # from midnight local time, not from the NWS window start).
        from datetime import datetime as _dt
        nwp_anchor = _EASTERN.localize(_dt(target_date.year, target_date.month, target_date.day, 0, 0))
        colors = {"HRRR": "green", "GFS": "orange", "ECMWF": "purple"}
        for model_name, forecast in nwp_forecasts.items():
            if forecast.hourly_temps:
                hours = [
                    (nwp_anchor + pd.Timedelta(hours=i)).astimezone(_EASTERN)
                    for i in range(len(forecast.hourly_temps))
                ]
                fig.add_trace(go.Scatter(
                    x=hours, y=forecast.hourly_temps,
                    mode="lines", name=f"{model_name} Forecast",
                    line=dict(color=colors.get(model_name, "grey"), width=1.5, dash="dash"),
                ))

        # Blended forecast curve — weighted average of available NWP hourly curves
        state = db_manager.get_system_state(target_date)
        model_weights = state.model_weights if state else {"HRRR": 0.5, "GFS": 0.3, "ECMWF": 0.2}
        blended_hourly: list[float] | None = None
        if nwp_forecasts:
            # Find the shortest available curve length to stay aligned
            curve_len = min(
                len(f.hourly_temps) for f in nwp_forecasts.values() if f.hourly_temps
            )
            if curve_len > 0:
                total_weight = sum(
                    model_weights.get(name, 0.0)
                    for name in nwp_forecasts
                    if nwp_forecasts[name].hourly_temps
                )
                if total_weight > 0:
                    blended_hourly = [0.0] * curve_len
                    for name, forecast in nwp_forecasts.items():
                        if not forecast.hourly_temps:
                            continue
                        w = model_weights.get(name, 0.0) / total_weight
                        for i in range(curve_len):
                            blended_hourly[i] += w * forecast.hourly_temps[i]

        if blended_hourly:
            blend_hours = [
                (nwp_anchor + pd.Timedelta(hours=i)).astimezone(_EASTERN)
                for i in range(len(blended_hourly))
            ]
            fig.add_trace(go.Scatter(
                x=blend_hours, y=blended_hourly,
                mode="lines", name="Blended Forecast",
                line=dict(color="darkorange", width=2.5),
            ))

        # Hard floor horizontal line
        if market and market.current_max_observed > -999:
            fig.add_hline(
                y=market.current_max_observed,
                line_dash="dot",
                line_color="red",
                annotation_text=f"Hard Floor: {market.current_max_observed}°F",
                annotation_position="bottom right",
            )

        # NOW vertical line
        now_et = datetime.now(timezone.utc).astimezone(_EASTERN)
        fig.add_vline(
            x=now_et.timestamp() * 1000,
            line_dash="dash",
            line_color="grey",
            annotation_text="NOW",
        )

        fig.update_layout(
            title=f"KBOS Temperature — {target_date}",
            xaxis_title="Time (Eastern)",
            yaxis_title="Temperature (°F)",
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

        # NWP model status table — helps identify which models are in DB vs missing
        with st.expander("NWP Model Status", expanded=not nwp_forecasts):
            model_rows = []
            for name in ["HRRR", "GFS", "ECMWF"]:
                f = nwp_forecasts.get(name)
                w = model_weights.get(name, 0.0)
                if f:
                    model_rows.append({
                        "Model": name,
                        "Status": "In DB",
                        "Predicted High": f"{f.predicted_daily_high}°F",
                        "Hours": len(f.hourly_temps),
                        "Weight": f"{w:.0%}",
                        "Color": colors.get(name, "grey"),
                    })
                else:
                    model_rows.append({
                        "Model": name,
                        "Status": "MISSING — fetch in Calibration tab",
                        "Predicted High": "—",
                        "Hours": 0,
                        "Weight": f"{w:.0%}",
                        "Color": colors.get(name, "grey"),
                    })
            st.dataframe(model_rows, use_container_width=True)


    except Exception as exc:
        st.error(f"Visualizer error: {exc}")
        import traceback
        st.code(traceback.format_exc())


# -----------------------------------------------------------------------
# Tab 3 — Calibration
# -----------------------------------------------------------------------


def render_calibration(target_date) -> None:
    """Render the calibration management tab.

    Args:
        target_date: Active trading date.

    Returns:
        None
    """
    st.header("Calibration & Model Management")

    try:
        state = db_manager.get_system_state(target_date)
    except Exception:
        state = None

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Weights")
        if state:
            weights = state.model_weights
            fig = go.Figure(go.Bar(
                x=list(weights.keys()),
                y=list(weights.values()),
                marker_color=["#1f77b4", "#ff7f0e", "#2ca02c"],
            ))
            fig.update_layout(
                title="NWP Model Weights",
                yaxis=dict(range=[0, 1]),
                height=300,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No system state available.")

    with col2:
        st.subheader("Model Parameters")
        if state:
            st.metric("Theta (mean-reversion)", f"{state.theta_decay:.4f}/hr")
            st.metric("Sigma (volatility)", f"{state.sigma_volatility:.3f} °F/√hr")
            st.metric("Morning Drift Adj.", f"{state.morning_drift_adjustment:+.3f}°F")
            st.metric("Afternoon Drift Adj.", f"{state.afternoon_drift_adjustment:+.3f}°F")
            if state.last_calibrated_utc:
                cal_et = state.last_calibrated_utc.astimezone(_EASTERN).strftime("%Y-%m-%d %H:%M ET")
                st.caption(f"Last calibrated: {cal_et}")

    st.divider()

    # Manual override section
    st.subheader("Manual Overrides")
    st.warning(
        "Manual overrides directly modify system_state.  "
        "Confirm below before applying."
    )

    confirmed = st.checkbox("I understand this will overwrite calibrated values")
    if confirmed and state:
        new_theta = st.slider(
            "Theta (mean-reversion speed)",
            min_value=0.01, max_value=2.0,
            value=float(state.theta_decay), step=0.01,
        )
        new_sigma = st.slider(
            "Sigma (volatility °F/√hr)",
            min_value=0.1, max_value=10.0,
            value=float(state.sigma_volatility), step=0.1,
        )
        new_morning_drift = st.slider(
            "Morning Drift Adjustment (°F)",
            min_value=-5.0, max_value=5.0,
            value=float(state.morning_drift_adjustment), step=0.1,
        )
        new_afternoon_drift = st.slider(
            "Afternoon Drift Adjustment (°F)",
            min_value=-5.0, max_value=5.0,
            value=float(state.afternoon_drift_adjustment), step=0.1,
        )

        if st.button("Apply Manual Overrides", type="primary"):
            try:
                state.theta_decay = new_theta
                state.sigma_volatility = new_sigma
                state.morning_drift_adjustment = new_morning_drift
                state.afternoon_drift_adjustment = new_afternoon_drift
                state.last_updated_utc = datetime.now(timezone.utc)
                db_manager.upsert_system_state(state)
                st.success("Overrides applied.")
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to apply overrides: {exc}")

    st.divider()

    # Kalshi API diagnostic
    st.subheader("Kalshi API Diagnostic")
    if st.button("🔍 Test Kalshi Connection", use_container_width=True):
        with st.spinner("Calling Kalshi API..."):
            try:
                import httpx
                from kalshi_weather_trader.ingestion.kalshi_fetcher import get_kalshi_fetcher
                fetcher = get_kalshi_fetcher()
                st.success("Key loaded OK — RSA key parsed successfully.")

                # Try balance endpoint with both signing path formats to identify correct one
                st.write("**Testing balance with different signing paths:**")
                import time as _time
                _balance_path = "/portfolio/balance"
                _balance_url = fetcher._base_url + _balance_path
                for sign_prefix, label in [
                    (fetcher._base_path, f"with prefix ({fetcher._base_path})"),
                    ("", "without prefix"),
                ]:
                    _ts = str(int(_time.time() * 1000))
                    _hdrs = fetcher._get_auth_headers("GET", sign_prefix + _balance_path)
                    with httpx.Client(timeout=httpx.Timeout(15.0)) as _c:
                        _r = _c.get(_balance_url, headers=_hdrs)
                    if _r.status_code == 200:
                        st.success(f"Balance {label} → HTTP {_r.status_code}: {_r.text[:300]}")
                    else:
                        st.warning(f"Balance {label} → HTTP {_r.status_code}: {_r.text[:300]}")

                # Show config
                from kalshi_weather_trader.config.settings import get_target_date
                td = get_target_date()
                from kalshi_weather_trader.config.settings import settings as _s
                st.write(f"API base URL: `{_s.kalshi_api_base_url}`  |  env: `{_s.kalshi_env}`")
                st.write(f"Access key being sent: `{fetcher._access_key[:8]}...`")

                date_str = td.strftime("%y%b%d").upper()
                event_ticker = f"KXHIGHTBOS-{date_str}"
                st.write(f"Target date: **{td}**  |  event ticker: **{event_ticker}**")

                # 1. GET /events/KXHIGHTBOS-26MAR15 directly
                st.divider()
                st.write(f"**GET /events/{event_ticker} (direct event lookup):**")
                try:
                    _path = f"/events/{event_ticker}"
                    _hdrs = fetcher._get_auth_headers("GET", fetcher._base_path + _path)
                    with httpx.Client(timeout=httpx.Timeout(15.0)) as _c:
                        _r = _c.get(fetcher._base_url + _path, headers=_hdrs)
                    st.write(f"HTTP {_r.status_code}")
                    st.code(_r.text[:800])
                except Exception as exc:
                    st.warning(f"Event lookup failed: {exc}")

                # 2. GET /markets?event_ticker=... without status filter
                st.divider()
                st.write(f"**GET /markets?event_ticker={event_ticker} (no status filter):**")
                try:
                    _hdrs = fetcher._get_auth_headers("GET", fetcher._base_path + "/markets")
                    with httpx.Client(timeout=httpx.Timeout(15.0)) as _c:
                        _r = _c.get(
                            fetcher._base_url + "/markets",
                            params={"event_ticker": event_ticker, "limit": 20},
                            headers=_hdrs,
                        )
                    st.write(f"HTTP {_r.status_code}")
                    data = _r.json()
                    markets = data.get("markets", [])
                    if markets:
                        st.success(f"Found {len(markets)} market(s)!")
                        for m in markets:
                            st.code(f"ticker={m.get('ticker')}  status={m.get('status')}  yes_bid={m.get('yes_bid')}  yes_ask={m.get('yes_ask')}")
                    else:
                        st.warning("Still no markets found.")
                        st.code(_r.text[:500])
                except Exception as exc:
                    st.warning(f"Market search failed: {exc}")

                # 3. GET /events?series_ticker=KXHIGHTBOS (no status filter)
                st.divider()
                st.write("**GET /events?series_ticker=KXHIGHTBOS (no status filter, shows all event tickers):**")
                try:
                    _hdrs = fetcher._get_auth_headers("GET", fetcher._base_path + "/events")
                    with httpx.Client(timeout=httpx.Timeout(15.0)) as _c:
                        _r = _c.get(
                            fetcher._base_url + "/events",
                            params={"series_ticker": "KXHIGHTBOS", "limit": 5},
                            headers=_hdrs,
                        )
                    st.write(f"HTTP {_r.status_code}")
                    st.code(_r.text[:1000])
                except Exception as exc:
                    st.warning(f"Events search failed: {exc}")

                # 4. GET /events/{event_ticker}/markets (nested resource pattern)
                st.divider()
                st.write(f"**GET /events/{event_ticker}/markets (nested markets endpoint):**")
                try:
                    _path = f"/events/{event_ticker}/markets"
                    _hdrs = fetcher._get_auth_headers("GET", fetcher._base_path + _path)
                    with httpx.Client(timeout=httpx.Timeout(15.0)) as _c:
                        _r = _c.get(
                            fetcher._base_url + _path,
                            params={"limit": 20},
                            headers=_hdrs,
                        )
                    st.write(f"HTTP {_r.status_code}")
                    st.code(_r.text[:2000])
                except Exception as exc:
                    st.warning(f"Nested markets lookup failed: {exc}")

                # 5. GET /markets?series_ticker=KXHIGHTBOS&status=active (wide search)
                st.divider()
                st.write("**GET /markets?series_ticker=KXHIGHTBOS&status=active (all active KBOS markets):**")
                try:
                    _hdrs = fetcher._get_auth_headers("GET", fetcher._base_path + "/markets")
                    with httpx.Client(timeout=httpx.Timeout(15.0)) as _c:
                        _r = _c.get(
                            fetcher._base_url + "/markets",
                            params={"series_ticker": "KXHIGHTBOS", "status": "active", "limit": 20},
                            headers=_hdrs,
                        )
                    st.write(f"HTTP {_r.status_code}")
                    st.code(_r.text[:2000])
                except Exception as exc:
                    st.warning(f"Wide market search failed: {exc}")
            except Exception as exc:
                st.error(f"Key load failed: {exc}")

    st.divider()

    # Action buttons
    st.subheader("Fetch NWP Now")
    if st.button("🌤️ Fetch All NWP Models", use_container_width=True):
        with st.spinner("Fetching HRRR, GFS, ECMWF from Open-Meteo..."):
            try:
                from kalshi_weather_trader.ingestion.nwp_fetcher import fetch_all_models
                results = fetch_all_models(target_date)
                # Store results before rerun — st.rerun() would discard any
                # st.success/st.error calls made in this same script pass.
                st.session_state["nwp_fetch_results"] = {
                    name: {
                        "high": doc.predicted_daily_high,
                        "hours": len(doc.hourly_temps),
                    }
                    for name, doc in results.items()
                }
                st.session_state["nwp_fetch_attempted"] = ["HRRR", "GFS", "ECMWF"]
            except Exception as exc:
                import traceback as _tb
                st.session_state["nwp_fetch_results"] = {}
                st.session_state["nwp_fetch_error"] = _tb.format_exc()
                st.session_state["nwp_fetch_attempted"] = ["HRRR", "GFS", "ECMWF"]
        st.rerun()

    # Display NWP fetch results from previous run (persisted across rerun)
    if "nwp_fetch_attempted" in st.session_state:
        results_map = st.session_state.get("nwp_fetch_results", {})
        for model_name in st.session_state["nwp_fetch_attempted"]:
            info = results_map.get(model_name)
            if info is not None:
                st.success(f"{model_name}: {info['high']}°F predicted high, {info['hours']} hrs of data fetched")
            else:
                st.error(f"{model_name}: FAILED — not returned by Open-Meteo (check logs for nwp.fetch.candidate_failed)")
        if "nwp_fetch_error" in st.session_state:
            st.error("Fetch exception:")
            st.code(st.session_state["nwp_fetch_error"])

    col_snap, col_cal = st.columns(2)
    with col_snap:
        st.subheader("Force Snapshot")
        if st.button("📸 Take Snapshot Now", use_container_width=True):
            with st.spinner("Running Monte Carlo and recording snapshot..."):
                try:
                    from kalshi_weather_trader.calibration.calibrator import record_snapshot
                    record_snapshot(target_date, is_forced=True)
                    st.success("Snapshot recorded.")
                except Exception as exc:
                    st.error(f"Snapshot failed: {exc}")

    with col_cal:
        st.subheader("Recalibrate All")
        if st.button("🔁 Run Full Calibration", use_container_width=True):
            with st.spinner("Running all 4 calibration routines..."):
                try:
                    from kalshi_weather_trader.calibration.calibrator import run_full_calibration
                    run_full_calibration(target_date)
                    st.success("Calibration complete.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Calibration failed: {exc}")

    st.divider()

    # Snapshot history table
    st.subheader("Intraday Snapshot History")
    try:
        snapshots = db_manager.get_snapshots_for_date(target_date)
        if snapshots:
            rows = [
                {
                    "Time (ET)": s.snapshot_time_eastern,
                    "ASOS (°F)": s.current_asos_temp_f,
                    "Max Obs. (°F)": s.current_max_observed_f,
                    "Blended (°F)": s.blended_predicted_high,
                    "Kalman T (°F)": s.kalman_temp_estimate,
                    "Bias (°F)": s.kalman_bias_estimate,
                    "Model P(YES)": f"{(s.model_fair_value_prob or 0)*100:.1f}%" if s.model_fair_value_prob else "N/A",
                    "Edge": f"{(s.model_edge or 0)*100:+.1f}%" if s.model_edge else "N/A",
                    "Forced": "✓" if s.is_forced else "",
                }
                for s in reversed(snapshots)
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No snapshots for today yet.")
    except Exception as exc:
        st.warning(f"Could not load snapshots: {exc}")


# -----------------------------------------------------------------------
# Tab 4 — Model Transparency
# -----------------------------------------------------------------------


def _staleness_color(minutes: float, green_thresh: int, yellow_thresh: int) -> str:
    """Return a CSS color name based on data staleness.

    Args:
        minutes:       Minutes since the data was last updated.
        green_thresh:  Minutes threshold below which data is considered fresh (green).
        yellow_thresh: Minutes threshold below which data is considered stale (orange).

    Returns:
        CSS color string: 'green', 'orange', or 'red'.
    """
    if minutes < green_thresh:
        return "green"
    if minutes < yellow_thresh:
        return "orange"
    return "red"


def _colored_label(label: str, color: str) -> None:
    """Render a colored bullet + bold label using markdown with HTML.

    Args:
        label: Text label to display.
        color: CSS color string.

    Returns:
        None
    """
    st.markdown(
        f'<span style="color:{color}; font-size:1.1em;">●</span> <b>{label}</b>',
        unsafe_allow_html=True,
    )


def render_model_transparency(target_date) -> None:
    """Render the Model Transparency audit-trail tab.

    Phase 1 covers:
      - Data Freshness Panel (always-visible, 4 sources)
      - Stage 1: Kalman Filter State (expandable)

    Phases 2-6 will be appended as additional st.expander() blocks.

    Args:
        target_date: Active trading date.

    Returns:
        None
    """
    st.header("Model Transparency")
    st.caption("Audit trail for every stage of the probability-estimation pipeline.")

    # -----------------------------------------------------------------------
    # Data Freshness Panel
    # -----------------------------------------------------------------------
    col_refresh, _ = st.columns([1, 5])
    with col_refresh:
        if st.button("🔄 Refresh Data", key="transparency_refresh"):
            st.rerun()

    now_utc = datetime.now(timezone.utc)

    c1, c2, c3, c4 = st.columns(4)

    # --- Col 1: ASOS ---
    with c1:
        st.subheader("ASOS")
        try:
            asos = db_manager.get_latest_asos_reading()
        except Exception:
            asos = None

        if asos is None:
            st.markdown('<span style="color:gray">No data</span>', unsafe_allow_html=True)
        else:
            age_min = (now_utc - asos.observation_time_utc).total_seconds() / 60
            color = _staleness_color(age_min, 10, 20)
            obs_et = asos.observation_time_utc.astimezone(_EASTERN)
            _colored_label(f"{asos.temperature_f:.1f}°F", color)
            st.caption(f"{obs_et.strftime('%H:%M ET')}  ({age_min:.0f} min ago)")
            # Source hint from raw_metar prefix
            if asos.raw_metar:
                source = "NWS" if asos.raw_metar.startswith("METAR") else "IEM"
                st.caption(f"Source: {source}")

    # --- Col 2: NWP ---
    with c2:
        st.subheader("NWP Models")
        try:
            nwp_forecasts = db_manager.get_latest_nwp_forecasts(target_date)
        except Exception:
            nwp_forecasts = {}

        if not nwp_forecasts:
            st.markdown('<span style="color:gray">No data</span>', unsafe_allow_html=True)
        else:
            for model_name in ["HRRR", "GFS", "ECMWF"]:
                f = nwp_forecasts.get(model_name)
                if f is None:
                    st.caption(f"{model_name}: missing")
                    continue
                age_hr = (now_utc - f.fetched_at_utc).total_seconds() / 3600
                color = _staleness_color(age_hr * 60, 120, 360)  # 2hr / 6hr thresholds in minutes
                fetched_et = f.fetched_at_utc.astimezone(_EASTERN)
                _colored_label(f"{model_name}: {f.predicted_daily_high:.1f}°F", color)
                st.caption(f"{fetched_et.strftime('%H:%M ET')}  ({age_hr:.1f} hr ago)")

    # --- Col 3: Kalshi ---
    with c3:
        st.subheader("Kalshi")
        try:
            market = db_manager.get_market(target_date)
        except Exception:
            market = None

        if market is None:
            st.markdown('<span style="color:gray">No data</span>', unsafe_allow_html=True)
        else:
            age_min = (now_utc - market.last_updated_utc).total_seconds() / 60
            color = _staleness_color(age_min, 5, 15)
            updated_et = market.last_updated_utc.astimezone(_EASTERN)
            _colored_label(f"Last update: {updated_et.strftime('%H:%M ET')}", color)
            st.caption(f"{age_min:.0f} min ago")
            st.caption(f"Status: {market.market_status}")

    # --- Col 4: System State ---
    with c4:
        st.subheader("Kalman State")
        try:
            state = db_manager.get_system_state(target_date)
        except Exception:
            state = None

        if state is None:
            st.markdown('<span style="color:gray">No data</span>', unsafe_allow_html=True)
        else:
            age_min = (now_utc - state.last_updated_utc).total_seconds() / 60
            color = _staleness_color(age_min, 10, 30)
            updated_et = state.last_updated_utc.astimezone(_EASTERN)
            _colored_label(f"Last update: {updated_et.strftime('%H:%M ET')}", color)
            st.caption(f"{age_min:.0f} min ago")

    st.divider()

    # -----------------------------------------------------------------------
    # Stage 1 — Kalman Filter State
    # -----------------------------------------------------------------------
    with st.expander("Stage 1: Kalman Filter State", expanded=False):
        try:
            asos_latest = db_manager.get_latest_asos_reading()
        except Exception:
            asos_latest = None

        try:
            state = db_manager.get_system_state(target_date)
        except Exception:
            state = None

        try:
            nwp_forecasts = db_manager.get_latest_nwp_forecasts(target_date)
        except Exception:
            nwp_forecasts = {}

        # Compute blended NWP at current ET hour (nwp_curve is ET-indexed)
        blended_now: float | None = None
        if nwp_forecasts and state:
            now_hour_et = datetime.now(timezone.utc).astimezone(_EASTERN).hour
            model_weights = state.model_weights
            total_w = sum(
                model_weights.get(n, 0.0)
                for n in nwp_forecasts
                if nwp_forecasts[n].hourly_temps and now_hour_et < len(nwp_forecasts[n].hourly_temps)
            )
            if total_w > 0:
                blended_now = 0.0
                for name, f in nwp_forecasts.items():
                    if not f.hourly_temps or now_hour_et >= len(f.hourly_temps):
                        continue
                    w = model_weights.get(name, 0.0) / total_w
                    blended_now += w * f.hourly_temps[now_hour_et]

        left_col, right_col = st.columns([1, 2])

        with left_col:
            # Row A — three metrics
            ma1, ma2, ma3 = st.columns(3)
            with ma1:
                if asos_latest:
                    st.metric("Raw ASOS Temp", f"{asos_latest.temperature_f:.1f}°F")
                else:
                    st.metric("Raw ASOS Temp", "N/A")

            with ma2:
                if state:
                    asos_val = asos_latest.temperature_f if asos_latest else None
                    diverged = asos_val is not None and abs(state.kalman_temp_estimate - asos_val) > 3.0
                    label = "⚠️ Kalman Estimate" if diverged else "Kalman Estimate"
                    st.metric(label, f"{state.kalman_temp_estimate:.1f}°F")
                else:
                    st.metric("Kalman Estimate", "N/A")

            with ma3:
                if blended_now is not None:
                    st.metric("NWP Blended Now", f"{blended_now:.1f}°F")
                else:
                    st.metric("NWP Blended Now", "N/A")

            # Row B — two metrics
            mb1, mb2 = st.columns(2)
            with mb1:
                if state:
                    bias_warn = abs(state.kalman_bias_estimate) > 5
                    bias_label = "⚠️ Kalman Bias" if bias_warn else "Kalman Bias"
                    st.metric(bias_label, f"{state.kalman_bias_estimate:+.2f}°F")
                else:
                    st.metric("Kalman Bias", "N/A")

            with mb2:
                if state and state.kalman_covariance:
                    try:
                        cov = state.kalman_covariance
                        temp_var = float(cov[0][0])
                        st.metric("Temp Variance P[0,0]", f"{temp_var:.4f}")
                    except (IndexError, TypeError, ValueError):
                        st.metric("Temp Variance P[0,0]", "N/A")
                else:
                    st.metric("Temp Variance P[0,0]", "N/A")

            # Row C — full covariance matrix + Kalman gains + innovation
            if state and state.kalman_covariance:
                try:
                    import numpy as _np_s1
                    _P = _np_s1.array(state.kalman_covariance, dtype=float)
                    _R_s1 = settings.kalman_r_obs
                    _S_s1 = float(_P[0, 0]) + _R_s1  # H P H^T + R (scalar)
                    _K_T = float(_P[0, 0]) / _S_s1
                    _K_B = float(_P[1, 0]) / _S_s1

                    st.markdown("**Covariance Matrix P (2×2)**")
                    cov_table = {
                        "": ["T (row 0)", "B (row 1)"],
                        "T (col 0)": [f"{float(_P[0,0]):.4f} — Temp variance", f"{float(_P[1,0]):.4f} — T–B cross-cov"],
                        "B (col 1)": [f"{float(_P[0,1]):.4f} — T–B cross-cov", f"{float(_P[1,1]):.4f} — Bias variance"],
                    }
                    st.dataframe(cov_table, use_container_width=True, hide_index=True)
                    st.caption(
                        "P[0,1] near zero means filter is not yet correcting NWP bias. "
                        "P[1,1] is the filter's uncertainty about the bias term."
                    )

                    mc1_s1, mc2_s1, mc3_s1 = st.columns(3)
                    with mc1_s1:
                        st.metric(
                            "K_T (temp gain)",
                            f"{_K_T:.4f}",
                            help="Fraction of ASOS innovation applied to temperature estimate. ~0.77 at cold start.",
                        )
                    with mc2_s1:
                        kt_delta = "⚠ frozen" if abs(_K_B) < 0.001 else None
                        st.metric(
                            "K_B (bias gain)",
                            f"{_K_B:.4f}",
                            delta=kt_delta,
                            help="Fraction of ASOS innovation applied to bias estimate. Near 0 at cold start, grows as P[1,0] builds.",
                        )
                    with mc3_s1:
                        if asos_latest and state:
                            _innov = asos_latest.temperature_f - state.kalman_temp_estimate
                            _innov_warn = abs(_innov) > 2.0
                            st.metric(
                                "⚠ Last Innovation" if _innov_warn else "Last Innovation",
                                f"{_innov:+.2f}°F",
                                help="ASOS − Kalman estimate. Should be < ±2°F in normal operation.",
                            )
                        else:
                            st.metric("Last Innovation", "N/A")

                    st.caption("K_B near zero = bias frozen (cold start). Converges toward ~0.1–0.3 after many cycles.")
                except Exception:
                    pass

            st.markdown(
                """
**Update rule** (every 5 min from ASOS):
```
innovation = ASOS − T_est
T_est     += K_T × innovation
Bias      += K_B × innovation
```
**Predict rule** (every hour from NWP delta):
```
T_est += NWP_delta
P     += Q          (process noise: Q_temp=0.1, Q_bias=0.05)
```
"""
            )

        with right_col:
            # Fetch recent data for charts
            recent_asos = db_manager.get_recent_asos_readings_by_hours(hours=3)
            recent_snaps = db_manager.get_recent_snapshots_by_hours(target_date, hours=3)

            if len(recent_snaps) < 2:
                st.info(
                    "Not enough data yet — innovations will appear after the first few scheduler cycles."
                )
            else:
                # Chart A: ASOS scatter + Kalman line
                fig_a = go.Figure()

                if recent_asos:
                    asos_times = [
                        r.observation_time_utc.astimezone(_EASTERN) for r in recent_asos
                    ]
                    asos_temps = [r.temperature_f for r in recent_asos]
                    fig_a.add_trace(go.Scatter(
                        x=asos_times,
                        y=asos_temps,
                        mode="markers",
                        name="Raw ASOS",
                        marker=dict(color="royalblue", size=7),
                    ))

                snap_times = [s.snapshot_time_utc.astimezone(_EASTERN) for s in recent_snaps]
                snap_kalman = [s.kalman_temp_estimate for s in recent_snaps]
                fig_a.add_trace(go.Scatter(
                    x=snap_times,
                    y=snap_kalman,
                    mode="lines",
                    name="Kalman Estimate",
                    line=dict(color="orange", width=2),
                ))

                now_et = datetime.now(timezone.utc).astimezone(_EASTERN)
                fig_a.add_vline(
                    x=now_et.timestamp() * 1000,
                    line_dash="dash",
                    line_color="grey",
                    annotation_text="NOW",
                )
                fig_a.update_layout(
                    title="ASOS vs Kalman — Last 3 Hours",
                    xaxis_title="Time (ET)",
                    yaxis_title="Temperature (°F)",
                    height=280,
                    margin=dict(t=40, b=40),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig_a, use_container_width=True)
                st.caption(
                    "Healthy: Kalman line (orange) tracks ASOS dots closely and is slightly smoother. "
                    "Warning: divergence > 3°F or Kalman stuck flat indicates filter has stalled."
                )

                # Chart B: Innovation residual
                innovations = [
                    s.current_asos_temp_f - s.kalman_temp_estimate for s in recent_snaps
                ]
                fig_b = go.Figure()
                fig_b.add_trace(go.Scatter(
                    x=snap_times,
                    y=innovations,
                    mode="lines+markers",
                    name="Innovation (ASOS − Kalman)",
                    line=dict(color="steelblue", width=1.5),
                    marker=dict(size=5),
                ))
                fig_b.add_hline(y=0, line_color="red", line_dash="solid", line_width=1)
                fig_b.update_layout(
                    title="Kalman Innovation Residual",
                    xaxis_title="Time (ET)",
                    yaxis_title="Residual (°F)",
                    height=220,
                    margin=dict(t=40, b=40),
                )
                st.plotly_chart(fig_b, use_container_width=True)
                st.caption(
                    "Healthy: random noise near zero. "
                    "Warning: trending up/down indicates systematic model error not being corrected."
                )

    # -----------------------------------------------------------------------
    # Stage 2 — NWP Forecast Snapshot
    # -----------------------------------------------------------------------
    with st.expander("Stage 2: NWP Forecast Snapshot", expanded=False):
        try:
            from kalshi_weather_trader.config.settings import get_trading_day_bounds

            nwp_forecasts_s2 = db_manager.get_latest_nwp_forecasts(target_date)
            state_s2 = db_manager.get_system_state(target_date)

            if not nwp_forecasts_s2:
                st.info("No NWP data in DB — fetch models in Calibration tab.")
            else:
                now_utc_s2 = datetime.now(timezone.utc)
                model_weights_s2 = state_s2.model_weights if state_s2 else {"HRRR": 0.5, "GFS": 0.3, "ECMWF": 0.2}
                colors_s2 = {"HRRR": "green", "GFS": "orange", "ECMWF": "purple"}

                # Top row — one column per model
                mc1, mc2, mc3 = st.columns(3)
                for col_s2, model_name_s2 in zip([mc1, mc2, mc3], ["HRRR", "GFS", "ECMWF"]):
                    with col_s2:
                        st.subheader(model_name_s2)
                        f_s2 = nwp_forecasts_s2.get(model_name_s2)
                        if f_s2 is None:
                            st.markdown('<span style="color:gray">Missing</span>', unsafe_allow_html=True)
                        else:
                            age_hr_s2 = (now_utc_s2 - f_s2.fetched_at_utc).total_seconds() / 3600
                            color_s2 = _staleness_color(age_hr_s2 * 60, 120, 360)
                            fetched_et_s2 = f_s2.fetched_at_utc.astimezone(_EASTERN)
                            w_s2 = model_weights_s2.get(model_name_s2, 0.0)
                            _colored_label(f"Pred. High: {f_s2.predicted_daily_high:.1f}°F", color_s2)
                            st.caption(f"Fetched: {fetched_et_s2.strftime('%H:%M ET')} ({age_hr_s2:.1f} hr ago)")
                            st.caption(f"Blend weight: {w_s2:.0%}")
                            freshness_s2 = "✓ Fresh" if age_hr_s2 < 2 else "⚠ Stale"
                            st.caption(f'<span style="color:{color_s2}">{freshness_s2}</span>', unsafe_allow_html=True)

                # Plotly chart — 24-hour temperature curves
                # Anchor at midnight ET of target_date (Open-Meteo indexes from midnight local time).
                from datetime import datetime as _dt
                nwp_anchor_s2 = _EASTERN.localize(_dt(target_date.year, target_date.month, target_date.day, 0, 0))
                fig_s2 = go.Figure()

                for model_name_s2, forecast_s2 in nwp_forecasts_s2.items():
                    if forecast_s2.hourly_temps:
                        w_s2 = model_weights_s2.get(model_name_s2, 0.0)
                        hours_s2 = [
                            (nwp_anchor_s2 + pd.Timedelta(hours=i)).astimezone(_EASTERN)
                            for i in range(len(forecast_s2.hourly_temps))
                        ]
                        fig_s2.add_trace(go.Scatter(
                            x=hours_s2, y=forecast_s2.hourly_temps,
                            mode="lines",
                            name=f"{model_name_s2} ({w_s2:.0%})",
                            line=dict(color=colors_s2.get(model_name_s2, "grey"), width=1.5, dash="dash"),
                        ))

                # Blended weighted average
                curve_len_s2 = min(
                    len(f.hourly_temps) for f in nwp_forecasts_s2.values() if f.hourly_temps
                )
                if curve_len_s2 > 0:
                    total_w_s2 = sum(
                        model_weights_s2.get(n, 0.0)
                        for n in nwp_forecasts_s2
                        if nwp_forecasts_s2[n].hourly_temps
                    )
                    if total_w_s2 > 0:
                        blended_s2 = [0.0] * curve_len_s2
                        for name_s2, f_s2b in nwp_forecasts_s2.items():
                            if not f_s2b.hourly_temps:
                                continue
                            w_s2b = model_weights_s2.get(name_s2, 0.0) / total_w_s2
                            for i in range(curve_len_s2):
                                blended_s2[i] += w_s2b * f_s2b.hourly_temps[i]
                        blend_hours_s2 = [
                            (nwp_anchor_s2 + pd.Timedelta(hours=i)).astimezone(_EASTERN)
                            for i in range(curve_len_s2)
                        ]
                        fig_s2.add_trace(go.Scatter(
                            x=blend_hours_s2, y=blended_s2,
                            mode="lines", name="Blended Forecast",
                            line=dict(color="darkorange", width=2.5),
                        ))

                # NOW vertical line
                now_et_s2 = now_utc_s2.astimezone(_EASTERN)
                fig_s2.add_vline(
                    x=now_et_s2.timestamp() * 1000,
                    line_dash="dash", line_color="grey",
                    annotation_text="NOW",
                )
                fig_s2.update_layout(
                    title=f"NWP Forecast Curves — {target_date}",
                    xaxis_title="Time (Eastern)",
                    yaxis_title="Temperature (°F)",
                    height=400,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig_s2, use_container_width=True)

                # Bottom row — drift adjustments, anchor offset, and corrected attractor
                if state_s2:
                    now_hour_et_s2 = now_utc_s2.astimezone(_EASTERN).hour
                    hour_et_s2 = now_hour_et_s2
                    drift_s2 = state_s2.morning_drift_adjustment if hour_et_s2 < 12 else state_s2.afternoon_drift_adjustment

                    # Blended NWP at current ET hour (nwp_curve is ET-indexed)
                    blended_at_now_s2: float | None = None
                    total_w_now_s2 = sum(
                        model_weights_s2.get(n, 0.0)
                        for n in nwp_forecasts_s2
                        if nwp_forecasts_s2[n].hourly_temps and now_hour_et_s2 < len(nwp_forecasts_s2[n].hourly_temps)
                    )
                    if total_w_now_s2 > 0:
                        blended_at_now_s2 = 0.0
                        for name_s2b, f_s2c in nwp_forecasts_s2.items():
                            if not f_s2c.hourly_temps or now_hour_et_s2 >= len(f_s2c.hourly_temps):
                                continue
                            w_n = model_weights_s2.get(name_s2b, 0.0) / total_w_now_s2
                            blended_at_now_s2 += w_n * f_s2c.hourly_temps[now_hour_et_s2]

                    # NWP anchor offset: T0 − NWP[hour_offset] (re-anchors simulation to current observed temp)
                    anchor_offset_s2: float | None = None
                    if blended_at_now_s2 is not None:
                        anchor_offset_s2 = state_s2.kalman_temp_estimate - blended_at_now_s2

                    # Corrected attractor: NWP[h] + anchor_offset + bias + drift
                    attractor_s2: float | None = None
                    if blended_at_now_s2 is not None and anchor_offset_s2 is not None:
                        attractor_s2 = blended_at_now_s2 + anchor_offset_s2 + state_s2.kalman_bias_estimate + drift_s2

                    bd1, bd2, bd3, bd4 = st.columns(4)
                    with bd1:
                        st.metric(
                            "NWP Blended Now",
                            f"{blended_at_now_s2:.1f}°F" if blended_at_now_s2 is not None else "N/A",
                        )
                    with bd2:
                        st.metric(
                            "NWP Anchor Offset",
                            f"{anchor_offset_s2:+.2f}°F" if anchor_offset_s2 is not None else "N/A",
                            help="T₀ − NWP[hour_offset]. Negative when model runs warm vs observation.",
                        )
                    with bd3:
                        st.metric(
                            "Kalman Bias",
                            f"{state_s2.kalman_bias_estimate:+.2f}°F",
                            help="Systematic NWP error absorbed by the Kalman filter over time.",
                        )
                    with bd4:
                        if attractor_s2 is not None:
                            st.metric("The Attractor (μ₀)", f"{attractor_s2:.1f}°F")
                        else:
                            st.metric("The Attractor (μ₀)", "N/A")

                    if attractor_s2 is not None and blended_at_now_s2 is not None and anchor_offset_s2 is not None:
                        st.caption(
                            f"μ₀ = NWP[h] + anchor_offset + bias + drift  "
                            f"= {blended_at_now_s2:.1f} + ({anchor_offset_s2:+.2f}) + ({state_s2.kalman_bias_estimate:+.2f}) + ({drift_s2:+.2f})"
                            f" = **{attractor_s2:.1f}°F**  \n"
                            f"anchor_offset = T₀ − NWP[hour_offset] — re-anchors step-0 attractor to current observed temp."
                        )
                    else:
                        st.caption("The mean-reversion target the Monte Carlo simulates toward right now.")

                    # Drift calibration provenance
                    st.markdown("**Drift Calibration Provenance (7-day rolling window)**")
                    try:
                        from datetime import timedelta as _td_s2
                        _drift_am_all: list[tuple[str, float]] = []  # (date_str, error)
                        _drift_pm_all: list[tuple[str, float]] = []
                        _drift_days_used = 0
                        for _d_s2 in range(1, 8):
                            _pd_s2 = target_date - _td_s2(days=_d_s2)
                            _mkt_s2 = db_manager.get_market(_pd_s2)
                            if _mkt_s2 is None or _mkt_s2.final_official_high is None:
                                continue
                            _snaps_s2 = db_manager.get_snapshots_for_date(_pd_s2)
                            if not _snaps_s2:
                                continue
                            _drift_days_used += 1
                            _oh_s2 = _mkt_s2.final_official_high
                            for _s in _snaps_s2:
                                try:
                                    _h = int(_s.snapshot_time_eastern.split(":")[0])
                                except (ValueError, AttributeError):
                                    continue
                                _err = _s.blended_predicted_high - _oh_s2
                                if _h < 12:
                                    _drift_am_all.append((str(_pd_s2), _err))
                                else:
                                    _drift_pm_all.append((str(_pd_s2), _err))

                        if _drift_days_used < 2:
                            st.warning(
                                f"Only {_drift_days_used} settled day(s) in the past 7 days — "
                                "drift calibration defaulted to 0.0"
                            )
                        else:
                            _am_mean = sum(e for _, e in _drift_am_all) / len(_drift_am_all) if _drift_am_all else 0.0
                            _pm_mean = sum(e for _, e in _drift_pm_all) / len(_drift_pm_all) if _drift_pm_all else 0.0
                            st.success(
                                f"Calibrated from {_drift_days_used} settled days (past 7):  \n"
                                f"AM: {len(_drift_am_all)} snapshots, mean error = {_am_mean:+.2f}°F → adj = {-_am_mean:+.2f}°F  \n"
                                f"PM: {len(_drift_pm_all)} snapshots, mean error = {_pm_mean:+.2f}°F → adj = {-_pm_mean:+.2f}°F"
                            )
                    except Exception as _exc_drift:
                        st.warning(f"Could not compute drift provenance: {_exc_drift}")

        except Exception as exc_s2:
            import traceback as _tb2
            st.error(f"Stage 2 error: {exc_s2}")
            st.code(_tb2.format_exc())

    # -----------------------------------------------------------------------
    # Stage 2.5 — Calibration Audit
    # -----------------------------------------------------------------------
    with st.expander("Stage 2.5: Calibration Audit", expanded=False):
        try:
            state_cal = db_manager.get_system_state(target_date)

            if state_cal is None:
                st.info("No system state — calibration values not yet available.")
            else:
                # Header: last calibration timestamp
                if state_cal.last_calibrated_utc:
                    import datetime as _dt_cal
                    age_cal_hr = (datetime.now(timezone.utc) - state_cal.last_calibrated_utc).total_seconds() / 3600
                    cal_ts_et = state_cal.last_calibrated_utc.astimezone(_EASTERN).strftime("%Y-%m-%d %H:%M ET")
                    if age_cal_hr > 36:
                        st.warning(
                            f"⚠ Calibration is stale ({age_cal_hr:.0f} hr ago, last: {cal_ts_et}) — "
                            "midnight job may have failed."
                        )
                    else:
                        st.info(f"Last calibrated: {cal_ts_et} ({age_cal_hr:.1f} hr ago)")
                else:
                    st.warning("⚠ No calibration timestamp — midnight job has not run yet.")

                # Compute derived values
                _sigma_cal = state_cal.sigma_volatility
                _theta_cal = state_cal.theta_decay
                _ou_stat_std = _sigma_cal / (2 * _theta_cal) ** 0.5 if _theta_cal > 0 else float("nan")
                _weights_cal = state_cal.model_weights
                _default_weights = {"HRRR": 0.5, "GFS": 0.3, "ECMWF": 0.2}
                _weights_are_default = all(
                    abs(_weights_cal.get(k, 0) - v) < 0.001 for k, v in _default_weights.items()
                )

                audit_rows = [
                    {
                        "Parameter": "Sigma (σ) °F/√hr",
                        "Formula & Source": "√(mean(ΔT²/Δt)) over 7-day ASOS 5-min diffs; gaps >30 min excluded",
                        "Current Value": f"{_sigma_cal:.4f}",
                    },
                    {
                        "Parameter": "Theta (θ) /hr",
                        "Formula & Source": "−ln(φ)/1 hr, φ from AR(1) on hourly ASOS sub-samples, 7-day lookback",
                        "Current Value": f"{_theta_cal:.4f}",
                    },
                    {
                        "Parameter": "OU Stationary σ (σ/√2θ)",
                        "Formula & Source": "Expected 1-sigma spread of temperature at equilibrium",
                        "Current Value": f"{_ou_stat_std:.3f}°F",
                    },
                    {
                        "Parameter": "Morning Drift Adj. °F",
                        "Formula & Source": "−mean(blended_pred_high − official_high) for AM snapshots, 7-day rolling window",
                        "Current Value": f"{state_cal.morning_drift_adjustment:+.4f}",
                    },
                    {
                        "Parameter": "Afternoon Drift Adj. °F",
                        "Formula & Source": "−mean(blended_pred_high − official_high) for PM snapshots, 7-day rolling window",
                        "Current Value": f"{state_cal.afternoon_drift_adjustment:+.4f}",
                    },
                    {
                        "Parameter": "HRRR weight",
                        "Formula & Source": "softmax(1/Brier_14d) — 14-day rolling Brier score",
                        "Current Value": f"{_weights_cal.get('HRRR', 0.5):.1%}",
                    },
                    {
                        "Parameter": "GFS weight",
                        "Formula & Source": "softmax(1/Brier_14d) — 14-day rolling Brier score",
                        "Current Value": f"{_weights_cal.get('GFS', 0.3):.1%}",
                    },
                    {
                        "Parameter": "ECMWF weight",
                        "Formula & Source": "softmax(1/Brier_14d) — 14-day rolling Brier score",
                        "Current Value": f"{_weights_cal.get('ECMWF', 0.2):.1%}",
                    },
                ]
                st.dataframe(audit_rows, use_container_width=True, hide_index=True)

                if _weights_are_default:
                    st.caption(
                        "Model weights are fixed at defaults (HRRR 50% / GFS 30% / ECMWF 20%) "
                        "until ≥2 days of settled market data exist for Brier score calibration."
                    )

                st.caption(
                    "Sigma, theta, and model weights are recalibrated once daily at 00:05 ET. "
                    "Kalman state (T_est, bias, P) updates continuously every 5 min."
                )

                st.markdown("---")

                # -----------------------------------------------------------
                # Sub-section A: Yesterday's Settlement & Drift Contribution
                # -----------------------------------------------------------
                with st.expander("📅 Yesterday's Settlement & Drift Contribution", expanded=False):
                    try:
                        _yest_date = target_date - timedelta(days=1)
                        _yest_market = db_manager.get_market(_yest_date)
                        _yest_nwp = db_manager.get_latest_nwp_forecasts(_yest_date)
                        _yest_state = db_manager.get_system_state(_yest_date)

                        if _yest_market is None:
                            st.info(f"No market data for yesterday ({_yest_date}).")
                        elif _yest_market.final_official_high is None:
                            st.warning(
                                f"⚠ Yesterday ({_yest_date}) is not yet settled — "
                                "drift calibration used data from earlier days only."
                            )
                        else:
                            _yest_official = _yest_market.final_official_high
                            # Blended NWP for yesterday
                            _yest_models = {"HRRR": None, "GFS": None, "ECMWF": None}
                            _yest_weights_used = state_cal.model_weights if state_cal else {"HRRR": 0.5, "GFS": 0.3, "ECMWF": 0.2}
                            _yest_blended_num = 0.0
                            _yest_blended_den = 0.0
                            for _m in ["HRRR", "GFS", "ECMWF"]:
                                if _m in _yest_nwp and _yest_nwp[_m].predicted_daily_high is not None:
                                    _yest_models[_m] = _yest_nwp[_m].predicted_daily_high
                                    _w = _yest_weights_used.get(_m, 0.0)
                                    _yest_blended_num += _yest_models[_m] * _w
                                    _yest_blended_den += _w
                            _yest_blended = (_yest_blended_num / _yest_blended_den) if _yest_blended_den > 0 else None
                            _yest_error = (_yest_blended - _yest_official) if _yest_blended is not None else None

                            # Prominent metric row
                            _c1, _c2, _c3, _c4 = st.columns(4)
                            _c1.metric("Date", str(_yest_date))
                            _c2.metric("NWS Official High", f"{_yest_official:.1f}°F")
                            if _yest_blended is not None:
                                _c3.metric("NWP Blended Forecast", f"{_yest_blended:.1f}°F")
                                _c4.metric(
                                    "Forecast Error",
                                    f"{_yest_error:+.2f}°F",
                                    delta=f"{_yest_error:+.2f}°F",
                                    delta_color="inverse",
                                )
                            else:
                                _c3.metric("NWP Blended Forecast", "N/A")
                                _c4.metric("Forecast Error", "N/A")

                            # Per-model table
                            _model_rows = []
                            for _m in ["HRRR", "GFS", "ECMWF"]:
                                _pred = _yest_models.get(_m)
                                _err = (_pred - _yest_official) if _pred is not None else None
                                _model_rows.append({
                                    "Model": _m,
                                    "Predicted High (°F)": f"{_pred:.1f}" if _pred is not None else "—",
                                    "Official High (°F)": f"{_yest_official:.1f}",
                                    "Error (°F)": f"{_err:+.2f}" if _err is not None else "—",
                                })
                            st.dataframe(_model_rows, use_container_width=True, hide_index=True)

                            # Drift contribution callout
                            _today_morn = state_cal.morning_drift_adjustment if state_cal else None
                            _today_aftn = state_cal.afternoon_drift_adjustment if state_cal else None
                            if _today_morn is not None and _today_aftn is not None:
                                st.info(
                                    f"This error was pooled with the prior 6 days' errors to compute "
                                    f"today's drift adjustments:  \n"
                                    f"**Morning:** {_today_morn:+.3f}°F  |  "
                                    f"**Afternoon:** {_today_aftn:+.3f}°F  \n"
                                    f"*(Negative drift = NWP runs warm; positive = NWP runs cold.)*"
                                )
                    except Exception as _exc_yest:
                        import traceback as _tb_yest
                        st.error(f"Yesterday's settlement error: {_exc_yest}")
                        st.code(_tb_yest.format_exc())

                # -----------------------------------------------------------
                # Sub-section B: Sigma & Theta Calibration Data
                # -----------------------------------------------------------
                with st.expander("📐 Sigma & Theta Calibration Data (7-day ASOS)", expanded=False):
                    try:
                        _asos_since = datetime.now(timezone.utc) - timedelta(hours=168)
                        _asos_readings = db_manager.get_asos_readings_since(since_utc=_asos_since)

                        if not _asos_readings or len(_asos_readings) < 3:
                            st.info(
                                "No ASOS readings in local DB for the past 7 days. "
                                "Sigma/theta audit will populate after the system has been running."
                            )
                        else:
                            # --- Sigma breakdown ---
                            _MAX_GAP_HOURS = 0.5
                            _dT_vals: list[float] = []
                            _contributions: list[float] = []
                            _skipped_sigma = 0
                            for _i in range(1, len(_asos_readings)):
                                _dt_hr = (
                                    _asos_readings[_i].observation_time_utc
                                    - _asos_readings[_i - 1].observation_time_utc
                                ).total_seconds() / 3600.0
                                if _dt_hr <= 0 or _dt_hr > _MAX_GAP_HOURS:
                                    _skipped_sigma += 1
                                    continue
                                _dT = _asos_readings[_i].temperature_f - _asos_readings[_i - 1].temperature_f
                                _dT_vals.append(abs(_dT))
                                _contributions.append(_dT ** 2 / _dt_hr)

                            _n_valid = len(_contributions)
                            _sigma_computed = float((_sum := sum(_contributions)) and (_n_valid > 0) and (_sum / _n_valid) ** 0.5) if _n_valid > 0 else None
                            if _n_valid > 0:
                                import math as _math_cal
                                _sigma_computed = _math_cal.sqrt(sum(_contributions) / _n_valid)

                            _date_range_start = _asos_readings[0].observation_time_utc.astimezone(_EASTERN).strftime("%m/%d %H:%M ET")
                            _date_range_end = _asos_readings[-1].observation_time_utc.astimezone(_EASTERN).strftime("%m/%d %H:%M ET")

                            st.markdown("**Sigma (σ) — Temperature Volatility**")
                            _sc1, _sc2, _sc3 = st.columns(3)
                            _sc1.metric("ASOS Readings", len(_asos_readings))
                            _sc2.metric("Valid Intervals", _n_valid)
                            _sc3.metric("Gaps Excluded", _skipped_sigma)
                            st.caption(f"Coverage: {_date_range_start} → {_date_range_end}")

                            if _n_valid > 0:
                                st.caption(
                                    f"σ = √(mean(ΔT²/Δt)) = √({sum(_contributions) / _n_valid:.4f}) "
                                    f"= **{_sigma_computed:.4f}°F/√hr**  "
                                    f"(stored: {state_cal.sigma_volatility:.4f})"
                                )

                                # Histogram of |dT| values
                                import numpy as _np_cal
                                _fig_sigma = go.Figure()
                                _fig_sigma.add_trace(go.Histogram(
                                    x=_dT_vals,
                                    nbinsx=30,
                                    name="|ΔT| per 5-min interval",
                                    marker_color="steelblue",
                                ))
                                _fig_sigma.update_layout(
                                    title="Distribution of |ΔT| across valid 5-min intervals (7-day ASOS)",
                                    xaxis_title="|ΔT| (°F)",
                                    yaxis_title="Count",
                                    height=300,
                                    margin=dict(t=40, b=30),
                                )
                                st.plotly_chart(_fig_sigma, use_container_width=True)

                            st.markdown("---")

                            # --- Theta breakdown ---
                            st.markdown("**Theta (θ) — Mean-Reversion Speed**")
                            import numpy as _np_theta
                            _all_temps = _np_theta.array([r.temperature_f for r in _asos_readings], dtype=float)
                            _hourly_temps = _all_temps[::12]  # every 12th ~5-min reading → hourly

                            if len(_hourly_temps) >= 4:
                                _y_theta = _hourly_temps[1:]
                                _x_theta = _hourly_temps[:-1]
                                _phi = float(_np_theta.cov(_x_theta, _y_theta)[0, 1] / _np_theta.var(_x_theta))
                                _phi_bounded = max(0.01, min(0.99, _phi))
                                _theta_computed = float(-_np_theta.log(_phi_bounded) / 1.0)
                                _theta_computed = max(0.01, min(2.0, _theta_computed))

                                _tc1, _tc2, _tc3 = st.columns(3)
                                _tc1.metric("Hourly Samples (AR input)", len(_hourly_temps))
                                _tc2.metric("φ (lag-1 autocorrelation)", f"{_phi_bounded:.4f}")
                                _tc3.metric("θ = −ln(φ)/1hr", f"{_theta_computed:.4f}")
                                st.caption(
                                    f"θ bounded to [0.01, 2.0].  "
                                    f"Stored value: **{state_cal.theta_decay:.4f}/hr**"
                                )

                                # Line chart: hourly temperature series
                                _hours_ago = list(range(len(_hourly_temps) - 1, -1, -1))
                                _fig_theta = go.Figure()
                                _fig_theta.add_trace(go.Scatter(
                                    x=_hours_ago,
                                    y=list(_hourly_temps),
                                    mode="lines",
                                    name="Hourly ASOS Temp (°F)",
                                    line=dict(color="tomato"),
                                ))
                                _fig_theta.update_layout(
                                    title="Hourly ASOS Temperature — AR(1) Input Series (7-day)",
                                    xaxis_title="Hours ago",
                                    xaxis=dict(autorange="reversed"),
                                    yaxis_title="Temperature (°F)",
                                    height=300,
                                    margin=dict(t=40, b=30),
                                )
                                st.plotly_chart(_fig_theta, use_container_width=True)
                            else:
                                st.info("Insufficient hourly samples for AR(1) fit (need ≥4).")

                            st.caption(
                                "⚠ This view uses DB-stored readings. The calibrator fetches live IEM data, "
                                "which may differ slightly if the app was offline for any period."
                            )
                    except Exception as _exc_sigma:
                        import traceback as _tb_sigma
                        st.error(f"Sigma/Theta audit error: {_exc_sigma}")
                        st.code(_tb_sigma.format_exc())

                # -----------------------------------------------------------
                # Sub-section C: Drift Rolling Window Breakdown
                # -----------------------------------------------------------
                with st.expander("📊 Drift Rolling Window Breakdown (7-day)", expanded=False):
                    try:
                        _drift_rows = []
                        _am_errors_drift: list[float] = []
                        _pm_errors_drift: list[float] = []

                        for _d_idx in range(1, 8):
                            _d_date = target_date - timedelta(days=_d_idx)
                            try:
                                _d_market = db_manager.get_market(_d_date)
                                _d_snaps = db_manager.get_snapshots_for_date(_d_date)
                                _d_official = _d_market.final_official_high if _d_market else None

                                _d_am_snaps = []
                                _d_pm_snaps = []
                                for _s in _d_snaps:
                                    try:
                                        _s_hr = int(_s.snapshot_time_eastern.split(":")[0])
                                    except (ValueError, AttributeError):
                                        continue
                                    if _s_hr < 12:
                                        _d_am_snaps.append(_s.blended_predicted_high)
                                    else:
                                        _d_pm_snaps.append(_s.blended_predicted_high)

                                _d_am_avg = (sum(_d_am_snaps) / len(_d_am_snaps)) if _d_am_snaps else None
                                _d_pm_avg = (sum(_d_pm_snaps) / len(_d_pm_snaps)) if _d_pm_snaps else None
                                _d_am_err = (_d_am_avg - _d_official) if (_d_am_avg is not None and _d_official is not None) else None
                                _d_pm_err = (_d_pm_avg - _d_official) if (_d_pm_avg is not None and _d_official is not None) else None

                                if _d_am_err is not None:
                                    _am_errors_drift.append(_d_am_err)
                                if _d_pm_err is not None:
                                    _pm_errors_drift.append(_d_pm_err)

                                _drift_rows.append({
                                    "Date": str(_d_date),
                                    "Settled?": "✓" if _d_official is not None else "—",
                                    "AM Snaps": len(_d_am_snaps),
                                    "AM Forecast Avg": f"{_d_am_avg:.1f}" if _d_am_avg is not None else "—",
                                    "PM Snaps": len(_d_pm_snaps),
                                    "PM Forecast Avg": f"{_d_pm_avg:.1f}" if _d_pm_avg is not None else "—",
                                    "Official High": f"{_d_official:.1f}" if _d_official is not None else "not settled",
                                    "AM Error": f"{_d_am_err:+.2f}" if _d_am_err is not None else "—",
                                    "PM Error": f"{_d_pm_err:+.2f}" if _d_pm_err is not None else "—",
                                })
                            except Exception:
                                _drift_rows.append({
                                    "Date": str(_d_date),
                                    "Settled?": "err",
                                    "AM Snaps": 0, "AM Forecast Avg": "—",
                                    "PM Snaps": 0, "PM Forecast Avg": "—",
                                    "Official High": "—", "AM Error": "—", "PM Error": "—",
                                })

                        if _drift_rows:
                            st.dataframe(_drift_rows, use_container_width=True, hide_index=True)

                        # Summary row
                        _drift_am_mean = (sum(_am_errors_drift) / len(_am_errors_drift)) if _am_errors_drift else None
                        _drift_pm_mean = (sum(_pm_errors_drift) / len(_pm_errors_drift)) if _pm_errors_drift else None
                        if _drift_am_mean is not None or _drift_pm_mean is not None:
                            st.markdown(
                                f"**Computed adjustments** — "
                                f"Morning: **{-_drift_am_mean:+.3f}°F** "
                                f"(mean AM error = {_drift_am_mean:+.3f}°F)  |  "
                                f"Afternoon: **{-_drift_pm_mean:+.3f}°F** "
                                f"(mean PM error = {_drift_pm_mean:+.3f}°F)"
                                if (_drift_am_mean is not None and _drift_pm_mean is not None) else
                                f"**Computed adjustments** — "
                                f"Morning: **{-_drift_am_mean:+.3f}°F**" if _drift_am_mean is not None else
                                f"**Computed adjustments** — "
                                f"Afternoon: **{-_drift_pm_mean:+.3f}°F**"
                            )
                        else:
                            st.info("No settled days in the past 7 days — drift adjustments defaulting to 0.0°F.")

                        st.caption(
                            "Drift = −mean(forecast − actual).  "
                            "Negative drift = NWP runs warm.  Positive = NWP runs cold."
                        )
                    except Exception as _exc_drift_win:
                        import traceback as _tb_drift_win
                        st.error(f"Drift window breakdown error: {_exc_drift_win}")
                        st.code(_tb_drift_win.format_exc())

        except Exception as exc_cal:
            import traceback as _tb_cal
            st.error(f"Calibration Audit error: {exc_cal}")
            st.code(_tb_cal.format_exc())

    # -----------------------------------------------------------------------
    # Stage 3 — Monte Carlo Inputs
    # -----------------------------------------------------------------------
    with st.expander("Stage 3: Monte Carlo Inputs", expanded=False):
        try:
            state_s3 = db_manager.get_system_state(target_date)
            market_s3 = db_manager.get_market(target_date)
            nwp_forecasts_s3 = db_manager.get_latest_nwp_forecasts(target_date)

            if state_s3 is None:
                st.info("No system state — run ASOS fetch first.")
            else:
                from kalshi_weather_trader.ingestion.nwp_fetcher import get_nwp_curve

                now_utc_s3 = datetime.now(timezone.utc)
                now_et_s3 = now_utc_s3.astimezone(_EASTERN)
                hour_et_s3 = now_et_s3.hour
                is_future_day_s3 = target_date > now_et_s3.date()
                is_dst_s3 = bool(now_et_s3.dst())
                hour_offset_s3 = (1 if is_dst_s3 else 0) if is_future_day_s3 else hour_et_s3

                hard_floor_s3 = (market_s3.current_max_observed if market_s3 else None) or state_s3.kalman_temp_estimate
                nwp_curve_s3 = get_nwp_curve(target_date)
                effective_curve_s3 = nwp_curve_s3 if nwp_curve_s3 else [state_s3.kalman_temp_estimate] * 24
                from kalshi_weather_trader.config.settings import get_remaining_day_fraction
                day_fraction_s3 = get_remaining_day_fraction() if not is_future_day_s3 else 1.0
                n_steps_s3 = int(day_fraction_s3 * 288)

                # Blended NWP at current ET hour (nwp_curve is ET-indexed)
                model_weights_s3 = state_s3.model_weights
                nwp_at_now_s3: float | None = None
                if nwp_forecasts_s3:
                    total_w_s3 = sum(
                        model_weights_s3.get(n, 0.0)
                        for n in nwp_forecasts_s3
                        if nwp_forecasts_s3[n].hourly_temps and hour_et_s3 < len(nwp_forecasts_s3[n].hourly_temps)
                    )
                    if total_w_s3 > 0:
                        nwp_at_now_s3 = 0.0
                        for name_s3, f_s3 in nwp_forecasts_s3.items():
                            if not f_s3.hourly_temps or hour_et_s3 >= len(f_s3.hourly_temps):
                                continue
                            nwp_at_now_s3 += model_weights_s3.get(name_s3, 0.0) / total_w_s3 * f_s3.hourly_temps[hour_et_s3]

                # NWP anchor offset: T0 − NWP[hour_offset]
                nwp_at_offset_s3: float | None = None
                if effective_curve_s3 and hour_offset_s3 < len(effective_curve_s3):
                    nwp_at_offset_s3 = effective_curve_s3[hour_offset_s3]
                nwp_anchor_offset_s3: float | None = None
                if nwp_at_offset_s3 is not None:
                    nwp_anchor_offset_s3 = state_s3.kalman_temp_estimate - nwp_at_offset_s3

                # OU stationary standard deviation
                _ou_std_s3 = state_s3.sigma_volatility / (2 * state_s3.theta_decay) ** 0.5 if state_s3.theta_decay > 0 else float("nan")

                # Drift adjustment
                _hour_et_s3 = datetime.now(timezone.utc).astimezone(_EASTERN).hour
                drift_adj_s3_display = state_s3.morning_drift_adjustment if _hour_et_s3 < 12 else state_s3.afternoon_drift_adjustment

                # Corrected attractor: NWP[h] + anchor_offset + bias + drift
                attractor_s3: float | None = None
                if nwp_at_now_s3 is not None and nwp_anchor_offset_s3 is not None:
                    attractor_s3 = nwp_at_now_s3 + nwp_anchor_offset_s3 + state_s3.kalman_bias_estimate + drift_adj_s3_display

                param_rows_s3 = [
                    {"Parameter": "Hard Floor (current_max_observed)", "Value": f"{hard_floor_s3:.1f}°F"},
                    {"Parameter": "Starting Temp (Kalman T₀)", "Value": f"{state_s3.kalman_temp_estimate:.1f}°F"},
                    {"Parameter": "Kalman Bias (B_t)", "Value": f"{state_s3.kalman_bias_estimate:+.3f}°F"},
                    {"Parameter": "NWP Anchor Offset (T₀ − NWP[h₀])", "Value": f"{nwp_anchor_offset_s3:+.2f}°F" if nwp_anchor_offset_s3 is not None else "N/A"},
                    {"Parameter": "Theta (mean-reversion speed /hr)", "Value": f"{state_s3.theta_decay:.4f}"},
                    {"Parameter": "Sigma (volatility °F/√hr)", "Value": f"{state_s3.sigma_volatility:.3f}"},
                    {"Parameter": "OU Stationary σ (σ/√2θ)", "Value": f"{_ou_std_s3:.3f}°F"},
                    {"Parameter": "Drift Adjustment (AM/PM)", "Value": f"{drift_adj_s3_display:+.3f}°F"},
                    {"Parameter": "Hour Offset (ET)", "Value": str(hour_offset_s3)},
                    {"Parameter": "Remaining Day Fraction", "Value": f"{day_fraction_s3:.3f}"},
                    {"Parameter": "N Steps (5-min intervals)", "Value": str(n_steps_s3)},
                    {"Parameter": "N Paths", "Value": str(settings.mc_n_paths)},
                    {"Parameter": "Step-0 Attractor (μ₀)", "Value": f"{attractor_s3:.1f}°F" if attractor_s3 is not None else "N/A"},
                ]
                st.dataframe(param_rows_s3, use_container_width=True, hide_index=True)
                st.caption(
                    "Step-0 attractor equals T₀ exactly (by anchor offset construction). "
                    "Subsequent steps follow NWP's rate of change, not its absolute level. "
                    "OU Stationary σ = σ/√(2θ) — expected 1-sigma spread at equilibrium."
                )

                # Run simulation button
                if st.button("▶ Run Simulation Now", key="transparency_run_mc"):
                    from kalshi_weather_trader.ingestion.kalshi_fetcher import KalshiFetcher
                    from kalshi_weather_trader.quant.monte_carlo import MCParams, price_full_distribution

                    with st.spinner("Running 10,000-path Monte Carlo simulation..."):
                        try:
                            hour_et_s3b = now_et_s3.hour
                            drift_adj_s3 = state_s3.morning_drift_adjustment if hour_et_s3b < 12 else state_s3.afternoon_drift_adjustment

                            params_s3 = MCParams(
                                T0=state_s3.kalman_temp_estimate,
                                hard_floor=hard_floor_s3,
                                nwp_curve=effective_curve_s3,
                                bias=state_s3.kalman_bias_estimate,
                                theta=state_s3.theta_decay,
                                sigma=state_s3.sigma_volatility,
                                drift_adj=drift_adj_s3,
                                hour_offset=hour_offset_s3,
                                n_paths=settings.mc_n_paths,
                            )

                            fetcher_s3 = KalshiFetcher()
                            markets_s3 = fetcher_s3.get_temperature_markets(target_date)
                            all_strikes_s3: set[float] = set()
                            for m_s3 in markets_s3:
                                fl = m_s3.get("floor_strike")
                                cp = m_s3.get("cap_strike")
                                ex = KalshiFetcher.extract_strike_from_market(m_s3)
                                if fl is not None:
                                    all_strikes_s3.add(float(fl))
                                    all_strikes_s3.add(float(fl) - 0.5)
                                if cp is not None:
                                    all_strikes_s3.add(float(cp))
                                    all_strikes_s3.add(float(cp) + 0.5)
                                if ex is not None:
                                    all_strikes_s3.add(ex)

                            mc_result_s3 = price_full_distribution(params_s3, sorted(all_strikes_s3), target_date)
                            st.session_state["transparency_mc_result"] = mc_result_s3
                            st.session_state["transparency_mc_params"] = params_s3
                            st.success("Simulation complete.")
                        except Exception as exc_run:
                            import traceback as _tb3
                            st.error(f"Simulation failed: {exc_run}")
                            st.code(_tb3.format_exc())

        except Exception as exc_s3:
            import traceback as _tb3b
            st.error(f"Stage 3 error: {exc_s3}")
            st.code(_tb3b.format_exc())

    # -----------------------------------------------------------------------
    # Stage 4 — Simulated Distribution
    # -----------------------------------------------------------------------
    with st.expander("Stage 4: Simulated Distribution", expanded=False):
        mc_result_s4 = st.session_state.get("transparency_mc_result")
        if mc_result_s4 is None:
            st.info("Run the simulation in Stage 3 first to see the distribution.")
        else:
            import numpy as np

            left_s4, right_s4 = st.columns([3, 2])

            with left_s4:
                # Approximate distribution with synthetic normal sample clipped at hard floor
                np.random.seed(42)
                samples_s4 = np.random.normal(mc_result_s4.mean_max, mc_result_s4.std_max, 5000)
                samples_s4 = np.clip(samples_s4, mc_result_s4.hard_floor, None)

                fig_s4 = go.Figure()
                fig_s4.add_trace(go.Histogram(
                    x=samples_s4, nbinsx=60, opacity=0.75,
                    name="Simulated Daily Max",
                    marker_color="steelblue",
                ))

                # Hard floor line
                fig_s4.add_vline(
                    x=mc_result_s4.hard_floor,
                    line_dash="solid", line_color="red",
                    annotation_text=f"Hard Floor: {mc_result_s4.hard_floor:.1f}°F",
                    annotation_position="top right",
                )

                # Strike lines from probabilities
                for strike_s4 in sorted(mc_result_s4.probabilities.keys()):
                    fig_s4.add_vline(
                        x=strike_s4,
                        line_dash="dash", line_color="grey",
                        annotation_text=f"{strike_s4:.1f}°F",
                        annotation_position="top left",
                    )

                fig_s4.update_layout(
                    title="Simulated Daily-Max Distribution",
                    xaxis_title="Daily Max Temperature (°F)",
                    yaxis_title="Count",
                    height=380,
                    showlegend=False,
                )
                st.plotly_chart(fig_s4, use_container_width=True)
                st.caption(
                    "Healthy: bell-shaped, centered above hard floor. "
                    "Warning: spike at hard floor = near-zero remaining day; "
                    "bimodal = drift miscalibration."
                )

            with right_s4:
                # Percentile table
                st.subheader("Percentiles")
                pct_rows_s4 = [
                    {"Percentile": "10th", "Temp (°F)": f"{mc_result_s4.percentile_10:.1f}"},
                    {"Percentile": "25th", "Temp (°F)": f"{mc_result_s4.percentile_25:.1f}"},
                    {"Percentile": "50th (median)", "Temp (°F)": f"{mc_result_s4.percentile_50:.1f}"},
                    {"Percentile": "75th", "Temp (°F)": f"{mc_result_s4.percentile_75:.1f}"},
                    {"Percentile": "90th", "Temp (°F)": f"{mc_result_s4.percentile_90:.1f}"},
                    {"Percentile": "Mean", "Temp (°F)": f"{mc_result_s4.mean_max:.1f}"},
                    {"Percentile": "Std Dev", "Temp (°F)": f"{mc_result_s4.std_max:.2f}"},
                ]
                st.dataframe(pct_rows_s4, use_container_width=True, hide_index=True)

                # Probability table
                st.subheader("Strike Probabilities")
                prob_rows_s4 = [
                    {
                        "Strike (°F)": f"{s:.1f}",
                        "P(max ≥ strike)": f"{p:.3f}",
                        "P(max < strike)": f"{1-p:.3f}",
                    }
                    for s, p in sorted(mc_result_s4.probabilities.items())
                ]
                if prob_rows_s4:
                    st.dataframe(prob_rows_s4, use_container_width=True, hide_index=True)
                else:
                    st.info("No strike probabilities computed.")

    # -----------------------------------------------------------------------
    # Stage 5 — Edge Calculation Breakdown
    # -----------------------------------------------------------------------
    with st.expander("Stage 5: Edge Calculation Breakdown", expanded=False):
        mc_result_s5 = st.session_state.get("transparency_mc_result")
        if mc_result_s5 is None:
            st.info("Run the simulation in Stage 3 first to calculate edges.")
        else:
            if settings.dry_run:
                st.warning("DRY RUN MODE — No real orders will be placed.")

            try:
                from kalshi_weather_trader.ingestion.kalshi_fetcher import KalshiFetcher
                from kalshi_weather_trader.quant.monte_carlo import compute_normalized_market_probs

                with st.spinner("Fetching live Kalshi markets..."):
                    fetcher_s5 = KalshiFetcher()
                    markets_s5 = fetcher_s5.get_temperature_markets(target_date)

                cumulative_probs_s5 = mc_result_s5.probabilities
                prob_by_ticker_s5, prob_sum_raw_s5, gaps_s5 = compute_normalized_market_probs(
                    markets_s5, cumulative_probs_s5
                )

                if abs(prob_sum_raw_s5 - 1.0) > 0.01:
                    st.warning(
                        f"⚠️ Partition sum = {prob_sum_raw_s5:.1%} (expected 100%). "
                        f"Partition has {len(gaps_s5)} gap(s). "
                        f"Probabilities normalized for edge calculation."
                    )

                edge_rows_s5: list[dict] = []
                best_row_s5: dict | None = None
                best_abs_edge_s5: float = 0.0

                for m_s5 in sorted(markets_s5, key=lambda x: KalshiFetcher.extract_strike_from_market(x) or 0):
                    if KalshiFetcher.extract_strike_from_market(m_s5) is None:
                        continue

                    fair_value_s5 = prob_by_ticker_s5.get(m_s5.get("ticker", ""), 0.5)

                    yes_bid_s5 = m_s5.get("yes_bid") or 0
                    yes_ask_s5 = m_s5.get("yes_ask") or 0

                    if yes_ask_s5 > 0:
                        ask_dec_s5 = yes_ask_s5 / 100.0
                        b_s5 = (1.0 / ask_dec_s5) - 1.0
                        kelly_s5 = (fair_value_s5 * b_s5 - (1.0 - fair_value_s5)) / b_s5
                        frac_kelly_s5 = max(0.0, kelly_s5 * 0.25)
                        dollar_bet_s5 = frac_kelly_s5 * settings.max_trade_size_usd
                        contracts_s5 = max(1, min(
                            int(frac_kelly_s5 * settings.max_trade_size_usd / (ask_dec_s5 * 100)),
                            settings.max_contracts_per_market,
                        )) if frac_kelly_s5 > 0 else 0
                        edge_yes_s5 = round(fair_value_s5 - ask_dec_s5, 4)
                        signal_s5 = "BUY YES" if edge_yes_s5 > settings.edge_threshold else "—"
                        kelly_pct_s5 = f"{frac_kelly_s5:.1%}"
                    elif yes_bid_s5 > 0:
                        bid_dec_s5 = yes_bid_s5 / 100.0
                        no_ask_dec_s5 = 1.0 - bid_dec_s5
                        b_s5 = (1.0 / no_ask_dec_s5) - 1.0 if no_ask_dec_s5 > 0 else 0.0
                        p_no_s5 = 1.0 - fair_value_s5
                        kelly_s5 = (p_no_s5 * b_s5 - fair_value_s5) / b_s5 if b_s5 > 0 else 0.0
                        frac_kelly_s5 = max(0.0, kelly_s5 * 0.25)
                        dollar_bet_s5 = frac_kelly_s5 * settings.max_trade_size_usd
                        contracts_s5 = max(1, min(
                            int(frac_kelly_s5 * settings.max_trade_size_usd / (no_ask_dec_s5 * 100)),
                            settings.max_contracts_per_market,
                        )) if frac_kelly_s5 > 0 else 0
                        edge_no_s5 = round(p_no_s5 - no_ask_dec_s5, 4)
                        edge_yes_s5 = -edge_no_s5
                        signal_s5 = "BUY NO" if edge_no_s5 > settings.edge_threshold else "—"
                        kelly_pct_s5 = f"{frac_kelly_s5:.1%}"
                    else:
                        edge_yes_s5 = None
                        contracts_s5 = 0
                        signal_s5 = "NO LIQUIDITY"
                        kelly_pct_s5 = "—"
                        dollar_bet_s5 = 0.0

                    row_s5 = {
                        "Range": KalshiFetcher.get_strike_label(m_s5),
                        "Fair Value": f"{fair_value_s5:.4f}",
                        "Kalshi Ask": f"{yes_ask_s5}¢" if yes_ask_s5 else "—",
                        "Kalshi Bid": f"{yes_bid_s5}¢" if yes_bid_s5 else "—",
                        "YES Edge": f"{edge_yes_s5:+.4f}" if edge_yes_s5 is not None else "N/A",
                        "NO Edge": f"{(-edge_yes_s5):+.4f}" if edge_yes_s5 is not None else "N/A",
                        "Kelly %": kelly_pct_s5,
                        "Contracts": contracts_s5,
                        "Signal": signal_s5,
                    }
                    edge_rows_s5.append(row_s5)

                    # Track best edge for the written-out Kelly block
                    if edge_yes_s5 is not None and abs(edge_yes_s5) > best_abs_edge_s5 and yes_ask_s5 > 0:
                        best_abs_edge_s5 = abs(edge_yes_s5)
                        best_row_s5 = {
                            "range": KalshiFetcher.get_strike_label(m_s5),
                            "fair_value": fair_value_s5,
                            "ask_cents": yes_ask_s5,
                            "ask_dec": yes_ask_s5 / 100.0,
                            "edge": edge_yes_s5,
                            "b": (1.0 / (yes_ask_s5 / 100.0)) - 1.0,
                            "kelly": (fair_value_s5 * ((1.0 / (yes_ask_s5 / 100.0)) - 1.0) - (1.0 - fair_value_s5)) / ((1.0 / (yes_ask_s5 / 100.0)) - 1.0),
                            "frac_kelly": max(0.0, ((fair_value_s5 * ((1.0 / (yes_ask_s5 / 100.0)) - 1.0) - (1.0 - fair_value_s5)) / ((1.0 / (yes_ask_s5 / 100.0)) - 1.0)) * 0.25),
                            "dollar_bet": dollar_bet_s5,
                            "contracts": contracts_s5,
                            "signal": signal_s5,
                        }

                # Sum row
                edge_rows_s5.append({
                    "Range": "─────────",
                    "Fair Value": f"Σ = {prob_sum_raw_s5:.1%} (raw)",
                    "Kalshi Ask": "",
                    "Kalshi Bid": "",
                    "YES Edge": "",
                    "NO Edge": "",
                    "Kelly %": "",
                    "Contracts": "",
                    "Signal": "✓ OK" if abs(prob_sum_raw_s5 - 1.0) < 0.01 else "⚠ GAP",
                })

                if edge_rows_s5:
                    st.dataframe(edge_rows_s5, use_container_width=True, hide_index=True)
                else:
                    st.info("No markets with parsed strikes found.")

                # Written-out Kelly calculation for the best-edge market
                all_no_liquidity_s5 = all(r["Signal"] == "NO LIQUIDITY" for r in edge_rows_s5)
                if all_no_liquidity_s5 or not edge_rows_s5:
                    st.info("No live quotes — markets have no resting orders yet.")
                elif best_row_s5 is not None:
                    st.subheader("Kelly Calculation Detail")
                    r5 = best_row_s5
                    threshold_s5 = settings.edge_threshold
                    st.code(
                        f"TRADE CALCULATION — Range: {r5['range']}\n"
                        f"─────────────────────────────────────────\n"
                        f"Fair Value Probability:    {r5['fair_value']:.4f}\n"
                        f"Kalshi Ask:                {r5['ask_dec']:.4f}  ({r5['ask_cents']}¢)\n"
                        f"Raw Edge:                  {r5['edge']:+.4f}  "
                        f"{'✓ Above' if r5['edge'] > threshold_s5 else '✗ Below'} {threshold_s5} threshold\n"
                        f"\n"
                        f"Kelly Criterion:\n"
                        f"  b = (1 / {r5['ask_dec']:.4f}) - 1        = {r5['b']:.4f}\n"
                        f"  Kelly = ({r5['fair_value']:.4f}×{r5['b']:.4f} - {1-r5['fair_value']:.4f}) / {r5['b']:.4f} = {r5['kelly']:.4f} ({r5['kelly']*100:.1f}%)\n"
                        f"  25% Fractional Kelly               = {r5['frac_kelly']:.4f} ({r5['frac_kelly']*100:.1f}%)\n"
                        f"\n"
                        f"Position Sizing:\n"
                        f"  Max position size:        ${settings.max_trade_size_usd:.2f}\n"
                        f"  Dollar bet:               ${r5['dollar_bet']:.2f}\n"
                        f"  Price per contract:       ${r5['ask_dec']:.2f}\n"
                        f"  Contracts:                {r5['contracts']}\n"
                        f"\n"
                        f"Signal: {r5['signal']}"
                    )

            except Exception as exc_s5:
                import traceback as _tb5
                st.error(f"Stage 5 error: {exc_s5}")
                st.code(_tb5.format_exc())

    # -----------------------------------------------------------------------
    # Stage 6 — Historical Calibration Performance
    # -----------------------------------------------------------------------
    with st.expander("Stage 6: Historical Calibration Performance", expanded=False):
        try:
            import math as _math_s6
            import numpy as _np_s6

            # --- Section A: Summary Strip ---
            _s6_settled_markets: list[tuple] = []  # (date, market)
            for _s6_i in range(1, 15):
                _s6_d = target_date - timedelta(days=_s6_i)
                try:
                    _s6_m = db_manager.get_market(_s6_d)
                    if _s6_m is not None and _s6_m.final_official_high is not None:
                        _s6_settled_markets.append((_s6_d, _s6_m))
                except Exception:
                    pass

            _s6_n_settled = len(_s6_settled_markets)
            _s6_state_today = db_manager.get_system_state(target_date)
            _s6_last_cal_ts = "Never"
            if _s6_state_today and _s6_state_today.last_calibrated_utc:
                _s6_last_cal_ts = _s6_state_today.last_calibrated_utc.astimezone(_EASTERN).strftime("%m/%d %H:%M ET")

            # Avg NWP error across settled days
            _s6_nwp_errors: list[float] = []
            for _s6_d, _s6_m in _s6_settled_markets:
                try:
                    _s6_nwp_d = db_manager.get_latest_nwp_forecasts(_s6_d)
                    _s6_wts = _s6_state_today.model_weights if _s6_state_today else {"HRRR": 0.5, "GFS": 0.3, "ECMWF": 0.2}
                    _s6_blend_num = sum(
                        _s6_nwp_d[_m].predicted_daily_high * _s6_wts.get(_m, 0.0)
                        for _m in ["HRRR", "GFS", "ECMWF"]
                        if _m in _s6_nwp_d and _s6_nwp_d[_m].predicted_daily_high is not None
                    )
                    _s6_blend_den = sum(
                        _s6_wts.get(_m, 0.0)
                        for _m in ["HRRR", "GFS", "ECMWF"]
                        if _m in _s6_nwp_d and _s6_nwp_d[_m].predicted_daily_high is not None
                    )
                    if _s6_blend_den > 0:
                        _s6_nwp_errors.append(_s6_blend_num / _s6_blend_den - _s6_m.final_official_high)
                except Exception:
                    pass

            _s6_avg_err = (sum(_s6_nwp_errors) / len(_s6_nwp_errors)) if _s6_nwp_errors else None
            _s6_last_settled_date = _s6_settled_markets[0][0] if _s6_settled_markets else None
            _s6_last_official = _s6_settled_markets[0][1].final_official_high if _s6_settled_markets else None

            _s6c1, _s6c2, _s6c3, _s6c4 = st.columns(4)
            _s6c1.metric("Days Settled (14-day window)", f"{_s6_n_settled} / 14")
            _s6c2.metric(
                "Last Settlement",
                f"{_s6_last_settled_date} — {_s6_last_official:.1f}°F" if _s6_last_official else "None",
            )
            _s6c3.metric(
                "Avg NWP Error (°F)",
                f"{_s6_avg_err:+.2f}" if _s6_avg_err is not None else "N/A",
            )
            _s6c4.metric("Last Calibration", _s6_last_cal_ts)

            if _s6_n_settled == 0:
                st.info(
                    "No settled days yet — historical performance will populate after first settlement. "
                    "Settlement occurs at 7 PM ET (preliminary) and 10:05 AM ET next day (authoritative)."
                )
            else:
                # --- Section B: Yesterday's Result ---
                st.markdown("---")
                st.markdown("#### Yesterday's Result")
                _s6_yest = target_date - timedelta(days=1)
                try:
                    _s6_yest_m = db_manager.get_market(_s6_yest)
                    _s6_yest_nwp = db_manager.get_latest_nwp_forecasts(_s6_yest)
                    if _s6_yest_m is not None and _s6_yest_m.final_official_high is not None:
                        _s6_yoff = _s6_yest_m.final_official_high
                        _s6_y_wts = _s6_state_today.model_weights if _s6_state_today else {"HRRR": 0.5, "GFS": 0.3, "ECMWF": 0.2}
                        _s6_y_model_preds: dict[str, float | None] = {}
                        _s6_y_bn = _s6_y_bd = 0.0
                        for _m in ["HRRR", "GFS", "ECMWF"]:
                            if _m in _s6_yest_nwp and _s6_yest_nwp[_m].predicted_daily_high is not None:
                                _p = _s6_yest_nwp[_m].predicted_daily_high
                                _s6_y_model_preds[_m] = _p
                                _s6_y_bn += _p * _s6_y_wts.get(_m, 0.0)
                                _s6_y_bd += _s6_y_wts.get(_m, 0.0)
                            else:
                                _s6_y_model_preds[_m] = None
                        _s6_y_blend = (_s6_y_bn / _s6_y_bd) if _s6_y_bd > 0 else None
                        _s6_y_miss = (_s6_y_blend - _s6_yoff) if _s6_y_blend is not None else None

                        _yc1, _yc2, _yc3 = st.columns(3)
                        _yc1.metric("NWS Official High", f"{_s6_yoff:.1f}°F")
                        _yc2.metric("NWP Blended Forecast", f"{_s6_y_blend:.1f}°F" if _s6_y_blend else "N/A")
                        _yc3.metric(
                            "Miss",
                            f"{_s6_y_miss:+.2f}°F" if _s6_y_miss is not None else "N/A",
                            delta=f"{_s6_y_miss:+.2f}°F" if _s6_y_miss is not None else None,
                            delta_color="inverse",
                        )

                        # Per-model breakdown — which was closest
                        _s6_model_rows_b = []
                        _closest_err = float("inf")
                        _closest_m = None
                        for _m in ["HRRR", "GFS", "ECMWF"]:
                            _p = _s6_y_model_preds.get(_m)
                            _e = (_p - _s6_yoff) if _p is not None else None
                            if _e is not None and abs(_e) < abs(_closest_err):
                                _closest_err = _e
                                _closest_m = _m
                            _s6_model_rows_b.append({
                                "Model": _m,
                                "Predicted": f"{_p:.1f}°F" if _p is not None else "—",
                                "Error": f"{_e:+.2f}°F" if _e is not None else "—",
                            })
                        if _closest_m:
                            st.caption(f"Closest model yesterday: **{_closest_m}** (error = {_closest_err:+.2f}°F)")
                        st.dataframe(_s6_model_rows_b, use_container_width=True, hide_index=True)

                        # Snapshot count
                        _s6_y_snaps = db_manager.get_snapshots_for_date(_s6_yest)
                        _s6_y_snaps_w_prob = [s for s in _s6_y_snaps if s.model_fair_value_prob is not None]
                        st.caption(
                            f"Yesterday had **{len(_s6_y_snaps)}** intraday snapshots recorded, "
                            f"**{len(_s6_y_snaps_w_prob)}** with fair-value probability stored."
                        )
                    else:
                        st.info(f"Yesterday ({_s6_yest}) is not yet settled.")
                except Exception as _exc_s6b:
                    st.warning(f"Could not load yesterday's result: {_exc_s6b}")

                # --- Section C: NWP Forecast Accuracy Over Time ---
                st.markdown("---")
                st.markdown("#### NWP Forecast Accuracy Over Time")
                try:
                    _s6_dates_c: list[str] = []
                    _s6_hrrr_errs: list[float | None] = []
                    _s6_gfs_errs: list[float | None] = []
                    _s6_ecmwf_errs: list[float | None] = []
                    _s6_blend_errs: list[float | None] = []

                    for _s6_d, _s6_m in reversed(_s6_settled_markets):
                        _s6_dates_c.append(str(_s6_d))
                        try:
                            _s6_nwp_c = db_manager.get_latest_nwp_forecasts(_s6_d)
                            _off = _s6_m.final_official_high
                            _wts_c = _s6_state_today.model_weights if _s6_state_today else {"HRRR": 0.5, "GFS": 0.3, "ECMWF": 0.2}
                            _h_err = (_s6_nwp_c["HRRR"].predicted_daily_high - _off) if "HRRR" in _s6_nwp_c and _s6_nwp_c["HRRR"].predicted_daily_high else None
                            _g_err = (_s6_nwp_c["GFS"].predicted_daily_high - _off) if "GFS" in _s6_nwp_c and _s6_nwp_c["GFS"].predicted_daily_high else None
                            _e_err = (_s6_nwp_c["ECMWF"].predicted_daily_high - _off) if "ECMWF" in _s6_nwp_c and _s6_nwp_c["ECMWF"].predicted_daily_high else None
                            _s6_hrrr_errs.append(_h_err)
                            _s6_gfs_errs.append(_g_err)
                            _s6_ecmwf_errs.append(_e_err)
                            # Blended
                            _b_n = sum(
                                _s6_nwp_c[_m].predicted_daily_high * _wts_c.get(_m, 0.0)
                                for _m in ["HRRR", "GFS", "ECMWF"]
                                if _m in _s6_nwp_c and _s6_nwp_c[_m].predicted_daily_high
                            )
                            _b_d = sum(
                                _wts_c.get(_m, 0.0)
                                for _m in ["HRRR", "GFS", "ECMWF"]
                                if _m in _s6_nwp_c and _s6_nwp_c[_m].predicted_daily_high
                            )
                            _s6_blend_errs.append((_b_n / _b_d - _off) if _b_d > 0 else None)
                        except Exception:
                            _s6_hrrr_errs.append(None)
                            _s6_gfs_errs.append(None)
                            _s6_ecmwf_errs.append(None)
                            _s6_blend_errs.append(None)

                    if _s6_dates_c:
                        _fig_c = go.Figure()
                        _model_colors = {"HRRR": "cornflowerblue", "GFS": "orange", "ECMWF": "green"}
                        for _m_name, _errs_list in [
                            ("HRRR", _s6_hrrr_errs),
                            ("GFS", _s6_gfs_errs),
                            ("ECMWF", _s6_ecmwf_errs),
                            ("Blended", _s6_blend_errs),
                        ]:
                            _valid_x = [_s6_dates_c[i] for i, e in enumerate(_errs_list) if e is not None]
                            _valid_y = [e for e in _errs_list if e is not None]
                            if _valid_y:
                                _fig_c.add_trace(go.Scatter(
                                    x=_valid_x,
                                    y=_valid_y,
                                    mode="lines+markers",
                                    name=_m_name,
                                    line=dict(
                                        color=_model_colors.get(_m_name, "purple"),
                                        dash="dot" if _m_name == "Blended" else "solid",
                                    ),
                                ))
                        _fig_c.add_hline(y=0, line_dash="dash", line_color="grey", annotation_text="Perfect forecast")
                        _fig_c.update_layout(
                            title="NWP Daily Forecast Error by Model (forecast − actual °F)",
                            xaxis_title="Date",
                            yaxis_title="Error (°F)",
                            height=350,
                            margin=dict(t=40, b=30),
                        )
                        st.plotly_chart(_fig_c, use_container_width=True)
                except Exception as _exc_s6c:
                    st.warning(f"Could not render NWP accuracy chart: {_exc_s6c}")

                # --- Section D: Model Weight History ---
                st.markdown("---")
                st.markdown("#### Model Weight History")
                try:
                    _s6_wt_dates: list[str] = []
                    _s6_wt_hrrr: list[float] = []
                    _s6_wt_gfs: list[float] = []
                    _s6_wt_ecmwf: list[float] = []

                    for _s6_i in range(13, -1, -1):
                        _s6_wd = target_date - timedelta(days=_s6_i)
                        try:
                            _s6_ws = db_manager.get_system_state(_s6_wd)
                            if _s6_ws and _s6_ws.model_weights:
                                _s6_wt_dates.append(str(_s6_wd))
                                _s6_wt_hrrr.append(_s6_ws.model_weights.get("HRRR", 0.5))
                                _s6_wt_gfs.append(_s6_ws.model_weights.get("GFS", 0.3))
                                _s6_wt_ecmwf.append(_s6_ws.model_weights.get("ECMWF", 0.2))
                        except Exception:
                            pass

                    if len(_s6_wt_dates) < 2:
                        # Single bar chart
                        _wts_now = _s6_state_today.model_weights if _s6_state_today else {"HRRR": 0.5, "GFS": 0.3, "ECMWF": 0.2}
                        _fig_wd = go.Figure(go.Bar(
                            x=list(_wts_now.keys()),
                            y=list(_wts_now.values()),
                            marker_color=["cornflowerblue", "orange", "green"],
                        ))
                        _fig_wd.update_layout(
                            title="Current Model Weights (insufficient history for trend)",
                            yaxis_title="Weight",
                            yaxis_range=[0, 1],
                            height=300,
                            margin=dict(t=40, b=30),
                        )
                        st.plotly_chart(_fig_wd, use_container_width=True)
                        st.caption("Insufficient history for trend — will populate over coming days.")
                    else:
                        _fig_wd = go.Figure()
                        for _m_name, _wt_series, _col in [
                            ("HRRR", _s6_wt_hrrr, "cornflowerblue"),
                            ("GFS", _s6_wt_gfs, "orange"),
                            ("ECMWF", _s6_wt_ecmwf, "green"),
                        ]:
                            _fig_wd.add_trace(go.Scatter(
                                x=_s6_wt_dates,
                                y=_wt_series,
                                mode="lines+markers",
                                name=_m_name,
                                line=dict(color=_col),
                            ))
                        _fig_wd.update_layout(
                            title="Model Weight History (14-day)",
                            xaxis_title="Date",
                            yaxis_title="Weight",
                            yaxis_range=[0, 1],
                            height=350,
                            margin=dict(t=40, b=30),
                        )
                        st.plotly_chart(_fig_wd, use_container_width=True)
                        st.caption(
                            "Weights update daily via softmax(1/Brier_14d).  "
                            "Requires ≥2 settled days."
                        )
                except Exception as _exc_s6d:
                    st.warning(f"Could not render weight history: {_exc_s6d}")

                # --- Section E: Calibration Scatter ---
                st.markdown("---")
                st.markdown("#### Calibration Scatter — Predicted Probability vs Actual Outcome")
                try:
                    _s6_scatter_x: list[float] = []
                    _s6_scatter_y: list[int] = []
                    _s6_scatter_days: int = 0

                    for _s6_d, _s6_m in _s6_settled_markets:
                        try:
                            _s6_snaps_e = db_manager.get_snapshots_for_date(_s6_d)
                            _s6_off_e = _s6_m.final_official_high
                            for _s6_sn in _s6_snaps_e:
                                if _s6_sn.model_fair_value_prob is None or _s6_sn.kalshi_strike is None:
                                    continue
                                _outcome = 1 if _s6_off_e >= float(_s6_sn.kalshi_strike) else 0
                                _s6_scatter_x.append(_s6_sn.model_fair_value_prob)
                                _s6_scatter_y.append(_outcome)
                            _s6_scatter_days += 1
                        except Exception:
                            pass

                    if len(_s6_scatter_x) < 10:
                        st.info(
                            f"Calibration scatter requires more settled days — currently "
                            f"**{len(_s6_scatter_x)}** points from **{_s6_scatter_days}** days. "
                            "Will improve over time."
                        )
                    else:
                        _fig_e = go.Figure()
                        _fig_e.add_trace(go.Scatter(
                            x=_s6_scatter_x,
                            y=_s6_scatter_y,
                            mode="markers",
                            name="Snapshots",
                            marker=dict(
                                color=_s6_scatter_y,
                                colorscale=[[0, "salmon"], [1, "steelblue"]],
                                opacity=0.6,
                                size=8,
                            ),
                        ))
                        _fig_e.add_shape(
                            type="line", x0=0, y0=0, x1=1, y1=1,
                            line=dict(dash="dash", color="grey"),
                        )
                        _fig_e.update_layout(
                            title=f"Model Calibration — {len(_s6_scatter_x)} snapshots from {_s6_scatter_days} settled days",
                            xaxis_title="Predicted Probability (model fair value)",
                            yaxis_title="Actual Outcome (1 = YES resolved, 0 = NO)",
                            yaxis=dict(tickvals=[0, 1], ticktext=["NO (0)", "YES (1)"]),
                            height=400,
                            margin=dict(t=40, b=30),
                        )
                        st.plotly_chart(_fig_e, use_container_width=True)
                        st.caption(
                            "Perfect calibration = points near the diagonal.  "
                            "Clusters near 0 or 1 on y-axis are expected — the scatter shows "
                            "whether high-confidence predictions resolved correctly."
                        )
                except Exception as _exc_s6e:
                    st.warning(f"Could not render calibration scatter: {_exc_s6e}")

                # --- Section F: Intraday Snapshot Replay ---
                st.markdown("---")
                st.markdown("#### Intraday Snapshot Replay")
                try:
                    _s6_settled_dates = [d for d, _ in _s6_settled_markets]
                    _s6_replay_date = st.date_input(
                        "Select date to replay",
                        value=_s6_settled_dates[0] if _s6_settled_dates else target_date - timedelta(days=1),
                        min_value=_s6_settled_dates[-1] if _s6_settled_dates else target_date - timedelta(days=14),
                        max_value=_s6_settled_dates[0] if _s6_settled_dates else target_date - timedelta(days=1),
                        key="s6_replay_date_picker",
                    )
                    _s6_replay_snaps = db_manager.get_snapshots_for_date(_s6_replay_date)
                    _s6_replay_market = db_manager.get_market(_s6_replay_date)
                    _s6_replay_official = (
                        _s6_replay_market.final_official_high
                        if _s6_replay_market and _s6_replay_market.final_official_high
                        else None
                    )

                    if not _s6_replay_snaps:
                        st.info(f"No snapshots recorded for {_s6_replay_date}.")
                    else:
                        _s6_times = [s.snapshot_time_eastern for s in _s6_replay_snaps]
                        _fig_f = go.Figure()

                        # Left y-axis: temperatures
                        _fig_f.add_trace(go.Scatter(
                            x=_s6_times,
                            y=[s.blended_predicted_high for s in _s6_replay_snaps],
                            mode="lines+markers",
                            name="Blended NWP Predicted High",
                            line=dict(color="steelblue"),
                        ))
                        _fig_f.add_trace(go.Scatter(
                            x=_s6_times,
                            y=[s.current_asos_temp_f for s in _s6_replay_snaps],
                            mode="lines",
                            name="ASOS Current Temp",
                            line=dict(color="tomato", dash="dot"),
                        ))
                        _fig_f.add_trace(go.Scatter(
                            x=_s6_times,
                            y=[s.current_max_observed_f for s in _s6_replay_snaps],
                            mode="lines",
                            name="Hard Floor (Max Observed)",
                            line=dict(color="orange", dash="dash"),
                        ))
                        if _s6_replay_official is not None:
                            _fig_f.add_hline(
                                y=_s6_replay_official,
                                line_dash="longdash",
                                line_color="darkgreen",
                                annotation_text=f"NWS Official: {_s6_replay_official:.1f}°F",
                                annotation_position="top left",
                            )

                        # Right y-axis: probabilities
                        _s6_probs = [s.model_fair_value_prob for s in _s6_replay_snaps]
                        _s6_implied = [s.kalshi_implied_prob_yes for s in _s6_replay_snaps]
                        if any(p is not None for p in _s6_probs):
                            _fig_f.add_trace(go.Scatter(
                                x=_s6_times,
                                y=_s6_probs,
                                mode="lines+markers",
                                name="Model Fair Value Prob",
                                yaxis="y2",
                                line=dict(color="purple"),
                            ))
                        if any(p is not None for p in _s6_implied):
                            _fig_f.add_trace(go.Scatter(
                                x=_s6_times,
                                y=_s6_implied,
                                mode="lines",
                                name="Kalshi Implied Prob",
                                yaxis="y2",
                                line=dict(color="mediumpurple", dash="dot"),
                            ))

                        _fig_f.update_layout(
                            title=f"Intraday Model vs Market — {_s6_replay_date}",
                            xaxis_title="Time (ET)",
                            yaxis=dict(title="Temperature (°F)"),
                            yaxis2=dict(
                                title="Probability",
                                overlaying="y",
                                side="right",
                                range=[0, 1],
                                tickformat=".0%",
                            ),
                            height=450,
                            margin=dict(t=40, b=30),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                        )
                        st.plotly_chart(_fig_f, use_container_width=True)
                        st.caption(f"{len(_s6_replay_snaps)} snapshots recorded for {_s6_replay_date}.")
                except Exception as _exc_s6f:
                    st.warning(f"Could not render snapshot replay: {_exc_s6f}")

        except Exception as exc_s6:
            import traceback as _tb_s6
            st.error(f"Stage 6 error: {exc_s6}")
            st.code(_tb_s6.format_exc())


# -----------------------------------------------------------------------
# Main app
# -----------------------------------------------------------------------


def main() -> None:
    """Entry point for the Streamlit app.

    Args:
        None

    Returns:
        None

    Raises:
        Nothing — all errors surface as st.error() messages.
    """
    target_date = get_target_date()

    st.title("🌡️ Kalshi KBOS Temperature Trader")
    st.caption(f"Target date: **{target_date}** | DRY RUN: {'✓' if settings.dry_run else '✗'}")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Trading Desk", "📈 Visualizer", "🔧 Calibration", "🔬 Model Transparency"
    ])

    with tab1:
        render_trading_desk(target_date)

    with tab2:
        render_visualizer(target_date)

    with tab3:
        render_calibration(target_date)

    with tab4:
        render_model_transparency(target_date)

    # Auto-refresh every 5 minutes — aligned with ASOS fetch and trade-eval cadence
    time.sleep(300)
    st.rerun()


if __name__ == "__main__":
    main()
