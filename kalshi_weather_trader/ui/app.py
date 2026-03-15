"""
Streamlit command center for the Kalshi weather trading system.

Four tabs:
  Tab 1 — Trading Desk:         Live metrics, kill switch, edge table, recent trades.
  Tab 2 — Visualizer:           Plotly chart with ASOS history, NWP curves, MC band,
                                hard floor line, and Kalshi implied vs model probability.
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
# Uses a module-level flag guarded by a lock so re-runs of the script
# (every page interaction) never create duplicate schedulers.
# -----------------------------------------------------------------------
_scheduler_lock = threading.Lock()
_scheduler_started = False


def _maybe_start_scheduler() -> None:
    """Start the APScheduler background scheduler once, if not already running."""
    global _scheduler_started
    with _scheduler_lock:
        if _scheduler_started:
            return
        try:
            from kalshi_weather_trader.scheduler.orchestrator import (
                build_scheduler,
                startup_sequence,
            )

            startup_sequence()
            _sched = build_scheduler()
            _sched.start()
            _scheduler_started = True
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
                # After 6 PM ET rollover target_date is tomorrow → start from curve index 0
                is_future_day_ui = target_date > now_et_ui.date()
                hour_offset_ui = 0 if is_future_day_ui else hour_et_ui

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
        colors = {"HRRR": "green", "GFS": "orange", "ECMWF": "purple"}
        for model_name, forecast in nwp_forecasts.items():
            if forecast.hourly_temps:
                hours = [
                    (day_start + pd.Timedelta(hours=i)).astimezone(_EASTERN)
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
                (day_start + pd.Timedelta(hours=i)).astimezone(_EASTERN)
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

        # Second chart: Kalshi implied vs model probability
        if snapshots and any(s.kalshi_implied_prob_yes is not None for s in snapshots):
            fig2 = go.Figure()
            snap_with_probs = [s for s in snapshots if s.kalshi_implied_prob_yes is not None]
            if snap_with_probs:
                times = [s.snapshot_time_utc.astimezone(_EASTERN) for s in snap_with_probs]
                implied = [s.kalshi_implied_prob_yes for s in snap_with_probs]
                fair_vals = [s.model_fair_value_prob for s in snap_with_probs]

                fig2.add_trace(go.Scatter(
                    x=times, y=implied,
                    mode="lines+markers", name="Kalshi Implied P(YES)",
                    line=dict(color="red"),
                ))
                if any(v is not None for v in fair_vals):
                    fig2.add_trace(go.Scatter(
                        x=times, y=[v for v in fair_vals if v is not None],
                        mode="lines+markers", name="Model Fair Value",
                        line=dict(color="blue"),
                    ))

                fig2.update_layout(
                    title="Kalshi Implied vs Model Probability",
                    xaxis_title="Time (Eastern)",
                    yaxis_title="Probability",
                    yaxis=dict(range=[0, 1]),
                    height=350,
                )
                st.plotly_chart(fig2, use_container_width=True)

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

        # Compute blended NWP at current UTC hour (same logic as Visualizer)
        blended_now: float | None = None
        if nwp_forecasts and state:
            now_hour_utc = datetime.now(timezone.utc).hour
            model_weights = state.model_weights
            total_w = sum(
                model_weights.get(n, 0.0)
                for n in nwp_forecasts
                if nwp_forecasts[n].hourly_temps and now_hour_utc < len(nwp_forecasts[n].hourly_temps)
            )
            if total_w > 0:
                blended_now = 0.0
                for name, f in nwp_forecasts.items():
                    if not f.hourly_temps or now_hour_utc >= len(f.hourly_temps):
                        continue
                    w = model_weights.get(name, 0.0) / total_w
                    blended_now += w * f.hourly_temps[now_hour_utc]

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
                        st.metric("Temp Variance", f"{temp_var:.4f}")
                    except (IndexError, TypeError, ValueError):
                        st.metric("Temp Variance", "N/A")
                else:
                    st.metric("Temp Variance", "N/A")

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
                day_start_s2, _ = get_trading_day_bounds()
                fig_s2 = go.Figure()

                for model_name_s2, forecast_s2 in nwp_forecasts_s2.items():
                    if forecast_s2.hourly_temps:
                        w_s2 = model_weights_s2.get(model_name_s2, 0.0)
                        hours_s2 = [
                            (day_start_s2 + pd.Timedelta(hours=i)).astimezone(_EASTERN)
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
                            (day_start_s2 + pd.Timedelta(hours=i)).astimezone(_EASTERN)
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

                # Bottom row — drift adjustments and attractor
                if state_s2:
                    now_hour_utc_s2 = now_utc_s2.hour
                    hour_et_s2 = now_utc_s2.astimezone(_EASTERN).hour
                    drift_s2 = state_s2.morning_drift_adjustment if hour_et_s2 < 12 else state_s2.afternoon_drift_adjustment

                    # Blended NWP at current UTC hour
                    blended_at_now_s2: float | None = None
                    total_w_now_s2 = sum(
                        model_weights_s2.get(n, 0.0)
                        for n in nwp_forecasts_s2
                        if nwp_forecasts_s2[n].hourly_temps and now_hour_utc_s2 < len(nwp_forecasts_s2[n].hourly_temps)
                    )
                    if total_w_now_s2 > 0:
                        blended_at_now_s2 = 0.0
                        for name_s2b, f_s2c in nwp_forecasts_s2.items():
                            if not f_s2c.hourly_temps or now_hour_utc_s2 >= len(f_s2c.hourly_temps):
                                continue
                            w_n = model_weights_s2.get(name_s2b, 0.0) / total_w_now_s2
                            blended_at_now_s2 += w_n * f_s2c.hourly_temps[now_hour_utc_s2]

                    attractor_s2: float | None = None
                    if blended_at_now_s2 is not None:
                        attractor_s2 = blended_at_now_s2 + state_s2.kalman_bias_estimate + drift_s2

                    bd1, bd2, bd3 = st.columns(3)
                    with bd1:
                        st.metric("Morning Drift Adj.", f"{state_s2.morning_drift_adjustment:+.3f}°F")
                    with bd2:
                        st.metric("Afternoon Drift Adj.", f"{state_s2.afternoon_drift_adjustment:+.3f}°F")
                    with bd3:
                        if attractor_s2 is not None:
                            st.metric("The Attractor (μ_t)", f"{attractor_s2:.1f}°F")
                        else:
                            st.metric("The Attractor (μ_t)", "N/A")
                    st.caption("The mean-reversion target the Monte Carlo simulates toward right now.")

        except Exception as exc_s2:
            import traceback as _tb2
            st.error(f"Stage 2 error: {exc_s2}")
            st.code(_tb2.format_exc())

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
                hour_utc_s3 = now_utc_s3.hour
                is_future_day_s3 = target_date > now_et_s3.date()
                hour_offset_s3 = 0 if is_future_day_s3 else hour_utc_s3

                hard_floor_s3 = (market_s3.current_max_observed if market_s3 else None) or state_s3.kalman_temp_estimate
                nwp_curve_s3 = get_nwp_curve(target_date)
                effective_curve_s3 = nwp_curve_s3 if nwp_curve_s3 else [state_s3.kalman_temp_estimate] * 24
                day_fraction_s3 = max(0.0, 1.0 - hour_offset_s3 / 24.0) if not is_future_day_s3 else 1.0
                n_steps_s3 = int(day_fraction_s3 * 288)

                # Blended NWP at current UTC hour
                model_weights_s3 = state_s3.model_weights
                nwp_at_now_s3: float | None = None
                if nwp_forecasts_s3:
                    total_w_s3 = sum(
                        model_weights_s3.get(n, 0.0)
                        for n in nwp_forecasts_s3
                        if nwp_forecasts_s3[n].hourly_temps and hour_utc_s3 < len(nwp_forecasts_s3[n].hourly_temps)
                    )
                    if total_w_s3 > 0:
                        nwp_at_now_s3 = 0.0
                        for name_s3, f_s3 in nwp_forecasts_s3.items():
                            if not f_s3.hourly_temps or hour_utc_s3 >= len(f_s3.hourly_temps):
                                continue
                            nwp_at_now_s3 += model_weights_s3.get(name_s3, 0.0) / total_w_s3 * f_s3.hourly_temps[hour_utc_s3]

                attractor_s3 = (nwp_at_now_s3 + state_s3.kalman_bias_estimate) if nwp_at_now_s3 is not None else None

                param_rows_s3 = [
                    {"Parameter": "Hard Floor (current_max_observed)", "Value": f"{hard_floor_s3:.1f}°F"},
                    {"Parameter": "Starting Temp (Kalman T₀)", "Value": f"{state_s3.kalman_temp_estimate:.1f}°F"},
                    {"Parameter": "Kalman Bias", "Value": f"{state_s3.kalman_bias_estimate:+.3f}°F"},
                    {"Parameter": "Theta (mean-reversion speed)", "Value": f"{state_s3.theta_decay:.4f}/hr"},
                    {"Parameter": "Sigma (volatility °F/√hr)", "Value": f"{state_s3.sigma_volatility:.3f}"},
                    {"Parameter": "Mu Drift", "Value": f"{state_s3.mu_drift:.4f}°F" if hasattr(state_s3, 'mu_drift') and state_s3.mu_drift is not None else "N/A"},
                    {"Parameter": "Hour Offset (UTC)", "Value": str(hour_offset_s3)},
                    {"Parameter": "Remaining Day Fraction", "Value": f"{day_fraction_s3:.3f}"},
                    {"Parameter": "N Steps (5-min intervals)", "Value": str(n_steps_s3)},
                    {"Parameter": "N Paths", "Value": str(settings.mc_n_paths)},
                    {"Parameter": "NWP Attractor at Current Hour", "Value": f"{attractor_s3:.1f}°F" if attractor_s3 is not None else "N/A"},
                ]
                st.dataframe(param_rows_s3, use_container_width=True, hide_index=True)

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

    # Auto-refresh every 60 seconds
    time.sleep(60)
    st.rerun()


if __name__ == "__main__":
    main()
