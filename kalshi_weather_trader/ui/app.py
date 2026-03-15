"""
Streamlit command center for the Kalshi weather trading system.

Three tabs:
  Tab 1 — Trading Desk:  Live metrics, kill switch, edge table, recent trades.
  Tab 2 — Visualizer:    Plotly chart with ASOS history, NWP curves, MC band,
                         hard floor line, and Kalshi implied vs model probability.
  Tab 3 — Calibration:   Model weights bar chart, drift sliders, force snapshot,
                         recalibrate button, snapshot history table.

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
                from kalshi_weather_trader.quant.monte_carlo import compute_yes_prob

                now_et_ui = datetime.now(timezone.utc).astimezone(_EASTERN)
                hour_et_ui = now_et_ui.hour
                # After 6 PM ET rollover target_date is tomorrow → start from curve index 0
                is_future_day_ui = target_date > now_et_ui.date()
                hour_offset_ui = 0 if is_future_day_ui else hour_et_ui

                mkt = db_manager.get_market(target_date)
                hard_floor = (mkt.current_max_observed if mkt else None) or state.kalman_temp_estimate

                # Fall back to a flat curve at current temp if NWP is missing
                effective_curve = nwp_curve if nwp_curve else [state.kalman_temp_estimate] * 24

                # Collect ALL threshold values (floor + cap) from every market so
                # compute_yes_prob can look up cumulative probs for each boundary.
                all_strikes_set_ui: set[float] = set()
                for m in markets:
                    floor_raw_ui = m.get("floor_strike")
                    cap_raw_ui = m.get("cap_strike")
                    if floor_raw_ui is not None:
                        all_strikes_set_ui.add(float(floor_raw_ui))
                    if cap_raw_ui is not None:
                        all_strikes_set_ui.add(float(cap_raw_ui))
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

                for m in sorted(markets, key=lambda x: KalshiFetcher.extract_strike_from_market(x) or 0):
                    if KalshiFetcher.extract_strike_from_market(m) is None:
                        continue
                    model_p = compute_yes_prob(
                        cumulative_probs, m.get("floor_strike"), m.get("cap_strike")
                    )

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

        except Exception as e:
            edge_error = _tb.format_exc()
            edge_diag.append(f"EXCEPTION: {e}")

        st.session_state["edge_table_rows"] = edge_rows
        st.session_state["edge_table_diag"] = edge_diag
        st.session_state["edge_table_error"] = edge_error

    # Display
    rows = st.session_state.get("edge_table_rows", [])
    diag = st.session_state.get("edge_table_diag", [])
    err = st.session_state.get("edge_table_error")

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

    tab1, tab2, tab3 = st.tabs(["📊 Trading Desk", "📈 Visualizer", "🔧 Calibration"])

    with tab1:
        render_trading_desk(target_date)

    with tab2:
        render_visualizer(target_date)

    with tab3:
        render_calibration(target_date)

    # Auto-refresh every 60 seconds
    time.sleep(60)
    st.rerun()


if __name__ == "__main__":
    main()
