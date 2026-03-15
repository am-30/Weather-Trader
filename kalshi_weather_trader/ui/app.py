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

import time
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

from kalshi_weather_trader.config.settings import get_target_date, settings  # noqa: E402
from kalshi_weather_trader.db import db_manager  # noqa: E402

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

    # Edge table from latest snapshots
    st.subheader("Latest Edge Estimates")
    try:
        snapshots = db_manager.get_snapshots_for_date(target_date)
        if snapshots:
            latest = snapshots[-1]
            if latest.kalshi_strike is not None:
                rows = [{
                    "Strike (°F)": latest.kalshi_strike,
                    "Market Bid": f"{(latest.kalshi_bid or 0)*100:.0f}¢" if latest.kalshi_bid else "N/A",
                    "Market Ask": f"{(latest.kalshi_ask or 0)*100:.0f}¢" if latest.kalshi_ask else "N/A",
                    "Model P(YES)": f"{(latest.model_fair_value_prob or 0)*100:.1f}%" if latest.model_fair_value_prob else "N/A",
                    "Edge": f"{(latest.model_edge or 0)*100:+.1f}%" if latest.model_edge else "N/A",
                    "Signal": "🟢 BUY YES" if (latest.model_edge or 0) > settings.edge_threshold
                              else "🔴 BUY NO" if (latest.model_edge or 0) < -settings.edge_threshold
                              else "⚪ NO TRADE",
                }]
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("No Kalshi market data in latest snapshot.")
        else:
            st.info("No snapshots yet for today.")
    except Exception as exc:
        st.warning(f"Could not load edge data: {exc}")

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

        # MC percentile band (25th–75th)
        if snapshots:
            snap_times = [s.snapshot_time_utc.astimezone(_EASTERN) for s in snapshots]
            # Use blended_predicted_high as a proxy for median
            blended = [s.blended_predicted_high for s in snapshots]
            fig.add_trace(go.Scatter(
                x=snap_times, y=blended,
                mode="lines", name="Blended Forecast",
                line=dict(color="darkorange", width=2),
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

    # Action buttons
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
