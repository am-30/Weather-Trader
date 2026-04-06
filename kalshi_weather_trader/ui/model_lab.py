"""
Model Lab — Tab 6 of the Kalshi Weather Trader Streamlit app.

Phase L1: Replay mode only.
  - Select a preset scenario (or stay with Production).
  - Choose a date range and eval hours.
  - Click Run — the parameterized replay engine re-runs the MC for every
    settled date using the scenario's parameter overrides.
  - See summary metrics (Brier, RMSE, Bias, Sharpness), calibration curve,
    per-date scatter, sortable result table, per-hour Brier bars, and an
    error distribution histogram.

Results are cached by (scenario hash, date range, eval hours) so that
a second run with the same settings is instant.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytz
import streamlit as st

_EASTERN = pytz.timezone("America/New_York")


def _today_et() -> date:
    return datetime.now(timezone.utc).astimezone(_EASTERN).date()


# ---------------------------------------------------------------------------
# Cached replay runner — keyed on scenario hash + date strings
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Running replay engine…", ttl=3600)
def _run_replay_cached(
    scenario_hash: int,
    scenario,
    start_str: str,
    end_str: str,
) -> list:
    """Run ParameterizedReplayEngine and return results.

    Args:
        scenario_hash: Hash of scenario (used as cache key; not used inside).
        scenario:      Scenario object with all overrides.
        start_str:     ISO date string for start_date.
        end_str:       ISO date string for end_date.

    Returns:
        list[ParameterizedReplayResult]
    """
    from kalshi_weather_trader.backtesting.replay_engine import ParameterizedReplayEngine

    engine = ParameterizedReplayEngine()
    start = date.fromisoformat(start_str)
    end = date.fromisoformat(end_str)
    return engine.replay_scenario(scenario, start_date=start, end_date=end)


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------


def _calibration_chart(calibration_curve: list) -> go.Figure:
    """Plotly calibration curve: predicted probability vs observed frequency."""
    if not calibration_curve:
        return go.Figure()

    x = [bin_["mean_pred"] for bin_ in calibration_curve]
    y = [bin_["observed_freq"] for bin_ in calibration_curve]
    sizes = [max(6, min(20, bin_["count"] / 5)) for bin_ in calibration_curve]
    counts = [bin_["count"] for bin_ in calibration_curve]

    fig = go.Figure()
    # Perfect calibration diagonal
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color="gray", dash="dash", width=1),
        name="Perfect calibration",
        hoverinfo="skip",
    ))
    # Calibration points
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="markers+lines",
        marker=dict(size=sizes, color="steelblue", opacity=0.85),
        text=[f"n={c}" for c in counts],
        hovertemplate="Predicted: %{x:.2f}<br>Observed: %{y:.2f}<br>%{text}<extra></extra>",
        name="Model calibration",
    ))
    fig.update_layout(
        title="Calibration Curve",
        xaxis_title="Predicted probability",
        yaxis_title="Observed frequency",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        height=380,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def _scatter_chart(results: list) -> go.Figure:
    """Actual high vs predicted mean_max scatter."""
    if not results:
        return go.Figure()

    x = [r.actual_high for r in results]
    y = [r.mean_max for r in results]
    hover = [
        f"Date: {r.target_date}<br>Hour: {r.eval_hour}h ET<br>"
        f"Predicted: {r.mean_max:.1f}°F<br>Actual: {r.actual_high:.1f}°F<br>"
        f"Error: {r.prediction_error:+.1f}°F"
        for r in results
    ]
    colors = [r.eval_hour for r in results]

    lo = min(min(x), min(y)) - 2
    hi = max(max(x), max(y)) + 2

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi],
        mode="lines",
        line=dict(color="gray", dash="dash", width=1),
        hoverinfo="skip",
        name="Perfect prediction",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="markers",
        marker=dict(
            color=colors,
            colorscale="RdYlGn_r",
            showscale=True,
            colorbar=dict(title="Eval hour ET"),
            size=7,
            opacity=0.8,
        ),
        text=hover,
        hovertemplate="%{text}<extra></extra>",
        name="Predictions",
    ))
    fig.update_layout(
        title="Predicted vs Actual High",
        xaxis_title="Actual high (°F)",
        yaxis_title="Predicted mean_max (°F)",
        height=380,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def _per_hour_brier_chart(per_hour: dict) -> go.Figure:
    """Bar chart of Brier score per eval hour."""
    if not per_hour:
        return go.Figure()

    hours = sorted(per_hour.keys())
    values = [per_hour[h] for h in hours]

    fig = go.Figure(go.Bar(
        x=[f"{h}h ET" for h in hours],
        y=values,
        marker_color="steelblue",
        text=[f"{v:.4f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Brier Score by Eval Hour",
        xaxis_title="Eval hour (ET)",
        yaxis_title="Mean Brier score",
        height=320,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def _error_histogram(results: list) -> go.Figure:
    """Histogram of prediction_error (mean_max − actual_high)."""
    if not results:
        return go.Figure()

    errors = [r.prediction_error for r in results]
    fig = go.Figure(go.Histogram(
        x=errors,
        nbinsx=20,
        marker_color="steelblue",
        opacity=0.8,
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    mean_err = float(np.mean(errors))
    fig.add_vline(x=mean_err, line_color="red", annotation_text=f"Mean: {mean_err:+.2f}°F")
    fig.update_layout(
        title="Prediction Error Distribution (mean_max − actual_high)",
        xaxis_title="Error (°F)",
        yaxis_title="Count",
        height=320,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Results table builder
# ---------------------------------------------------------------------------


def _build_results_df(results: list) -> pd.DataFrame:
    rows = []
    for r in results:
        sigma_str = ", ".join(f"{k}: {v:.3f}" for k, v in r.sigma_used.items())
        theta_str = ", ".join(f"{k}: {v:.3f}" for k, v in r.theta_used.items())
        brier_mean = float(np.mean(list(r.brier_components.values()))) if r.brier_components else float("nan")
        rows.append({
            "Date": str(r.target_date),
            "Hour ET": r.eval_hour,
            "Predicted (°F)": round(r.mean_max, 1),
            "Actual (°F)": round(r.actual_high, 1),
            "Error (°F)": round(r.prediction_error, 1),
            "Brier": round(brier_mean, 4),
            "Sigma": sigma_str,
            "Theta": theta_str,
            "Bias (°F)": round(r.bias_used, 2),
            "Drift (°F)": round(r.drift_used, 2),
            "Attractor Peak (°F)": round(r.attractor_peak, 1),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------


def render_model_lab() -> None:
    """Render the Model Lab tab (Phase L1: Replay mode)."""
    from kalshi_weather_trader.backtesting.metrics import compute_aggregate_metrics
    from kalshi_weather_trader.backtesting.scenarios import PRESET_MAP, Scenario

    st.header("Model Lab")
    st.caption(
        "Backtest model configurations against settled historical dates. "
        "Select a scenario, choose a date range, and click **Run Replay**."
    )

    col_cfg, col_res = st.columns([0.30, 0.70])

    # ------------------------------------------------------------------
    # Config panel
    # ------------------------------------------------------------------
    with col_cfg:
        st.subheader("Configuration")

        # Phase L1: Replay only; mode selector is a placeholder for L2/L3
        mode = st.radio(
            "Mode",
            ["Replay"],
            help="Compare and Optimize modes are coming in Phase L2/L3.",
        )

        preset_name = st.selectbox(
            "Scenario",
            list(PRESET_MAP.keys()),
            help="Choose a preset parameter configuration to replay.",
        )
        scenario: Scenario = PRESET_MAP[preset_name]

        st.divider()

        # Date range
        today = _today_et()
        default_end = today - timedelta(days=1)
        default_start = default_end - timedelta(days=30)

        date_range = st.date_input(
            "Date range",
            value=(default_start, default_end),
            max_value=default_end,
            help="Replay will include all settled dates in this range.",
        )
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            start_date, end_date = date_range[0], date_range[1]
        else:
            start_date = end_date = date_range if isinstance(date_range, date) else default_start

        # Eval hours
        eval_hours = st.multiselect(
            "Eval hours (ET)",
            options=[6, 8, 10, 12, 14, 16, 18],
            default=[8, 10, 12, 14, 16],
            help="Hours at which to anchor the MC simulation on each date.",
        )
        if not eval_hours:
            eval_hours = [10, 14]

        scenario = replace(scenario, eval_hours=sorted(eval_hours))

        st.divider()
        st.caption(f"**Scenario:** {scenario.name}")
        st.caption(
            f"Drift in attractor: {'✓' if scenario.use_drift_in_attractor else '✗'}  |  "
            f"Anchor offset: {'✓' if scenario.use_anchor_offset else '✗'}  |  "
            f"Time-varying σ: {'✓' if scenario.use_time_varying_sigma else '✗'}"
        )
        if scenario.ou_max_stationary_std_override is not None:
            st.caption(f"σ cap override: {scenario.ou_max_stationary_std_override}°F")
        if scenario.kalman_bias_override is not None:
            st.caption(f"Bias override: {scenario.kalman_bias_override:+.2f}°F")

        run_clicked = st.button("▶ Run Replay", type="primary", use_container_width=True)

    # ------------------------------------------------------------------
    # Results panel
    # ------------------------------------------------------------------
    with col_res:
        if not run_clicked and "lab_results" not in st.session_state:
            st.info("Configure a scenario and click **Run Replay** to begin.")
            return

        if run_clicked:
            with st.spinner("Loading settled dates and running replay…"):
                try:
                    results = _run_replay_cached(
                        hash(scenario),
                        scenario,
                        str(start_date),
                        str(end_date),
                    )
                    st.session_state["lab_results"] = results
                    st.session_state["lab_scenario_name"] = scenario.name
                except Exception as exc:
                    st.error(f"Replay failed: {exc}")
                    return

        results = st.session_state.get("lab_results", [])
        scenario_name_shown = st.session_state.get("lab_scenario_name", scenario.name)

        if not results:
            st.warning(
                "No settled dates found in the selected range. "
                "Ensure the DB has markets with `cli_settlement_confirmed=True`."
            )
            return

        metrics = compute_aggregate_metrics(results)

        # ------------------------------------------------------------------
        # Section 1 — Summary metrics
        # ------------------------------------------------------------------
        st.subheader(f"Results — {scenario_name_shown}")
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("Brier Score", f"{metrics.get('brier_score', float('nan')):.4f}")
        mc2.metric("RMSE", f"{metrics.get('rmse', float('nan')):.2f}°F")
        mc3.metric("Mean Bias", f"{metrics.get('mean_bias', float('nan')):+.2f}°F")
        mc4.metric("Sharpness", f"{metrics.get('sharpness', float('nan')):.3f}")
        mc5.metric("Dates", str(metrics.get("n_dates", 0)))

        st.divider()

        # ------------------------------------------------------------------
        # Section 2 — Calibration curve + per-date scatter
        # ------------------------------------------------------------------
        ch1, ch2 = st.columns(2)
        with ch1:
            st.plotly_chart(
                _calibration_chart(metrics.get("calibration_curve", [])),
                use_container_width=True,
            )
        with ch2:
            st.plotly_chart(
                _scatter_chart(results),
                use_container_width=True,
            )

        # ------------------------------------------------------------------
        # Section 3 — Per-date results table
        # ------------------------------------------------------------------
        st.subheader("Per-Date Results")
        df = _build_results_df(results)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Error (°F)": st.column_config.NumberColumn(format="%+.1f"),
                "Brier": st.column_config.NumberColumn(format="%.4f"),
            },
        )

        # ------------------------------------------------------------------
        # Section 4 — Per-hour Brier + error distribution
        # ------------------------------------------------------------------
        bh1, bh2 = st.columns(2)
        with bh1:
            st.plotly_chart(
                _per_hour_brier_chart(metrics.get("per_hour_brier", {})),
                use_container_width=True,
            )
        with bh2:
            st.plotly_chart(
                _error_histogram(results),
                use_container_width=True,
            )

        # ------------------------------------------------------------------
        # Section 5 — Diagnostics expander
        # ------------------------------------------------------------------
        with st.expander("Diagnostics"):
            st.json({
                "n_predictions": metrics.get("n_predictions", 0),
                "log_loss": round(metrics.get("log_loss", float("nan")), 4),
                "per_hour_brier": {
                    str(k): round(v, 4)
                    for k, v in metrics.get("per_hour_brier", {}).items()
                },
                "per_hour_rmse": {
                    str(k): round(v, 3)
                    for k, v in metrics.get("per_hour_rmse", {}).items()
                },
                "per_hour_bias": {
                    str(k): round(v, 3)
                    for k, v in metrics.get("per_hour_bias", {}).items()
                },
            })
