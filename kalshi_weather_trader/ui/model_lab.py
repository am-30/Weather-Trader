"""
Model Lab — Tab 6 of the Kalshi Weather Trader Streamlit app.

Phase L1: Replay mode — single scenario backtest over settled dates.
Phase L2: Compare mode + Custom slider panel.
  - Compare mode: run two scenarios (A vs B), paired bootstrap significance test,
    dual calibration curves, per-date and per-hour comparison charts.
  - Custom slider panel: build any Scenario from structural toggles + parameter
    sliders without touching code. Available in both Replay and Compare modes.
Phase L3: Sweep mode — vary one parameter across N values, see the full Brier
  score landscape, and identify which deviations are statistically significant
  versus the production baseline.

Results are cached by (scenario hash, date range) so that a second run with
the same settings is instant.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
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
# Phase L3: Sweep mode parameter registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SweepParam:
    """Metadata for one sweepable Scenario field."""
    field_name: str       # Scenario attribute name
    default_min: float
    default_max: float
    step: float           # suggested UI step


SWEEP_PARAMS: dict[str, _SweepParam] = {
    "σ cap (ou_max_stationary_std)": _SweepParam("ou_max_stationary_std_override", 0.5, 3.0, 0.1),
    "σ scalar":                      _SweepParam("sigma_override",                 0.2, 2.0, 0.1),
    "θ scalar":                      _SweepParam("theta_override",                 0.05, 1.0, 0.05),
    "θ AM":                          _SweepParam("theta_am_override",              0.05, 1.0, 0.05),
    "θ PM":                          _SweepParam("theta_pm_override",              0.05, 1.0, 0.05),
    "Anchor weight multiplier":      _SweepParam("anchor_weight_multiplier",       0.0, 2.0, 0.1),
    "Kalman bias (°F)":              _SweepParam("kalman_bias_override",           -15.0, 15.0, 0.5),
    "Daily-max bias (°F)":          _SweepParam("daily_max_bias_override",          -3.0,  3.0, 0.25),
    "Drift AM (°F)":                 _SweepParam("drift_am_override",              -1.5, 1.5, 0.2),
    "Drift PM (°F)":                 _SweepParam("drift_pm_override",              -1.5, 1.5, 0.2),
    "Persistence offset (°F)":       _SweepParam("persistence_filter_offset_override", 0.0, 1.5, 0.1),
}


def _make_sweep_scenario(base, param_cfg: _SweepParam, value: float):
    """Clone base scenario with exactly one field overridden to value."""
    return replace(base, **{param_cfg.field_name: round(float(value), 6)})


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
        scenario_hash: Hash of scenario (cache key; not used inside).
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
# Chart helpers — Replay mode
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
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color="gray", dash="dash", width=1),
        name="Perfect calibration",
        hoverinfo="skip",
    ))
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
# Chart helpers — Compare mode
# ---------------------------------------------------------------------------


def _dual_calibration_chart(
    cal_a: list,
    cal_b: list,
    name_a: str,
    name_b: str,
) -> go.Figure:
    """Single Plotly chart with calibration curves for two scenarios overlaid."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color="gray", dash="dash", width=1),
        name="Perfect calibration",
        hoverinfo="skip",
    ))
    for cal, name, color in [(cal_a, name_a, "steelblue"), (cal_b, name_b, "tomato")]:
        if not cal:
            continue
        x = [b["mean_pred"] for b in cal]
        y = [b["observed_freq"] for b in cal]
        counts = [b["count"] for b in cal]
        sizes = [max(6, min(20, c / 5)) for c in counts]
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode="markers+lines",
            marker=dict(size=sizes, color=color, opacity=0.85),
            text=[f"n={c}" for c in counts],
            hovertemplate=f"[{name}] Predicted: %{{x:.2f}}<br>Observed: %{{y:.2f}}<br>%{{text}}<extra></extra>",
            name=name,
        ))
    fig.update_layout(
        title="Calibration Curves — A vs B",
        xaxis_title="Predicted probability",
        yaxis_title="Observed frequency",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        height=380,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def _comparison_per_hour_chart(
    a_per_hour: dict,
    b_per_hour: dict,
    name_a: str,
    name_b: str,
) -> go.Figure:
    """Grouped bar chart of per-hour Brier scores for two scenarios."""
    hours = sorted(set(a_per_hour) | set(b_per_hour))
    if not hours:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"{h}h ET" for h in hours],
        y=[a_per_hour.get(h) for h in hours],
        name=name_a,
        marker_color="steelblue",
    ))
    fig.add_trace(go.Bar(
        x=[f"{h}h ET" for h in hours],
        y=[b_per_hour.get(h) for h in hours],
        name=name_b,
        marker_color="tomato",
    ))
    fig.update_layout(
        barmode="group",
        title="Brier Score by Eval Hour — A vs B",
        xaxis_title="Eval hour (ET)",
        yaxis_title="Mean Brier score",
        height=320,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def _build_comparison_df(a_results: list, b_results: list) -> pd.DataFrame:
    """Build a per-(date, hour) comparison table from two scenario result lists."""
    a_dict = {(r.target_date, r.eval_hour): r for r in a_results}
    b_dict = {(r.target_date, r.eval_hour): r for r in b_results}
    keys = sorted(set(a_dict) & set(b_dict))

    rows = []
    for d, h in keys:
        ra = a_dict[(d, h)]
        rb = b_dict[(d, h)]
        ba = float(np.mean(list(ra.brier_components.values()))) if ra.brier_components else float("nan")
        bb = float(np.mean(list(rb.brier_components.values()))) if rb.brier_components else float("nan")
        rows.append({
            "Date": str(d),
            "Hour ET": h,
            "Actual (°F)": round(ra.actual_high, 1),
            "Pred A (°F)": round(ra.mean_max, 1),
            "Pred B (°F)": round(rb.mean_max, 1),
            "Err A (°F)": round(ra.prediction_error, 1),
            "Err B (°F)": round(rb.prediction_error, 1),
            "Brier A": round(ba, 4),
            "Brier B": round(bb, 4),
            "Δ (A−B)": round(ba - bb, 4),
            "Better": "A" if ba < bb else ("B" if bb < ba else "Tie"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Chart helpers — Sweep mode (Phase L3)
# ---------------------------------------------------------------------------


def _sweep_brier_chart(
    sweep_points: list,
    prod_value,
    prod_brier: float,
    param_label: str,
) -> go.Figure:
    """Plotly line chart of Brier score across a parameter sweep.

    Args:
        sweep_points:  list of (param_value, metrics_dict, bootstrap_or_None)
        prod_value:    Current production value for this parameter (float or None).
                       None means the production scenario uses "historical calibrated".
        prod_brier:    Mean Brier score of the production baseline.
        param_label:   Human-readable parameter name (x-axis title).

    Returns:
        Plotly Figure with significance markers and production reference lines.
    """
    if not sweep_points:
        return go.Figure()

    xs = [pt[0] for pt in sweep_points]
    ys = [pt[1].get("brier_score", float("nan")) for pt in sweep_points]

    # Colour each marker by significance vs production
    colors = []
    for _, _, boot in sweep_points:
        if boot is not None and boot.is_significant:
            colors.append("green" if boot.mean_diff < 0 else "red")
        else:
            colors.append("steelblue")

    hover_texts = []
    for val, metrics, boot in sweep_points:
        brier = metrics.get("brier_score", float("nan"))
        rmse  = metrics.get("rmse", float("nan"))
        bias  = metrics.get("mean_bias", float("nan"))
        text  = (
            f"{param_label}: {val:.4g}<br>"
            f"Brier: {brier:.4f}<br>"
            f"RMSE: {rmse:.2f}°F<br>"
            f"Bias: {bias:+.2f}°F"
        )
        if boot is not None:
            sig_str = "★ significant" if boot.is_significant else "ns"
            text += f"<br>vs production: Δ={boot.mean_diff:+.4f}, p={boot.p_value:.3f} ({sig_str})"
        hover_texts.append(text)

    fig = go.Figure()

    # Horizontal reference: production Brier
    fig.add_hline(
        y=prod_brier,
        line_dash="dash",
        line_color="tomato",
        annotation_text=f"Production Brier: {prod_brier:.4f}",
        annotation_position="bottom right",
    )

    # Vertical reference: production parameter value (if known)
    if prod_value is not None and xs and xs[0] <= prod_value <= xs[-1]:
        fig.add_vline(
            x=prod_value,
            line_dash="dot",
            line_color="tomato",
            annotation_text=f"Production: {prod_value:.4g}",
            annotation_position="top right",
        )

    # Sweep curve
    fig.add_trace(go.Scatter(
        x=xs,
        y=ys,
        mode="lines+markers",
        line=dict(color="steelblue", width=2),
        marker=dict(color=colors, size=10, line=dict(color="white", width=1)),
        text=hover_texts,
        hovertemplate="%{text}<extra></extra>",
        name="Sweep",
    ))

    # Legend annotations for marker colours
    for color, label in [("green", "Sig. better"), ("red", "Sig. worse"), ("steelblue", "Not sig.")]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(color=color, size=10),
            name=label,
            showlegend=True,
        ))

    fig.update_layout(
        title=f"Brier Score vs {param_label}",
        xaxis_title=param_label,
        yaxis_title="Mean Brier score",
        height=420,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _sweep_secondary_chart(
    sweep_points: list,
    metric_key: str,
    metric_label: str,
    param_label: str,
    prod_metric: float,
    fmt: str = ".3f",
) -> go.Figure:
    """Simple line chart for a secondary metric (RMSE or Mean Bias) across the sweep."""
    if not sweep_points:
        return go.Figure()

    xs = [pt[0] for pt in sweep_points]
    ys = [pt[1].get(metric_key, float("nan")) for pt in sweep_points]

    fig = go.Figure()
    fig.add_hline(
        y=prod_metric,
        line_dash="dash",
        line_color="tomato",
        annotation_text=f"Production: {prod_metric:{fmt}}",
        annotation_position="bottom right",
    )
    fig.add_trace(go.Scatter(
        x=xs,
        y=ys,
        mode="lines+markers",
        line=dict(color="steelblue", width=2),
        marker=dict(color="steelblue", size=8),
        hovertemplate=f"{param_label}: %{{x:.4g}}<br>{metric_label}: %{{y:{fmt}}}<extra></extra>",
        name=metric_label,
    ))
    fig.update_layout(
        title=f"{metric_label} vs {param_label}",
        xaxis_title=param_label,
        yaxis_title=metric_label,
        height=300,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def _sweep_metrics_table(
    sweep_points: list,
    param_label: str,
    prod_brier: float,
) -> pd.DataFrame:
    """Build a DataFrame summarising Brier, RMSE, Bias for each sweep point.

    Args:
        sweep_points: list of (value, metrics_dict, bootstrap_or_None)
        param_label:  Human-readable name of the swept parameter.
        prod_brier:   Production baseline Brier (for Δ column).
    """
    if not sweep_points:
        return pd.DataFrame()

    best_brier = min(
        pt[1].get("brier_score", float("nan"))
        for pt in sweep_points
        if not np.isnan(pt[1].get("brier_score", float("nan")))
    )

    rows = []
    for val, metrics, boot in sweep_points:
        brier = metrics.get("brier_score", float("nan"))
        rmse  = metrics.get("rmse", float("nan"))
        bias  = metrics.get("mean_bias", float("nan"))
        delta = brier - prod_brier

        sig_str = ""
        p_str   = "—"
        if boot is not None:
            p_str = f"{boot.p_value:.3f}"
            if boot.is_significant:
                sig_str = "Better" if boot.mean_diff < 0 else "Worse"

        best_marker = " ★" if not np.isnan(brier) and abs(brier - best_brier) < 1e-9 else ""

        rows.append({
            param_label:  round(val, 4),
            "Brier":      round(brier, 4) if not np.isnan(brier) else float("nan"),
            "RMSE (°F)":  round(rmse,  2) if not np.isnan(rmse)  else float("nan"),
            "Bias (°F)":  round(bias,  2) if not np.isnan(bias)  else float("nan"),
            "Δ Brier":    round(delta, 4) if not np.isnan(delta) else float("nan"),
            "p-value":    p_str,
            "vs Prod":    sig_str + best_marker,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Results table builder — Replay mode
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
# Custom scenario builder (Phase L2)
# ---------------------------------------------------------------------------


def _custom_scenario_panel(key_prefix: str, default_name: str = "Custom"):
    """Render structural toggles + parameter sliders, return a Scenario.

    Conditional sliders — drift AM/PM only when drift is enabled; cloud/ensemble
    params only when their respective toggles are on. This resolves L1 Deviation D2
    (cloud/ensemble sigma factors were deferred from L1).

    Args:
        key_prefix:   Unique string prefix for all Streamlit widget keys.
                      Pass "replay", "cmp_a", or "cmp_b" to avoid collisions.
        default_name: Name embedded in the returned Scenario.

    Returns:
        Scenario built from current widget values.
    """
    from kalshi_weather_trader.backtesting.scenarios import Scenario

    st.markdown("**Structural toggles**")
    c1, c2 = st.columns(2)
    use_drift    = c1.checkbox("Drift in attractor",    value=False, key=f"{key_prefix}_drift")
    use_anchor   = c2.checkbox("Anchor offset",          value=True,  key=f"{key_prefix}_anchor")
    use_tv_sigma = c1.checkbox("Time-varying σ",         value=True,  key=f"{key_prefix}_tv_sigma")
    use_tv_theta = c2.checkbox("Time-varying θ",         value=True,  key=f"{key_prefix}_tv_theta")
    use_cloud    = c1.checkbox("Cloud cover adj.",       value=False, key=f"{key_prefix}_cloud")
    use_ensemble = c2.checkbox("Ensemble spread adj.",   value=False, key=f"{key_prefix}_ensemble")
    use_pers     = c1.checkbox("Persistence offset",    value=True,  key=f"{key_prefix}_pers")

    # Scalar override sentinels
    sigma_val = theta_val = theta_am = theta_pm = None
    cap_val = pers_val = bias_val = drift_am = drift_pm = None
    anchor_mult = 1.0
    cloud_ov = 0.8; cloud_cl = 1.1
    ens_thr = 3.0; ens_fac = 1.3
    model_weights_override = None

    with st.expander("Parameter overrides (None = use historical calibrated)"):
        if st.checkbox("Override σ (scalar)", key=f"{key_prefix}_sigma_en"):
            sigma_val = st.slider("σ (°F/√hr)", 0.20, 2.50, 0.80, 0.05, key=f"{key_prefix}_sigma")

        if st.checkbox("Override σ cap (ou_max_stationary_std)", key=f"{key_prefix}_cap_en"):
            cap_val = st.slider("σ cap (°F)", 0.5, 4.0, 2.0, 0.25, key=f"{key_prefix}_cap")

        if st.checkbox("Override θ (scalar)", key=f"{key_prefix}_theta_en"):
            theta_val = st.slider("θ (per hr)", 0.05, 1.00, 0.30, 0.05, key=f"{key_prefix}_theta")

        if use_tv_theta:
            if st.checkbox("Override θ AM", key=f"{key_prefix}_theta_am_en"):
                theta_am = st.slider("θ AM", 0.05, 1.00, 0.30, 0.05, key=f"{key_prefix}_theta_am")
            if st.checkbox("Override θ PM", key=f"{key_prefix}_theta_pm_en"):
                theta_pm = st.slider("θ PM", 0.05, 1.00, 0.30, 0.05, key=f"{key_prefix}_theta_pm")

        if use_pers and st.checkbox("Override persistence offset", key=f"{key_prefix}_pers_en"):
            pers_val = st.slider("Persistence offset", 0.0, 1.5, 0.30, 0.10, key=f"{key_prefix}_pers_val")

        anchor_mult = st.slider(
            "Anchor weight multiplier (0 = off, 1 = normal)",
            0.0, 2.0, 1.0, 0.1,
            key=f"{key_prefix}_anchor_mult",
        )

        if st.checkbox("Override Kalman bias", key=f"{key_prefix}_bias_en"):
            bias_val = st.slider("Bias (°F)", -15.0, 15.0, 0.0, 0.25, key=f"{key_prefix}_bias")

        if use_drift:
            if st.checkbox("Override drift AM", key=f"{key_prefix}_drift_am_en"):
                drift_am = st.slider("Drift AM (°F)", -2.0, 3.0, 0.0, 0.25, key=f"{key_prefix}_drift_am")
            if st.checkbox("Override drift PM", key=f"{key_prefix}_drift_pm_en"):
                drift_pm = st.slider("Drift PM (°F)", -2.0, 3.0, 0.0, 0.25, key=f"{key_prefix}_drift_pm")

        if use_cloud:
            cloud_ov = st.slider("Cloud σ factor (overcast >80%)", 0.50, 1.00, 0.80, 0.05, key=f"{key_prefix}_cloud_ov")
            cloud_cl = st.slider("Cloud σ factor (clear <20%)",    1.00, 1.50, 1.10, 0.05, key=f"{key_prefix}_cloud_cl")

        if use_ensemble:
            ens_thr = st.slider("Ensemble spread threshold (°F)", 1.0, 6.0, 3.0, 0.5,  key=f"{key_prefix}_ens_thr")
            ens_fac = st.slider("Ensemble σ factor",               1.0, 2.0, 1.3, 0.10, key=f"{key_prefix}_ens_fac")

        if st.checkbox("Override model weights", key=f"{key_prefix}_mw_en"):
            st.caption("Weights are normalised automatically — only the ratios matter.")
            w_hrrr = st.slider("HRRR weight", 0.0, 1.0, 0.50, 0.05, key=f"{key_prefix}_mw_hrrr")
            w_gfs  = st.slider("GFS weight",  0.0, 1.0, 0.30, 0.05, key=f"{key_prefix}_mw_gfs")
            w_ecmwf = st.slider("ECMWF weight", 0.0, 1.0, 0.20, 0.05, key=f"{key_prefix}_mw_ecmwf")
            total = w_hrrr + w_gfs + w_ecmwf
            if total > 0:
                model_weights_override = {
                    "HRRR": round(w_hrrr / total, 6),
                    "GFS": round(w_gfs / total, 6),
                    "ECMWF": round(w_ecmwf / total, 6),
                }
                st.caption(f"Normalised: HRRR {model_weights_override['HRRR']:.3f} / GFS {model_weights_override['GFS']:.3f} / ECMWF {model_weights_override['ECMWF']:.3f}")
            else:
                st.warning("At least one model weight must be > 0.")

    return Scenario(
        name=default_name,
        use_drift_in_attractor=use_drift,
        use_anchor_offset=use_anchor,
        use_time_varying_sigma=use_tv_sigma,
        use_time_varying_theta=use_tv_theta,
        use_cloud_cover_adjustment=use_cloud,
        use_ensemble_spread_adjustment=use_ensemble,
        use_persistence_offset=use_pers,
        sigma_override=sigma_val,
        ou_max_stationary_std_override=cap_val,
        theta_override=theta_val,
        theta_am_override=theta_am,
        theta_pm_override=theta_pm,
        persistence_filter_offset_override=pers_val,
        anchor_weight_multiplier=anchor_mult,
        kalman_bias_override=bias_val,
        drift_am_override=drift_am,
        drift_pm_override=drift_pm,
        model_weights_override=model_weights_override,
        cloud_cover_overcast_sigma_factor=cloud_ov,
        cloud_cover_clear_sigma_factor=cloud_cl,
        ensemble_spread_threshold=ens_thr,
        ensemble_spread_sigma_factor=ens_fac,
    )


def _scenario_selector(label: str, key_prefix: str):
    """Selectbox with all presets + 'Custom...' option.

    When 'Custom...' is selected, renders the full slider panel below.
    Returns a Scenario object.
    """
    from kalshi_weather_trader.backtesting.scenarios import PRESET_MAP

    options = list(PRESET_MAP.keys()) + ["Custom..."]
    chosen = st.selectbox(label, options, key=f"{key_prefix}_preset")
    if chosen == "Custom...":
        return _custom_scenario_panel(key_prefix, default_name=f"Custom ({key_prefix})")
    return PRESET_MAP[chosen]


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------


def render_model_lab() -> None:
    """Render the Model Lab tab (Phases L1–L3: Replay, Compare, Sweep)."""
    from kalshi_weather_trader.backtesting.metrics import (
        compute_aggregate_metrics,
        compute_paired_bootstrap,
    )

    st.header("Model Lab")
    st.caption(
        "Backtest model configurations against settled historical dates. "
        "Select a scenario, choose a date range, and run."
    )

    col_cfg, col_res = st.columns([0.30, 0.70])

    # ------------------------------------------------------------------
    # Shared date range / eval hour widgets (used by both modes)
    # ------------------------------------------------------------------
    today = _today_et()
    default_end   = today - timedelta(days=1)
    default_start = default_end - timedelta(days=30)

    # ------------------------------------------------------------------
    # Config panel
    # ------------------------------------------------------------------
    with col_cfg:
        st.subheader("Configuration")

        mode = st.radio("Mode", ["Replay", "Compare", "Sweep"], horizontal=True)

        st.divider()

        if mode == "Replay":
            scenario = _scenario_selector("Scenario", "replay")
            st.divider()

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

            eval_hours = st.multiselect(
                "Eval hours (ET)",
                options=[6, 8, 10, 12, 14, 16, 18],
                default=[8, 10, 12, 14, 16],
            )
            if not eval_hours:
                eval_hours = [10, 14]
            scenario = replace(scenario, eval_hours=sorted(eval_hours))

            replay_kalman = st.checkbox(
                "Replay Kalman bias",
                value=False,
                key="replay_kalman_bias",
                help=(
                    "Re-run the current Kalman filter (H=[[1,1]], bias decay, "
                    "covariance cap) over historical ASOS readings instead of using "
                    "the stored kalman_bias_estimate. Corrects for dates whose stored "
                    "bias was written by a pre-Phase-A or pre-Phase-C filter."
                ),
            )
            scenario = replace(scenario, replay_kalman_bias=replay_kalman)

            st.divider()
            st.caption(f"**Scenario:** {scenario.name}")
            st.caption(
                f"Drift: {'✓' if scenario.use_drift_in_attractor else '✗'}  |  "
                f"Anchor: {'✓' if scenario.use_anchor_offset else '✗'}  |  "
                f"TV-σ: {'✓' if scenario.use_time_varying_sigma else '✗'}  |  "
                f"Kalman replay: {'✓' if scenario.replay_kalman_bias else '✗'}"
            )

            run_clicked = st.button("▶ Run Replay", type="primary", use_container_width=True)

        elif mode == "Compare":  # Compare mode
            scenario_a = _scenario_selector("Scenario A", "cmp_a")
            st.divider()
            scenario_b = _scenario_selector("Scenario B", "cmp_b")
            st.divider()

            date_range = st.date_input(
                "Date range",
                value=(default_start, default_end),
                max_value=default_end,
                key="cmp_date_range",
                help="Both scenarios will be run over the same settled dates.",
            )
            if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
                start_date, end_date = date_range[0], date_range[1]
            else:
                start_date = end_date = date_range if isinstance(date_range, date) else default_start

            eval_hours = st.multiselect(
                "Eval hours (ET)",
                options=[6, 8, 10, 12, 14, 16, 18],
                default=[8, 10, 12, 14, 16],
                key="cmp_eval_hours",
            )
            if not eval_hours:
                eval_hours = [10, 14]
            scenario_a = replace(scenario_a, eval_hours=sorted(eval_hours))
            scenario_b = replace(scenario_b, eval_hours=sorted(eval_hours))

            cmp_replay_kalman = st.checkbox(
                "Replay Kalman bias (both scenarios)",
                value=False,
                key="cmp_replay_kalman_bias",
                help=(
                    "Re-run the current Kalman filter over historical ASOS readings "
                    "for both scenarios instead of using stored kalman_bias_estimate. "
                    "Corrects for pre-Phase-A / pre-Phase-C stored states."
                ),
            )
            scenario_a = replace(scenario_a, replay_kalman_bias=cmp_replay_kalman)
            scenario_b = replace(scenario_b, replay_kalman_bias=cmp_replay_kalman)

            st.divider()
            st.caption(f"**A:** {scenario_a.name}")
            st.caption(f"**B:** {scenario_b.name}")

            run_clicked = st.button("▶ Run Comparison", type="primary", use_container_width=True)

        else:  # Sweep mode
            from kalshi_weather_trader.backtesting.scenarios import preset_production

            sweep_base = _scenario_selector("Base scenario", "sweep_base")
            st.divider()

            date_range = st.date_input(
                "Date range",
                value=(default_start, default_end),
                max_value=default_end,
                key="sweep_date_range",
                help="All settled dates in this range will be used for every sweep point.",
            )
            if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
                start_date, end_date = date_range[0], date_range[1]
            else:
                start_date = end_date = date_range if isinstance(date_range, date) else default_start

            eval_hours = st.multiselect(
                "Eval hours (ET)",
                options=[6, 8, 10, 12, 14, 16, 18],
                default=[8, 10, 12, 14, 16],
                key="sweep_eval_hours",
            )
            if not eval_hours:
                eval_hours = [10, 14]

            st.divider()
            param_label = st.selectbox(
                "Parameter to sweep",
                list(SWEEP_PARAMS.keys()),
                key="sweep_param",
            )
            param_cfg = SWEEP_PARAMS[param_label]

            sc1, sc2 = st.columns(2)
            sweep_min = sc1.number_input(
                "Min", value=param_cfg.default_min, step=param_cfg.step, key="sweep_min"
            )
            sweep_max = sc2.number_input(
                "Max", value=param_cfg.default_max, step=param_cfg.step, key="sweep_max"
            )
            step_count = st.slider("Steps", min_value=5, max_value=20, value=10, key="sweep_steps")

            sweep_values = np.linspace(float(sweep_min), float(sweep_max), int(step_count))
            st.caption(
                "Values: " + ", ".join(f"{v:.4g}" for v in sweep_values)
            )

            # Production baseline scenario (always run as reference)
            prod_scenario = replace(
                preset_production(),
                eval_hours=sorted(eval_hours),
            )

            st.divider()
            run_clicked = st.button("▶ Run Sweep", type="primary", use_container_width=True)

    # ------------------------------------------------------------------
    # Results panel
    # ------------------------------------------------------------------
    with col_res:

        # ==============================================================
        # REPLAY MODE RESULTS
        # ==============================================================
        if mode == "Replay":
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

            # Summary metrics
            st.subheader(f"Results — {scenario_name_shown}")
            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            mc1.metric("Brier Score", f"{metrics.get('brier_score', float('nan')):.4f}")
            mc2.metric("RMSE", f"{metrics.get('rmse', float('nan')):.2f}°F")
            mc3.metric("Mean Bias", f"{metrics.get('mean_bias', float('nan')):+.2f}°F")
            mc4.metric("Sharpness", f"{metrics.get('sharpness', float('nan')):.3f}")
            mc5.metric("Dates", str(metrics.get("n_dates", 0)))

            st.divider()

            ch1, ch2 = st.columns(2)
            with ch1:
                st.plotly_chart(
                    _calibration_chart(metrics.get("calibration_curve", [])),
                    use_container_width=True,
                )
            with ch2:
                st.plotly_chart(_scatter_chart(results), use_container_width=True)

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

            bh1, bh2 = st.columns(2)
            with bh1:
                st.plotly_chart(
                    _per_hour_brier_chart(metrics.get("per_hour_brier", {})),
                    use_container_width=True,
                )
            with bh2:
                st.plotly_chart(_error_histogram(results), use_container_width=True)

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

        # ==============================================================
        # COMPARE MODE RESULTS
        # ==============================================================
        elif mode == "Compare":
            if not run_clicked and "lab_cmp_a_results" not in st.session_state:
                st.info(
                    "Select Scenario A and Scenario B, configure a date range, "
                    "and click **Run Comparison** to begin."
                )
                return

            if run_clicked:
                with st.spinner("Running both scenarios…"):
                    try:
                        a_results = _run_replay_cached(
                            hash(scenario_a), scenario_a, str(start_date), str(end_date)
                        )
                        b_results = _run_replay_cached(
                            hash(scenario_b), scenario_b, str(start_date), str(end_date)
                        )
                        st.session_state["lab_cmp_a_results"] = a_results
                        st.session_state["lab_cmp_b_results"] = b_results
                        st.session_state["lab_cmp_name_a"] = scenario_a.name
                        st.session_state["lab_cmp_name_b"] = scenario_b.name
                    except Exception as exc:
                        st.error(f"Comparison failed: {exc}")
                        return

            a_results = st.session_state.get("lab_cmp_a_results", [])
            b_results = st.session_state.get("lab_cmp_b_results", [])
            name_a    = st.session_state.get("lab_cmp_name_a", "Scenario A")
            name_b    = st.session_state.get("lab_cmp_name_b", "Scenario B")

            if not a_results and not b_results:
                st.warning("No settled dates found in the selected range for either scenario.")
                return

            ma = compute_aggregate_metrics(a_results)
            mb = compute_aggregate_metrics(b_results)

            # ------------------------------------------------------------------
            # Section 1 — Head-to-head summary (3 columns: A | B | Diff)
            # ------------------------------------------------------------------
            st.subheader(f"Results — {name_a}  vs  {name_b}")

            hc1, hc2, hc3 = st.columns(3)
            metric_keys = ["brier_score", "rmse", "mean_bias", "sharpness"]
            metric_fmt  = {
                "brier_score": ("Brier Score", ".4f"),
                "rmse":        ("RMSE (°F)",   ".2f"),
                "mean_bias":   ("Mean Bias",   "+.2f"),
                "sharpness":   ("Sharpness",   ".3f"),
            }

            with hc1:
                st.markdown(f"**{name_a}**")
                for k, (label, fmt) in metric_fmt.items():
                    v = ma.get(k, float("nan"))
                    st.metric(label, format(v, fmt) if not np.isnan(v) else "—")

            with hc2:
                st.markdown(f"**{name_b}**")
                for k, (label, fmt) in metric_fmt.items():
                    v = mb.get(k, float("nan"))
                    st.metric(label, format(v, fmt) if not np.isnan(v) else "—")

            with hc3:
                st.markdown("**Difference (A − B)**")
                for k, (label, fmt) in metric_fmt.items():
                    va = ma.get(k, float("nan"))
                    vb = mb.get(k, float("nan"))
                    diff = va - vb
                    delta_fmt = "+.4f" if k == "brier_score" else "+.3f"
                    st.metric(label, format(diff, delta_fmt) if not (np.isnan(va) or np.isnan(vb)) else "—")

            # ------------------------------------------------------------------
            # Section 2 — Bootstrap significance banner
            # ------------------------------------------------------------------
            bootstrap = None
            if a_results and b_results:
                try:
                    bootstrap = compute_paired_bootstrap(a_results, b_results)
                except ValueError as exc:
                    st.warning(f"Bootstrap test skipped: {exc}")

            if bootstrap is not None:
                bs = bootstrap
                ci_str = f"CI: [{bs.ci_low:+.4f}, {bs.ci_high:+.4f}]"
                p_str  = f"p = {bs.p_value:.3f}"
                n_str  = f"n = {bs.n_shared_dates} dates"
                diff_str = f"Δ = {bs.mean_diff:+.4f}"

                if bs.is_significant:
                    direction = "WORSE" if bs.mean_diff > 0 else "BETTER"
                    msg = f"A is significantly {direction} than B  ({diff_str}, {ci_str}, {p_str}, {n_str})"
                    if bs.mean_diff > 0:
                        st.error(msg)
                    else:
                        st.success(msg)
                else:
                    msg = f"No significant difference  ({diff_str}, {ci_str}, {p_str}, {n_str})"
                    st.info(msg)

            st.divider()

            # ------------------------------------------------------------------
            # Section 3 — Dual calibration curves
            # ------------------------------------------------------------------
            st.plotly_chart(
                _dual_calibration_chart(
                    ma.get("calibration_curve", []),
                    mb.get("calibration_curve", []),
                    name_a,
                    name_b,
                ),
                use_container_width=True,
            )

            # ------------------------------------------------------------------
            # Section 4 — Per-date comparison table
            # ------------------------------------------------------------------
            st.subheader("Per-Date Comparison")
            cmp_df = _build_comparison_df(a_results, b_results)
            if not cmp_df.empty:
                st.dataframe(
                    cmp_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Err A (°F)": st.column_config.NumberColumn(format="%+.1f"),
                        "Err B (°F)": st.column_config.NumberColumn(format="%+.1f"),
                        "Δ (A−B)":    st.column_config.NumberColumn(format="%+.4f"),
                    },
                )
            else:
                st.info("No overlapping (date, hour) pairs found between scenarios.")

            # ------------------------------------------------------------------
            # Section 5 — Per-hour Brier comparison bar chart
            # ------------------------------------------------------------------
            st.plotly_chart(
                _comparison_per_hour_chart(
                    ma.get("per_hour_brier", {}),
                    mb.get("per_hour_brier", {}),
                    name_a,
                    name_b,
                ),
                use_container_width=True,
            )

        # ==============================================================
        # SWEEP MODE RESULTS
        # ==============================================================
        else:
            if not run_clicked and "lab_sweep_points" not in st.session_state:
                st.info(
                    "Choose a base scenario and a parameter to sweep, then click "
                    "**Run Sweep**.  Each point is cached independently — changing "
                    "the range re-uses any previously computed values."
                )
                return

            if run_clicked:
                if sweep_min >= sweep_max:
                    st.error("Min must be less than Max.")
                    return

                with st.spinner("Running sweep (one cached replay per step)…"):
                    try:
                        # Production baseline — always run as reference
                        prod_results = _run_replay_cached(
                            hash(prod_scenario),
                            prod_scenario,
                            str(start_date),
                            str(end_date),
                        )
                        prod_metrics = compute_aggregate_metrics(prod_results)

                        sweep_points = []
                        for val in sweep_values:
                            s = _make_sweep_scenario(
                                replace(sweep_base, eval_hours=sorted(eval_hours)),
                                param_cfg,
                                val,
                            )
                            results = _run_replay_cached(
                                hash(s), s, str(start_date), str(end_date)
                            )
                            metrics = compute_aggregate_metrics(results)
                            boot = None
                            if results and prod_results:
                                try:
                                    boot = compute_paired_bootstrap(results, prod_results)
                                except ValueError:
                                    pass
                            sweep_points.append((float(val), metrics, boot))

                        st.session_state["lab_sweep_points"]      = sweep_points
                        st.session_state["lab_sweep_prod_metrics"] = prod_metrics
                        st.session_state["lab_sweep_param_label"] = param_label
                        st.session_state["lab_sweep_param_cfg"]   = param_cfg
                    except Exception as exc:
                        st.error(f"Sweep failed: {exc}")
                        return

            sweep_points      = st.session_state.get("lab_sweep_points", [])
            prod_metrics      = st.session_state.get("lab_sweep_prod_metrics", {})
            shown_param_label = st.session_state.get("lab_sweep_param_label", param_label)
            shown_param_cfg   = st.session_state.get("lab_sweep_param_cfg", param_cfg)

            if not sweep_points:
                st.warning("No settled dates found in the selected range.")
                return

            prod_brier   = prod_metrics.get("brier_score", float("nan"))
            prod_rmse    = prod_metrics.get("rmse", float("nan"))
            prod_bias    = prod_metrics.get("mean_bias", float("nan"))

            # Current production value for this parameter (None = uses historical calibrated)
            from kalshi_weather_trader.backtesting.scenarios import preset_production as _pp
            _prod_ref_val = getattr(_pp(), shown_param_cfg.field_name, None)

            # ------------------------------------------------------------------
            # Section 1 — Brier score curve
            # ------------------------------------------------------------------
            st.subheader(f"Sweep: {shown_param_label}")
            st.caption(
                f"Production baseline — Brier: {prod_brier:.4f} | "
                f"RMSE: {prod_rmse:.2f}°F | Bias: {prod_bias:+.2f}°F | "
                f"Production value: {_prod_ref_val if _prod_ref_val is not None else 'historical calibrated'}"
            )

            st.plotly_chart(
                _sweep_brier_chart(sweep_points, _prod_ref_val, prod_brier, shown_param_label),
                use_container_width=True,
            )

            # ------------------------------------------------------------------
            # Section 2 — RMSE and Bias secondary charts
            # ------------------------------------------------------------------
            sec1, sec2 = st.columns(2)
            with sec1:
                st.plotly_chart(
                    _sweep_secondary_chart(
                        sweep_points, "rmse", "RMSE (°F)", shown_param_label, prod_rmse, ".2f"
                    ),
                    use_container_width=True,
                )
            with sec2:
                st.plotly_chart(
                    _sweep_secondary_chart(
                        sweep_points, "mean_bias", "Mean Bias (°F)", shown_param_label, prod_bias, "+.2f"
                    ),
                    use_container_width=True,
                )

            # ------------------------------------------------------------------
            # Section 3 — Metrics table + CSV export
            # ------------------------------------------------------------------
            st.subheader("Sweep Results Table")
            sweep_df = _sweep_metrics_table(sweep_points, shown_param_label, prod_brier)
            if not sweep_df.empty:
                st.dataframe(
                    sweep_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Brier":     st.column_config.NumberColumn(format="%.4f"),
                        "RMSE (°F)": st.column_config.NumberColumn(format="%.2f"),
                        "Bias (°F)": st.column_config.NumberColumn(format="%+.2f"),
                        "Δ Brier":   st.column_config.NumberColumn(format="%+.4f"),
                    },
                )
                csv_bytes = sweep_df.to_csv(index=False).encode()
                st.download_button(
                    "⬇ Download CSV",
                    data=csv_bytes,
                    file_name=f"sweep_{shown_param_label.replace(' ', '_')}.csv",
                    mime="text/csv",
                )
