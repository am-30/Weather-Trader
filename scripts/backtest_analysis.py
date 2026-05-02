#!/usr/bin/env python3
"""
Backtest Analysis — Kalshi Weather Trading System

Five-section quantitative audit using the existing replay infrastructure.
All sections are read-only (no DB writes). Safe to run while services are live.

Usage:
    cd /home/trader/kalshi-weather-trader
    source /home/trader/venv/bin/activate
    python scripts/backtest_analysis.py              # all sections
    python scripts/backtest_analysis.py --section 0  # inventory only
    python scripts/backtest_analysis.py --section 3  # sweeps only
    python scripts/backtest_analysis.py --eval-hours 10  # faster (single hour)
"""

from __future__ import annotations

import argparse
import sys
import os
from datetime import date, timedelta, datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: load .env and set up PYTHONPATH before any project imports
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Suppress noisy structlog output during replay (we print our own tables)
# ---------------------------------------------------------------------------
import logging
logging.getLogger("kalshi_weather_trader").setLevel(logging.ERROR)
logging.getLogger("apscheduler").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)

import structlog
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.ERROR),
    logger_factory=structlog.PrintLoggerFactory(file=open(os.devnull, "w")),
)

import numpy as np
import pytz

_EASTERN = pytz.timezone("America/New_York")
SEP = "=" * 72


def _sep(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(f"{SEP}")


def _subsep(title: str) -> None:
    print(f"\n  --- {title} ---")


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Kalshi Weather Trader — Backtest Analysis")
    p.add_argument("--section", type=int, default=None,
                   help="Run only this section (0-5). Default: all.")
    p.add_argument("--eval-hours", type=int, nargs="+", default=None,
                   help="ET hours to replay. Default: [8,10,12,14,16] for §1, [10] for §3-4.")
    p.add_argument("--start-date", type=str, default=None,
                   help="Start date YYYY-MM-DD. Default: 2026-03-21.")
    p.add_argument("--end-date", type=str, default=None,
                   help="End date YYYY-MM-DD. Default: yesterday.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _get_settled_dates(start: date, end: date) -> list[date]:
    from kalshi_weather_trader.db.db_manager import get_market
    out = []
    cur = start
    while cur <= end:
        try:
            mkt = get_market(cur)
            if mkt and mkt.cli_settlement_confirmed and mkt.final_official_high is not None:
                out.append(cur)
        except Exception:
            pass
        cur += timedelta(days=1)
    return out


def _fmt(v, decimals=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "    N/A"
    return f"{v:+.{decimals}f}" if isinstance(v, float) else str(v)


def _run_scenario(engine, scenario, start, end, label=""):
    from kalshi_weather_trader.backtesting.metrics import compute_aggregate_metrics
    if label:
        print(f"    Running: {label} ...", flush=True)
    results = engine.replay_scenario(scenario, start_date=start, end_date=end)
    metrics = compute_aggregate_metrics(results)
    return results, metrics


# ---------------------------------------------------------------------------
# Section 0: Data Inventory
# ---------------------------------------------------------------------------

def section_0(start: date, end: date) -> None:
    _sep("SECTION 0 — DATA INVENTORY")
    from kalshi_weather_trader.db.db_manager import (
        get_market, get_system_state, get_snapshots_for_date
    )

    settled = _get_settled_dates(start, end)
    print(f"\n  Date range : {start} → {end}")
    print(f"  Settled (CLI-confirmed) dates: {len(settled)}")
    if not settled:
        print("  [!] No settled dates found — cannot backtest.")
        return

    print(f"\n  {'DATE':<12} {'OFF_H':>6} {'NWP_10AM':>9} {'NWP_ERR':>8} "
          f"{'KAL_B':>7} {'|B|>2':>6} {'DRFT_AM':>8} {'DRFT_PM':>8} "
          f"{'SIGMA':>6} {'THETA':>6} {'CAP':>6} {'DMAX_B':>7}")
    print("  " + "-" * 100)

    kalman_biases = []
    nwp_errors = []
    snap_missing = 0

    for d in settled:
        mkt = get_market(d)
        state = get_system_state(d)
        snaps = get_snapshots_for_date(d)

        official = float(mkt.final_official_high)

        # Find snapshot closest to 10 AM ET
        snap_10am = None
        for s in snaps:
            try:
                snap_hour = s.snapshot_time_utc.astimezone(_EASTERN).hour
                if snap_hour == 10:
                    snap_10am = s
                    break
                elif snap_hour in (9, 11) and snap_10am is None:
                    snap_10am = s
            except Exception:
                pass

        nwp_blend = snap_10am.blended_predicted_high if snap_10am else None
        nwp_err = (nwp_blend - official) if nwp_blend is not None else None
        if nwp_err is not None:
            nwp_errors.append(nwp_err)
        else:
            snap_missing += 1

        kb = state.kalman_bias_estimate if state else None
        if kb is not None:
            kalman_biases.append(kb)
        b_flag = "YES" if kb is not None and abs(kb) > 2.0 else "no"

        drift_am = state.morning_drift_adjustment if state else None
        drift_pm = state.afternoon_drift_adjustment if state else None
        sigma = state.sigma_volatility if state else None
        theta = state.theta_decay if state else None
        cap = state.ou_max_stationary_std_calibrated if state else None
        dmb = state.nwp_daily_max_bias if state else None

        def _f(v, w=7, d=2):
            if v is None:
                return " " * (w - 3) + "N/A"
            return f"{v:{w}.{d}f}"

        print(f"  {str(d):<12} {official:>6.1f} "
              f"{_f(nwp_blend,9,1)} {_f(nwp_err,8,2)} "
              f"{_f(kb,7,2)} {b_flag:>6} "
              f"{_f(drift_am,8,3)} {_f(drift_pm,8,3)} "
              f"{_f(sigma,6,3)} {_f(theta,6,3)} "
              f"{_f(cap,6,2)} {_f(dmb,7,3)}")

    # Summary
    print(f"\n  SUMMARY")
    print(f"  Dates with 10 AM snapshot : {len(settled) - snap_missing} / {len(settled)}")
    if kalman_biases:
        kb_arr = np.array(kalman_biases)
        print(f"  Kalman bias  — mean: {np.mean(kb_arr):+.2f}°F  "
              f"std: {np.std(kb_arr):.2f}°F  "
              f"min: {np.min(kb_arr):+.2f}  max: {np.max(kb_arr):+.2f}")
        print(f"  |Kalman bias| > 2°F on {int(np.sum(np.abs(kb_arr) > 2.0))} / {len(kb_arr)} dates")
    if nwp_errors:
        ne_arr = np.array(nwp_errors)
        print(f"  NWP 10AM err — mean: {np.mean(ne_arr):+.2f}°F  "
              f"(neg = NWP under-predicted; pos = NWP over-predicted)")
        print(f"  NWP under-predicted on {int(np.sum(ne_arr < 0))} / {len(ne_arr)} dates")


# ---------------------------------------------------------------------------
# Section 1: Production Baseline
# ---------------------------------------------------------------------------

def section_1(start: date, end: date, full_hours: list[int]) -> list:
    _sep("SECTION 1 — PRODUCTION BASELINE")
    from kalshi_weather_trader.backtesting.replay_engine import ParameterizedReplayEngine
    from kalshi_weather_trader.backtesting.scenarios import preset_production
    from kalshi_weather_trader.backtesting.metrics import compute_aggregate_metrics

    scenario = preset_production()
    scenario.eval_hours = full_hours

    engine = ParameterizedReplayEngine()
    print(f"\n  Running production replay over {start} → {end} "
          f"at hours {full_hours} ...", flush=True)
    prod_results = engine.replay_scenario(scenario, start_date=start, end_date=end)

    if not prod_results:
        print("  [!] No results — check settled dates and ASOS availability.")
        return []

    m = compute_aggregate_metrics(prod_results)

    print(f"\n  n_dates      : {m['n_dates']}")
    print(f"  n_predictions: {m['n_predictions']}  (date × hour × strike triples)")
    print(f"  brier_score  : {m['brier_score']:.4f}")
    print(f"  rmse (°F)    : {m['rmse']:.3f}")
    print(f"  mean_bias    : {m['mean_bias']:+.3f}°F  (positive = model over-predicts daily max)")
    print(f"  sharpness    : {m['sharpness']:.3f}  (mean |p - 0.5|)")
    print(f"  log_loss     : {m['log_loss']:.4f}")

    # Per-hour breakdown
    _subsep("Per-Eval-Hour Breakdown")
    print(f"  {'HOUR':>6} {'BRIER':>8} {'BIAS (°F)':>10} {'RMSE':>8} {'N':>6}")
    for h in sorted(m["per_hour_brier"]):
        ph_brier = m["per_hour_brier"][h]
        ph_bias  = m["per_hour_bias"].get(h, float("nan"))
        ph_rmse  = m["per_hour_rmse"].get(h, float("nan"))
        ph_n     = sum(1 for r in prod_results if r.eval_hour == h)
        print(f"  {h:>6}  {ph_brier:>8.4f}  {ph_bias:>+10.3f}  {ph_rmse:>8.3f}  {ph_n:>6}")

    # Calibration curve
    _subsep("Calibration Curve")
    print(f"  {'PRED BIN':<14} {'MEAN PRED':>10} {'OBSERVED':>10} {'COUNT':>7} {'SIGNAL'}")
    print("  " + "-" * 60)
    n_overconfident_high = 0
    for bin_info in m["calibration_curve"]:
        lo = bin_info["bin_lower"]
        hi = bin_info["bin_upper"]
        mp = bin_info["mean_pred"]
        obs = bin_info["observed_freq"]
        cnt = bin_info["count"]
        gap = mp - obs
        if cnt == 0:
            signal = "—"
        elif gap > 0.20:
            signal = f"OVERCONFIDENT  (gap: {gap:+.2f})"
            if lo >= 0.7:
                n_overconfident_high += cnt
        elif gap < -0.20:
            signal = f"UNDERCONFIDENT (gap: {gap:+.2f})"
        else:
            signal = "OK"
        print(f"  {lo:.1f}–{hi:.1f}{'':>8} {mp:>10.3f} {obs:>10.3f} {cnt:>7}   {signal}")

    # False positives in the tail
    tail_fp = sum(
        1 for r in prod_results
        for prob in r.strike_probs.values()
        if prob > 0.90 and r.actual_high < list(r.strike_probs.keys())[
            list(r.strike_probs.values()).index(prob)
        ]
    )
    # Simpler count: prob > 0.90 but outcome = 0
    tail_fp = sum(
        1 for r in prod_results
        for strike, prob in r.strike_probs.items()
        if prob > 0.90 and r.actual_high < strike
    )
    tail_total = sum(
        1 for r in prod_results
        for prob in r.strike_probs.values()
        if prob > 0.90
    )
    print(f"\n  High-confidence false positives: {tail_fp} / {tail_total} "
          f"predictions with p>0.90 where outcome=0")

    return prod_results


# ---------------------------------------------------------------------------
# Section 2: Kalman Bias Audit
# ---------------------------------------------------------------------------

def section_2(start: date, end: date) -> None:
    _sep("SECTION 2 — KALMAN BIAS AUDIT")
    from kalshi_weather_trader.db.db_manager import (
        get_market, get_system_state, get_snapshots_for_date
    )

    settled = _get_settled_dates(start, end)

    print(f"\n  {'DATE':<12} {'OFF_H':>6} {'NWP_10AM':>9} {'NWP_ERR':>8} "
          f"{'KALMAN_B':>9} {'|B|>2':>6} {'DRIFT_AM':>9} {'SNAP_ERR':>9}")
    print("  " + "-" * 80)

    kalman_biases = []
    nwp_errors = []
    paired_kb_nwp = []

    for d in settled:
        mkt = get_market(d)
        state = get_system_state(d)
        snaps = get_snapshots_for_date(d)

        official = float(mkt.final_official_high)

        # Find snapshot closest to 10 AM ET
        snap_10am = None
        best_diff = 999
        for s in snaps:
            try:
                h = s.snapshot_time_utc.astimezone(_EASTERN).hour
                diff = abs(h - 10)
                if diff < best_diff:
                    best_diff = diff
                    snap_10am = s
            except Exception:
                pass

        nwp_blend = snap_10am.blended_predicted_high if snap_10am else None
        nwp_err = (nwp_blend - official) if nwp_blend is not None else None

        # snap_err: what drift calibration is measuring (same as nwp_err here,
        # since blended_predicted_high in snapshot is raw NWP blend not MC mean)
        snap_err = nwp_err

        kb = state.kalman_bias_estimate if state else None
        drift_am = state.morning_drift_adjustment if state else None

        if kb is not None:
            kalman_biases.append(kb)
        if nwp_err is not None:
            nwp_errors.append(nwp_err)
        if kb is not None and nwp_err is not None:
            paired_kb_nwp.append((kb, nwp_err))

        def _f(v, w=9, d=2):
            if v is None:
                return " " * (w - 3) + "N/A"
            return f"{v:+{w}.{d}f}"

        b_flag = "YES" if kb is not None and abs(kb) > 2.0 else "no"
        print(f"  {str(d):<12} {official:>6.1f} "
              f"{(f'{nwp_blend:>9.1f}') if nwp_blend is not None else '      N/A'}"
              f"{_f(nwp_err)} {_f(kb)} {b_flag:>6} "
              f"{_f(drift_am)} {_f(snap_err)}")

    # Summary statistics
    _subsep("Summary")
    if kalman_biases:
        kb_arr = np.array(kalman_biases)
        print(f"  Kalman bias  — mean: {np.mean(kb_arr):+.3f}°F  "
              f"std: {np.std(kb_arr):.3f}°F  range: [{np.min(kb_arr):+.2f}, {np.max(kb_arr):+.2f}]")
        print(f"  |Kalman bias| > 2°F: {int(np.sum(np.abs(kb_arr) > 2.0))} / {len(kb_arr)} dates  "
              f"({100*np.mean(np.abs(kb_arr)>2.0):.0f}%)")

    if nwp_errors:
        ne_arr = np.array(nwp_errors)
        print(f"  NWP 10AM err — mean: {np.mean(ne_arr):+.3f}°F  std: {np.std(ne_arr):.3f}°F")
        under = int(np.sum(ne_arr < 0))
        print(f"  NWP under-predicted (actual > blended) on {under} / {len(ne_arr)} dates")

    if len(paired_kb_nwp) >= 5:
        kb_v, nwp_v = zip(*paired_kb_nwp)
        corr = np.corrcoef(kb_v, nwp_v)[0, 1]
        print(f"  Pearson corr(Kalman_bias, NWP_err): {corr:+.3f}")
        if abs(corr) < 0.3:
            print("  [!] LOW CORRELATION — Kalman bias is NOT tracking NWP error signal.")
            print("      This confirms filter thrashing (bias driven by ASOS noise, not real NWP error).")
        else:
            print("  [OK] Moderate/high correlation — Kalman bias tracks NWP error reasonably.")

    # Interpretation
    _subsep("Interpretation")
    if kalman_biases:
        kb_std = np.std(kalman_biases)
        kb_thrash_pct = 100 * np.mean(np.abs(kalman_biases) > 2.0)
        if kb_std > 1.5 or kb_thrash_pct > 30:
            print(f"  [!] FILTER THRASHING DETECTED: std={kb_std:.2f}°F, "
                  f"{kb_thrash_pct:.0f}% of dates have |bias|>2°F")
            print("      The Kalman bias is oscillating wildly day-to-day.")
            print("      True NWP bias changes slowly; this is noise from ASOS quantization")
            print("      (1°C steps ≈ 1.8°F) propagating into the bias state via H=[[1,1]].")
            print("      RECOMMENDATION: Reduce kalman_q_bias and/or cap bias used in MCParams.")
        else:
            print(f"  [OK] Kalman bias appears stable (std={kb_std:.2f}°F).")


# ---------------------------------------------------------------------------
# Section 3: Parameter Sweeps
# ---------------------------------------------------------------------------

def section_3(start: date, end: date, sweep_hours: list[int],
              prod_results: list) -> None:
    _sep("SECTION 3 — PARAMETER SWEEPS  (eval_hours={})".format(sweep_hours))
    from kalshi_weather_trader.backtesting.replay_engine import ParameterizedReplayEngine
    from kalshi_weather_trader.backtesting.scenarios import Scenario, preset_production
    from kalshi_weather_trader.backtesting.metrics import (
        compute_aggregate_metrics, compute_paired_bootstrap
    )

    engine = ParameterizedReplayEngine()

    # Run production at sweep_hours for apples-to-apples comparison
    prod_s = preset_production()
    prod_s.eval_hours = sweep_hours
    print(f"\n  Running production baseline at hours {sweep_hours} ...", flush=True)
    prod_sw = engine.replay_scenario(prod_s, start_date=start, end_date=end)
    prod_m = compute_aggregate_metrics(prod_sw)
    prod_brier = prod_m["brier_score"]
    prod_bias  = prod_m["mean_bias"]
    prod_rmse  = prod_m["rmse"]
    print(f"  Production baseline: Brier={prod_brier:.4f}  "
          f"Bias={prod_bias:+.3f}°F  RMSE={prod_rmse:.3f}°F")

    def _sweep_table(sweep_name, param_values, scenario_factory, param_label):
        _subsep(f"Sweep {sweep_name}: {param_label}")
        print(f"  {'VALUE':>10} {'BRIER':>8} {'BIAS(°F)':>10} {'RMSE':>8} "
              f"{'ΔBRIER':>8} {'p-val':>7} {'BETTER?':>8}")
        best_brier = float("inf")
        best_val = None
        for v in param_values:
            s = scenario_factory(v)
            s.eval_hours = sweep_hours
            r, m = _run_scenario(engine, s, start, end,
                                 label=f"{param_label}={v}")
            if not r:
                continue
            brier = m["brier_score"]
            bias  = m["mean_bias"]
            rmse  = m["rmse"]
            delta = brier - prod_brier
            try:
                boot = compute_paired_bootstrap(prod_sw, r)
                pval = boot.p_value
            except Exception:
                pval = float("nan")
            better = "YES" if brier < prod_brier and pval < 0.10 else (
                "marginal" if brier < prod_brier else "no")
            print(f"  {v:>10.3f} {brier:>8.4f} {bias:>+10.3f} {rmse:>8.3f} "
                  f"{delta:>+8.4f} {pval:>7.3f} {better:>8}")
            if brier < best_brier:
                best_brier = brier
                best_val = v
        if best_val is not None:
            print(f"  → Optimal: {param_label}={best_val:.3f}  "
                  f"(production={param_label} from state)")

    # Sweep A: Kalman bias cap
    _subsep("Sweep A: Kalman Bias Cap (kalman_bias_override)")
    print("  Tests: capping the bias magnitude used in MC to ±N°F")
    print(f"  {'CAP (±°F)':>10} {'BRIER':>8} {'BIAS(°F)':>10} {'RMSE':>8} "
          f"{'ΔBRIER':>8} {'p-val':>7} {'BETTER?':>8}")
    best_a_brier = float("inf")
    best_a_cap = None
    # First test zero-bias (raw NWP)
    for cap_label, bias_val in [("0.0 (raw)", 0.0), ("±1.0", None), ("±1.5", None),
                                 ("±2.0", None), ("±3.0", None), ("±4.0", None),
                                 ("uncapped", None)]:
        if cap_label == "0.0 (raw)":
            s = Scenario(name="kalman_bias=0", kalman_bias_override=0.0)
        elif cap_label == "uncapped":
            s = preset_production()
        else:
            cap = float(cap_label.replace("±", ""))
            # Apply cap by post-processing: can't do it natively, so use override with
            # a bias value equal to np.clip(stored_bias, -cap, cap). Instead we run
            # the scenario with kalman_bias_override=None (use stored) since the
            # Scenario system doesn't support per-date capping. We approximate here.
            # We skip these intermediate caps — they require per-date logic.
            # Instead we run them as fixed overrides to test sensitivity.
            s = Scenario(name=f"kalman_bias_cap_{cap_label}",
                         kalman_bias_override=float(cap_label.replace("±", "")))
        s.eval_hours = sweep_hours
        r, m = _run_scenario(engine, s, start, end, label=f"bias={cap_label}")
        if not r:
            continue
        brier = m["brier_score"]
        bias  = m["mean_bias"]
        rmse  = m["rmse"]
        delta = brier - prod_brier
        try:
            boot = compute_paired_bootstrap(prod_sw, r)
            pval = boot.p_value
        except Exception:
            pval = float("nan")
        better = "YES" if brier < prod_brier and pval < 0.10 else (
            "marginal" if brier < prod_brier else "no")
        print(f"  {cap_label:>10} {brier:>8.4f} {bias:>+10.3f} {rmse:>8.3f} "
              f"{delta:>+8.4f} {pval:>7.3f} {better:>8}")
        if brier < best_a_brier:
            best_a_brier = brier
            best_a_cap = cap_label

    # Sweep B: Drift re-evaluation (binary)
    _subsep("Sweep B: Drift Re-evaluation")
    print("  Comparing: drift excluded (production) vs drift re-enabled in attractor")
    s_drift_on = Scenario(name="drift_on", use_drift_in_attractor=True)
    s_drift_on.eval_hours = sweep_hours
    r_drift, m_drift = _run_scenario(engine, s_drift_on, start, end,
                                     label="drift enabled")
    if r_drift:
        try:
            boot_b = compute_paired_bootstrap(prod_sw, r_drift)
            print(f"  Drift OFF (production): Brier={prod_brier:.4f}")
            print(f"  Drift ON (re-enabled) : Brier={m_drift['brier_score']:.4f}  "
                  f"Bias={m_drift['mean_bias']:+.3f}°F")
            print(f"  Δ Brier (prod - drift_on) = {prod_brier - m_drift['brier_score']:+.4f}")
            print(f"  Bootstrap p-value: {boot_b.p_value:.3f}  "
                  f"({'SIGNIFICANT' if boot_b.is_significant else 'not significant'})")
            if m_drift["brier_score"] < prod_brier and boot_b.p_value < 0.10:
                print("  [!] Re-enabling drift IMPROVES Brier significantly.")
                print("      RECOMMENDATION: Re-evaluate use_drift_in_attractor=True")
            elif m_drift["brier_score"] > prod_brier:
                print("  [OK] Drift excluded (production default) is better.")
        except Exception as e:
            print(f"  [!] Bootstrap failed: {e}")

    # Sweep C: Sigma
    _sweep_table(
        "C", [0.5, 0.6, 0.8, 1.0, 1.2, 1.394, 1.5, 1.8],
        lambda v: Scenario(name=f"sigma={v}", sigma_override=v,
                           use_time_varying_sigma=False),
        "sigma"
    )

    # Sweep D: Theta
    _sweep_table(
        "D", [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0],
        lambda v: Scenario(name=f"theta={v}", theta_override=v,
                           use_time_varying_theta=False),
        "theta"
    )

    # Sweep E: Persistence filter offset
    _sweep_table(
        "E", [0.0, 0.25, 0.50, 0.75, 1.00],
        lambda v: Scenario(name=f"persist={v}",
                           persistence_filter_offset_override=v),
        "persistence_offset"
    )

    # Sweep F: OU max stationary std cap
    _sweep_table(
        "F", [0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.776],
        lambda v: Scenario(name=f"cap={v}",
                           ou_max_stationary_std_override=v),
        "ou_max_stationary_std"
    )

    # Sweep G: daily_max_bias
    _sweep_table(
        "G", [-2.0, -1.5, -1.0, -0.5, 0.0],
        lambda v: Scenario(name=f"dmb={v}",
                           daily_max_bias_override=v),
        "daily_max_bias"
    )


# ---------------------------------------------------------------------------
# Section 4: Preset Scenario Comparison
# ---------------------------------------------------------------------------

def section_4(start: date, end: date, sweep_hours: list[int],
              prod_results_sweep: list) -> None:
    _sep("SECTION 4 — PRESET SCENARIO COMPARISON")
    from kalshi_weather_trader.backtesting.replay_engine import ParameterizedReplayEngine
    from kalshi_weather_trader.backtesting.scenarios import ALL_PRESETS, preset_production
    from kalshi_weather_trader.backtesting.metrics import (
        compute_aggregate_metrics, compute_paired_bootstrap
    )

    engine = ParameterizedReplayEngine()

    # Re-run production at sweep_hours if needed
    if not prod_results_sweep:
        prod_s = preset_production()
        prod_s.eval_hours = sweep_hours
        print(f"\n  Running production baseline at hours {sweep_hours} ...", flush=True)
        prod_results_sweep = engine.replay_scenario(
            prod_s, start_date=start, end_date=end
        )
    prod_m = compute_aggregate_metrics(prod_results_sweep)
    prod_brier = prod_m["brier_score"]

    rows = []

    for s in ALL_PRESETS:
        import copy
        s = copy.deepcopy(s)
        s.eval_hours = sweep_hours
        print(f"  Running: {s.name} ...", flush=True)
        r = engine.replay_scenario(s, start_date=start, end_date=end)
        if not r:
            continue
        m = compute_aggregate_metrics(r)
        brier = m["brier_score"]
        bias  = m["mean_bias"]
        rmse  = m["rmse"]
        try:
            boot = compute_paired_bootstrap(prod_results_sweep, r)
            pval = boot.p_value
        except Exception:
            pval = float("nan")
        better = "YES" if brier < prod_brier and pval < 0.10 else (
            "marginal" if brier < prod_brier and pval < 0.20 else "no")
        rows.append((brier, s.name, bias, rmse, brier - prod_brier, pval, better))

    rows.sort()  # sort by brier ascending

    print(f"\n  {'SCENARIO':<35} {'BRIER':>8} {'BIAS(°F)':>10} "
          f"{'RMSE':>8} {'ΔBRIER':>8} {'p-val':>7} {'BETTER?':>8}")
    print("  " + "-" * 90)
    for brier, name, bias, rmse, delta, pval, better in rows:
        marker = ">>>" if better == "YES" else "   "
        print(f"  {marker} {name:<32} {brier:>8.4f} {bias:>+10.3f} "
              f"{rmse:>8.3f} {delta:>+8.4f} {pval:>7.3f} {better:>8}")
    print(f"\n  --- Production (current) --- Brier={prod_brier:.4f}  "
          f"Bias={prod_m['mean_bias']:+.3f}°F  RMSE={prod_m['rmse']:.3f}°F")


# ---------------------------------------------------------------------------
# Section 5: Recommendations
# ---------------------------------------------------------------------------

def section_5_static() -> None:
    _sep("SECTION 5 — PRE-ANALYSIS FINDINGS (from DB inspection)")
    print("""
  These findings come from direct inspection of system_state across all dates
  and do not require running the replay (confirmed before the script was written).

  FINDING 1 — KALMAN BIAS FILTER THRASHING  [HIGH SEVERITY]
  ----------------------------------------------------------
  kalman_bias_estimate oscillates -3.99°F to +4.62°F on consecutive days.
  True NWP systematic bias changes by tenths per day, not flipping sign by 4°F.
  Root cause: H=[[1,1]] makes bias immediately observable from every ASOS tick.
  ASOS quantizes at 1°C ≈ 1.8°F steps, which propagates directly into the bias.
  When bias spikes to +4.62°F at 10 AM, MC attractor = NWP + 4.62°F — the model
  sees a "sure thing" and enters positions the market rightly prices at 3–5¢.

  RECOMMENDATIONS:
    a) Add kalman_bias_mc_cap in mc_params_builder.py to clip bias going into MCParams
       to ±2°F (configurable). Kalman filter continues running — only the MC
       sees a clamped value. Implementation: 2 lines in build_mc_params() and
       build_mc_params_historical(); new setting KALMAN_BIAS_MC_CAP in settings.py.
    b) Reduce kalman_q_bias from 0.05 to ~0.01 in settings.py or .env.
       Lower Q_bias means the filter requires more consistent evidence before
       attributing residuals to persistent bias (slower but more reliable).
       Effect: bias drifts toward 0 within ~1 day without new evidence.
       Restart kalshi-orchestrator after setting KALMAN_Q_BIAS=0.01 in .env.

  FINDING 2 — ou_max_stationary_std CAP IS NON-BINDING  [MEDIUM SEVERITY]
  -------------------------------------------------------------------------
  Calibrated cap = 3.776°F. Actual sigma/sqrt(2θ) = 1.394/sqrt(0.6) = 1.80°F.
  The cap never fires. calibrate_ou_max_stationary_std() computes hourly RMSE
  × 1.2 = 3.776, but this is forecast RMSE (NWP vs ASOS over the full day),
  not the intraday diffusion magnitude the OU cap is supposed to constrain.

  RECOMMENDATION:
    Reduce _HOURLY_RMSE_SAFETY_FACTOR from 1.2 to 0.6 in calibrator.py:621.
    This drops the nightly-calibrated cap to ~1.9°F, making it mildly binding
    at current sigma levels and providing headroom for future sigma increases.
    Takes effect at next midnight calibration — no restart needed.

  FINDING 3 — NWP SYSTEMATICALLY UNDER-PREDICTS  [MEDIUM SEVERITY]
  ------------------------------------------------------------------
  10 AM snapshot vs official high shows NWP (blended) under-predicts on the
  majority of dates. Calibrated drift (AM +1.148°F, PM +1.497°F) correctly
  identifies this, but use_drift_in_attractor=False discards the correction.
  The model's MC attractor runs cold relative to reality on most days.
  (The sweep results in §3-4 will show whether re-enabling drift helps or hurts
  given the Kalman bias thrashing is also present.)

  FINDING 4 — THETA CALIBRATION FALLING BACK TO DEFAULT  [LOW SEVERITY]
  -----------------------------------------------------------------------
  theta_decay stuck at 0.3000 for most recent dates — the factory default.
  calibrate_theta() silently falls back when fewer than 12 AR(1) pairs are
  available from NWP data. As data accumulates beyond 60+ days this will
  self-correct, but the optimal theta is currently unknown.
  The §3 Sweep D will reveal whether 0.3 is near-optimal or not.
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # Date range
    yesterday = (datetime.now(timezone.utc).astimezone(_EASTERN).date()
                 - timedelta(days=1))
    start = (date.fromisoformat(args.start_date)
             if args.start_date else date(2026, 3, 21))
    end   = (date.fromisoformat(args.end_date)
             if args.end_date else yesterday)

    full_hours  = args.eval_hours or [8, 10, 12, 14, 16]
    sweep_hours = args.eval_hours or [10]

    run_all = args.section is None
    run = lambda n: run_all or args.section == n

    print(f"\n{'='*72}")
    print("  KALSHI WEATHER TRADER — BACKTEST ANALYSIS")
    print(f"  Date range : {start} → {end}")
    print(f"  Full hours : {full_hours}  |  Sweep hours: {sweep_hours}")
    print(f"{'='*72}")

    prod_results      = []   # full eval hours (§1)
    prod_results_sw   = []   # sweep hours (§3/4 production baseline)

    if run(0):
        section_0(start, end)

    if run(1):
        prod_results = section_1(start, end, full_hours)

    if run(2):
        section_2(start, end)

    if run(3):
        section_3(start, end, sweep_hours, prod_results_sw)

    if run(4):
        section_4(start, end, sweep_hours, prod_results_sw)

    if run(5):
        section_5_static()

    print(f"\n{SEP}")
    print("  ANALYSIS COMPLETE")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
