"""
Backtesting accuracy metrics for the Kalshi weather MC model.

Computes aggregate and per-hour statistics from a replay DataFrame produced
by ``ReplayEngine.replay_all()``. All metrics include 90% bootstrap confidence
intervals based on 1000 iterations with date-level resampling (same-day
predictions are correlated and must not be resampled independently).

Functions
---------
compute_backtest_metrics(results)
    Brier score, calibration curve, mean bias, RMSE, log-loss, sharpness,
    per-hour breakdown — all with bootstrap CIs.

compute_climatological_baseline(results, historical_highs)
    No-model baseline: predict historical exceedance frequency per strike.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

_N_BOOTSTRAP = 1000
_CI_LOWER = 5.0   # 90% CI
_CI_UPPER = 95.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_strike_rows(results: pd.DataFrame) -> pd.DataFrame:
    """Melt probability and outcome columns into (date, hour, strike, prob, outcome).

    Args:
        results: DataFrame from ReplayEngine.replay_all().

    Returns:
        Long-form DataFrame with columns:
        target_date, eval_hour, strike, prob, outcome, brier.
    """
    prob_cols = [c for c in results.columns if c.startswith("prob_")]
    if not prob_cols:
        return pd.DataFrame(columns=["target_date", "eval_hour", "strike", "prob", "outcome", "brier"])

    records = []
    for _, row in results.iterrows():
        actual_high = row["actual_high"]
        for col in prob_cols:
            strike = float(col[5:])  # "prob_40.0" -> 40.0
            prob = row[col]
            if pd.isna(prob):
                continue
            outcome = 1.0 if actual_high >= strike else 0.0
            brier_col = f"brier_{strike:.1f}"
            brier = row.get(brier_col, (prob - outcome) ** 2)
            records.append({
                "target_date": row["target_date"],
                "eval_hour": row["eval_hour"],
                "strike": strike,
                "prob": float(prob),
                "outcome": outcome,
                "brier": float(brier),
            })

    return pd.DataFrame(records)


def _bootstrap_ci(
    dates: np.ndarray,
    values_by_date: dict,
    agg_fn,
    n_bootstrap: int = _N_BOOTSTRAP,
) -> tuple[float, float]:
    """Compute bootstrap CI by resampling at the date level.

    Args:
        dates:          Array of unique date values.
        values_by_date: Dict mapping date → array of values for that date.
        agg_fn:         Function (array) → scalar; applied to pooled values
                        from each bootstrap resample.
        n_bootstrap:    Number of bootstrap iterations.

    Returns:
        Tuple (ci_lower, ci_upper) at the 5th/95th percentile.
    """
    boot_stats = []
    rng = np.random.default_rng(0)
    for _ in range(n_bootstrap):
        sampled_dates = rng.choice(dates, size=len(dates), replace=True)
        pooled = np.concatenate([values_by_date[d] for d in sampled_dates])
        boot_stats.append(agg_fn(pooled))
    return (
        float(np.percentile(boot_stats, _CI_LOWER)),
        float(np.percentile(boot_stats, _CI_UPPER)),
    )


def _point_and_ci(
    long_df: pd.DataFrame,
    value_col: str,
    agg_fn,
    n_bootstrap: int = _N_BOOTSTRAP,
) -> dict:
    """Compute a scalar metric with bootstrap CI.

    Args:
        long_df:    Long-form DataFrame with target_date column and value_col.
        value_col:  Column to aggregate.
        agg_fn:     Aggregation function (array → scalar).
        n_bootstrap: Bootstrap iterations.

    Returns:
        Dict with keys: value, ci_lower, ci_upper.
    """
    if long_df.empty:
        return {"value": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan")}

    point = agg_fn(long_df[value_col].values)
    dates = long_df["target_date"].unique()
    values_by_date = {
        d: long_df.loc[long_df["target_date"] == d, value_col].values
        for d in dates
    }
    ci_lower, ci_upper = _bootstrap_ci(dates, values_by_date, agg_fn, n_bootstrap)
    return {"value": float(point), "ci_lower": ci_lower, "ci_upper": ci_upper}


def _log_loss_fn(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Compute mean binary cross-entropy (log-loss).

    Args:
        probs:    Predicted probabilities, clipped to [1e-7, 1-1e-7].
        outcomes: Binary outcomes (0 or 1).

    Returns:
        Mean log-loss (lower is better).
    """
    p = np.clip(probs, 1e-7, 1 - 1e-7)
    return float(-np.mean(outcomes * np.log(p) + (1 - outcomes) * np.log(1 - p)))


def _calibration_curve(long_df: pd.DataFrame, n_bins: int = 10) -> list[dict]:
    """Compute calibration curve by binning predicted probabilities.

    Args:
        long_df: Long-form DataFrame with prob and outcome columns.
        n_bins:  Number of equal-width bins (default 10 → deciles).

    Returns:
        List of dicts with keys: bin_lower, bin_upper, mean_pred, observed_freq, count.
        Bins with no predictions are omitted.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    curve = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (long_df["prob"] >= lo) & (long_df["prob"] < hi)
        if i == n_bins - 1:
            mask = (long_df["prob"] >= lo) & (long_df["prob"] <= hi)
        subset = long_df[mask]
        if subset.empty:
            continue
        curve.append({
            "bin_lower": float(lo),
            "bin_upper": float(hi),
            "mean_pred": float(subset["prob"].mean()),
            "observed_freq": float(subset["outcome"].mean()),
            "count": int(len(subset)),
        })
    return curve


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_backtest_metrics(
    results: pd.DataFrame,
    n_bootstrap: int = _N_BOOTSTRAP,
) -> dict:
    """Compute accuracy metrics from replay_all() output.

    Metrics include Brier score, calibration curve, mean bias, RMSE, log-loss,
    sharpness, and per-hour breakdowns. All scalar metrics include 90% bootstrap
    confidence intervals (date-level resampling, n_bootstrap iterations).

    Args:
        results:     DataFrame from ``ReplayEngine.replay_all()``.
        n_bootstrap: Bootstrap iterations for CI computation. Default 1000.

    Returns:
        Dict with keys:
          - ``brier_score``: dict(value, ci_lower, ci_upper) — aggregate Brier score
          - ``brier_by_strike``: dict of strike → dict(value, ci_lower, ci_upper)
          - ``calibration_curve``: list of bin dicts (mean_pred, observed_freq, count)
          - ``mean_bias``: dict(value, ci_lower, ci_upper) — mean(mean_max - actual_high)
          - ``rmse``: dict(value, ci_lower, ci_upper) — RMSE of mean_max vs actual_high
          - ``log_loss``: float — mean binary cross-entropy across all strikes
          - ``sharpness``: dict(value, ci_lower, ci_upper) — mean |p - 0.5|
          - ``n_dates``: int — number of settled dates in the dataset
          - ``n_rows``: int — total (date, hour) rows
          - ``per_hour``: dict of eval_hour → sub-dict with brier, mean_bias, rmse, sharpness

    Raises:
        Nothing — errors are logged; affected metrics return NaN.
    """
    if results.empty:
        logger.warning("metrics.empty_results")
        return {}

    long = _extract_strike_rows(results)
    n_dates = int(results["target_date"].nunique())

    # ----------------------------------------------------------------
    # Aggregate Brier score
    # ----------------------------------------------------------------
    brier_score = _point_and_ci(long, "brier", np.mean, n_bootstrap)

    # ----------------------------------------------------------------
    # Per-strike Brier scores
    # ----------------------------------------------------------------
    brier_by_strike: dict[float, dict] = {}
    for strike, group in long.groupby("strike"):
        brier_by_strike[float(strike)] = _point_and_ci(
            group.reset_index(drop=True), "brier", np.mean, n_bootstrap
        )

    # ----------------------------------------------------------------
    # Calibration curve
    # ----------------------------------------------------------------
    calibration_curve = _calibration_curve(long)

    # ----------------------------------------------------------------
    # Mean bias and RMSE (on mean_max vs actual_high; per date+hour row)
    # ----------------------------------------------------------------
    point_df = results[["target_date", "eval_hour", "mean_max", "actual_high"]].copy()
    point_df = point_df.dropna(subset=["mean_max", "actual_high"])
    point_df["bias_val"] = point_df["mean_max"] - point_df["actual_high"]
    point_df["sq_err"] = point_df["bias_val"] ** 2

    mean_bias = _point_and_ci(point_df, "bias_val", np.mean, n_bootstrap)
    rmse_val = _point_and_ci(
        point_df, "sq_err", lambda x: float(np.sqrt(np.mean(x))), n_bootstrap
    )

    # ----------------------------------------------------------------
    # Log-loss (scalar, no CI — too noisy with few dates)
    # ----------------------------------------------------------------
    if not long.empty:
        log_loss = _log_loss_fn(long["prob"].values, long["outcome"].values)
    else:
        log_loss = float("nan")

    # ----------------------------------------------------------------
    # Sharpness: mean |p - 0.5|
    # ----------------------------------------------------------------
    long["sharpness_val"] = (long["prob"] - 0.5).abs()
    sharpness = _point_and_ci(long, "sharpness_val", np.mean, n_bootstrap)

    # ----------------------------------------------------------------
    # Per-hour breakdown
    # ----------------------------------------------------------------
    per_hour: dict[int, dict] = {}
    for hour, group in long.groupby("eval_hour"):
        h = int(hour)
        g = group.reset_index(drop=True)
        h_point_df = results[results["eval_hour"] == hour][
            ["target_date", "mean_max", "actual_high"]
        ].copy().dropna()
        h_point_df["bias_val"] = h_point_df["mean_max"] - h_point_df["actual_high"]
        h_point_df["sq_err"] = h_point_df["bias_val"] ** 2
        g["sharpness_val"] = (g["prob"] - 0.5).abs()

        per_hour[h] = {
            "brier_score": _point_and_ci(g, "brier", np.mean, n_bootstrap),
            "mean_bias": _point_and_ci(h_point_df, "bias_val", np.mean, n_bootstrap),
            "rmse": _point_and_ci(
                h_point_df, "sq_err", lambda x: float(np.sqrt(np.mean(x))), n_bootstrap
            ),
            "sharpness": _point_and_ci(g, "sharpness_val", np.mean, n_bootstrap),
            "n_rows": int(len(h_point_df)),
        }

    return {
        "brier_score": brier_score,
        "brier_by_strike": brier_by_strike,
        "calibration_curve": calibration_curve,
        "mean_bias": mean_bias,
        "rmse": rmse_val,
        "log_loss": log_loss,
        "sharpness": sharpness,
        "n_dates": n_dates,
        "n_rows": len(results),
        "per_hour": per_hour,
    }


def compute_climatological_baseline(
    results: pd.DataFrame,
    historical_highs: list[float],
    n_bootstrap: int = _N_BOOTSTRAP,
) -> dict:
    """Compute the no-model climatological baseline metrics.

    For each strike, predicts the historical exceedance frequency:
    P(daily_max >= strike) = fraction of historical_highs >= strike.

    This is the bar the MC model must clear to have edge. A model with Brier
    score equal to or worse than this baseline has no skill.

    Args:
        results:          DataFrame from ``ReplayEngine.replay_all()`` (for
                          extracting strike list and actual outcomes).
        historical_highs: List of NWS daily maximum temperatures (°F) from
                          historical records for the same calendar month(s).
                          Source: IEM CLImate data or NOAA GHCN-Daily.
        n_bootstrap:      Bootstrap iterations for CI computation.

    Returns:
        Same structure as ``compute_backtest_metrics()`` but computed using
        historical-frequency predictions instead of MC model predictions.

    Raises:
        Nothing — errors logged and empty dict returned on failure.
    """
    if results.empty or not historical_highs:
        return {}

    highs = np.array(historical_highs, dtype=float)
    long = _extract_strike_rows(results)
    if long.empty:
        return {}

    # Replace MC probs with climatological frequencies
    clim_probs: dict[float, float] = {}
    for strike in long["strike"].unique():
        clim_probs[float(strike)] = float(np.mean(highs >= strike))

    long["prob"] = long["strike"].map(clim_probs)
    long["brier"] = (long["prob"] - long["outcome"]) ** 2
    long["sharpness_val"] = (long["prob"] - 0.5).abs()

    brier_score = _point_and_ci(long, "brier", np.mean, n_bootstrap)
    log_loss = _log_loss_fn(long["prob"].values, long["outcome"].values)
    sharpness = _point_and_ci(long, "sharpness_val", np.mean, n_bootstrap)

    return {
        "brier_score": brier_score,
        "log_loss": log_loss,
        "sharpness": sharpness,
        "climatological_probs": clim_probs,
        "n_dates": int(results["target_date"].nunique()),
        "n_rows": len(results),
    }
