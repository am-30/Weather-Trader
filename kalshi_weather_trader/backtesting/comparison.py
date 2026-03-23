"""
Model variant comparison framework using paired bootstrap tests.

Compares two replay DataFrames (baseline vs proposed variant) to determine
whether the variant's improvement in Brier score is statistically significant.

The bootstrap unit is the DATE, not individual (date, hour, strike) triples —
predictions within a single day are correlated (same weather), so resampling
individual predictions inflates significance and leads to false conclusions.

Usage::

    from kalshi_weather_trader.backtesting.comparison import compare_variants

    summary = compare_variants(
        baseline_results=baseline_df,
        variant_results=variant_df,
        label_baseline="current_model",
        label_variant="time_varying_sigma",
    )
    # Returns dict with mean_diff, ci_lower, ci_upper, p_value
    # Also prints a human-readable summary
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

from kalshi_weather_trader.backtesting.metrics import _extract_strike_rows

logger = structlog.get_logger(__name__)

_N_BOOTSTRAP = 1000


def compare_variants(
    baseline_results: pd.DataFrame,
    variant_results: pd.DataFrame,
    label_baseline: str = "current",
    label_variant: str = "proposed",
    n_bootstrap: int = _N_BOOTSTRAP,
) -> dict:
    """Paired bootstrap test comparing Brier scores of two model variants.

    Both DataFrames must cover the same set of (date, eval_hour) pairs so that
    differences are paired. Dates present in one but not the other are excluded.

    The bootstrap procedure resamples DATES (not individual predictions) with
    replacement, computes the per-resample difference in mean Brier score, and
    derives a 95% CI and p-value from the resulting distribution.

    Args:
        baseline_results: DataFrame from ``ReplayEngine.replay_all()`` for the
                          current model.
        variant_results:  DataFrame from ``ReplayEngine.replay_all()`` for the
                          proposed model variant.
        label_baseline:   Human-readable name for the baseline.
        label_variant:    Human-readable name for the variant.
        n_bootstrap:      Bootstrap iterations. Default 1000.

    Returns:
        Dict with keys:
          - ``brier_baseline``: float — baseline mean Brier score
          - ``brier_variant``:  float — variant mean Brier score
          - ``mean_diff``:      float — brier_variant - brier_baseline
                                (positive = variant is worse)
          - ``ci_lower``:       float — 2.5th percentile of bootstrap differences
          - ``ci_upper``:       float — 97.5th percentile of bootstrap differences
          - ``p_value``:        float — fraction of bootstrap samples where
                                variant is worse than or equal to baseline
          - ``significant``:    bool — True if 95% CI excludes 0 and p_value < 0.05
          - ``n_dates``:        int — number of paired dates used

    Raises:
        Nothing — errors are logged and an empty dict returned.
    """
    if baseline_results.empty or variant_results.empty:
        logger.warning("compare_variants.empty_input")
        return {}

    try:
        return _run_comparison(
            baseline_results,
            variant_results,
            label_baseline,
            label_variant,
            n_bootstrap,
        )
    except Exception as exc:
        logger.error("compare_variants.failed", error=str(exc))
        return {}


def _run_comparison(
    baseline_results: pd.DataFrame,
    variant_results: pd.DataFrame,
    label_baseline: str,
    label_variant: str,
    n_bootstrap: int,
) -> dict:
    """Internal implementation of compare_variants.

    Args:
        baseline_results: Baseline replay DataFrame.
        variant_results:  Variant replay DataFrame.
        label_baseline:   Baseline label.
        label_variant:    Variant label.
        n_bootstrap:      Bootstrap iterations.

    Returns:
        Comparison result dict.

    Raises:
        ValueError: If no common dates exist.
    """
    # Expand both DataFrames to long form (date, hour, strike, prob, outcome, brier)
    base_long = _extract_strike_rows(baseline_results)
    var_long = _extract_strike_rows(variant_results)

    if base_long.empty or var_long.empty:
        logger.warning("compare_variants.no_strike_rows")
        return {}

    # Find common dates (intersection)
    base_dates = set(base_long["target_date"].unique())
    var_dates = set(var_long["target_date"].unique())
    common_dates = sorted(base_dates & var_dates)

    if not common_dates:
        raise ValueError(
            "No common dates between baseline and variant results. "
            "Both must cover the same trading days."
        )

    base_filtered = base_long[base_long["target_date"].isin(common_dates)]
    var_filtered = var_long[var_long["target_date"].isin(common_dates)]

    # Compute per-date mean Brier scores for each variant
    def _per_date_brier(long_df: pd.DataFrame) -> dict:
        return {
            d: float(group["brier"].mean())
            for d, group in long_df.groupby("target_date")
        }

    base_by_date = _per_date_brier(base_filtered)
    var_by_date = _per_date_brier(var_filtered)

    dates_arr = np.array(common_dates)
    base_arr = np.array([base_by_date[d] for d in dates_arr])
    var_arr = np.array([var_by_date[d] for d in dates_arr])

    # Point estimates
    brier_baseline = float(np.mean(base_arr))
    brier_variant = float(np.mean(var_arr))
    mean_diff = brier_variant - brier_baseline  # positive = variant worse

    # Paired bootstrap: resample dates with replacement
    rng = np.random.default_rng(0)
    boot_diffs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(dates_arr), size=len(dates_arr))
        diff = float(np.mean(var_arr[idx]) - np.mean(base_arr[idx]))
        boot_diffs.append(diff)

    boot_diffs_arr = np.array(boot_diffs)
    ci_lower = float(np.percentile(boot_diffs_arr, 2.5))
    ci_upper = float(np.percentile(boot_diffs_arr, 97.5))
    # p_value: fraction of bootstrap samples where variant is worse (diff >= 0)
    p_value = float(np.mean(boot_diffs_arr >= 0))
    significant = (ci_lower > 0 or ci_upper < 0) and p_value < 0.05

    summary = {
        "brier_baseline": brier_baseline,
        "brier_variant": brier_variant,
        "mean_diff": mean_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "significant": significant,
        "n_dates": len(common_dates),
    }

    _print_summary(summary, label_baseline, label_variant)
    return summary


def _print_summary(summary: dict, label_baseline: str, label_variant: str) -> None:
    """Print a human-readable comparison summary.

    Args:
        summary:        Result dict from _run_comparison.
        label_baseline: Baseline label.
        label_variant:  Variant label.
    """
    diff = summary["mean_diff"]
    direction = "WORSE" if diff > 0 else "BETTER"
    sig_str = "SIGNIFICANT" if summary["significant"] else "NOT significant"

    print(
        f"\nVariant '{label_variant}' Brier: {summary['brier_variant']:.4f} "
        f"vs baseline '{label_baseline}': {summary['brier_baseline']:.4f}.\n"
        f"  Difference: {diff:+.4f} ({direction}) "
        f"(95% CI: [{summary['ci_lower']:.4f}, {summary['ci_upper']:.4f}]).\n"
        f"  p-value: {summary['p_value']:.3f} — {sig_str} at p=0.05.\n"
        f"  Dates used: {summary['n_dates']}.\n"
    )
