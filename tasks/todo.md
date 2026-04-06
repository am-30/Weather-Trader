# Phase L2: Compare Mode + Custom Sliders

## Status: COMPLETE (2026-04-06)

## Key Deviations / Improvements over Spec
- D1: `BootstrapResult` is a typed dataclass (spec implied dict return) — cleaner for tests
- D2: `compare_scenarios()` is a method on `ParameterizedReplayEngine` (spec said standalone) — consistent API
- D3: A/B replays cached independently via `_run_replay_cached` — changing A doesn't bust B's cache
- D4: Custom slider panel exposes cloud/ensemble parameters conditionally (resolves L1 Deviation D2)
- D5: `test_comparison_resamples_by_date` checks `n_shared_dates == 5` instead of bootstrap internals
- D6: `BootstrapResult` uses `mean_diff`, `ci_low`, `ci_high`, `n_shared_dates` naming (from cuddly-imagining-hippo.md — clearer)
- D7: p_value uses shifted-bootstrap formula (handles identical-scenario edge case correctly)

## Files Modified
- `backtesting/metrics.py` — BootstrapResult dataclass + compute_paired_bootstrap()
- `backtesting/replay_engine.py` — ComparisonResult dataclass + compare_scenarios() method
- `ui/model_lab.py` — Compare mode UI + Custom slider panel (full rewrite)
- `tests/test_model_lab.py` — 4 Phase L2 tests added

## Implementation Steps

- [x] Step 1: `backtesting/metrics.py` — BootstrapResult + compute_paired_bootstrap()
- [x] Step 2: `backtesting/replay_engine.py` — ComparisonResult + compare_scenarios()
- [x] Step 3: `ui/model_lab.py` — Compare mode UI + Custom slider panel
- [x] Step 4: `tests/test_model_lab.py` — L2 tests
- [x] Step 5: Verify — 27/27 L2+L1 tests pass, 7/7 regression tests pass

## Definition of Done
- [x] All Phase L2 tests pass (4 new tests)
- [x] All Phase L1 tests still pass (23 existing tests — no regression)
- [x] test_backtesting.py passes (7 tests — no regression)
- [x] Custom slider panel exposes all Scenario fields including cloud/ensemble (D2 from L1 resolved)
