# Phase L1: Model Lab — Replay Engine + Basic UI

## Status: COMPLETE (2026-04-06)

## Key Deviations from Spec
- D1: Add `anchor_weight_multiplier` to MCParams (minimal change, backwards-compatible)
- D2: Cloud/ensemble per-scenario sigma factors deferred to L2 (presets use settings defaults)
- D3: New result named `ParameterizedReplayResult` to coexist with old `ReplayResult`
- D4: No `get_settled_dates()` DB function exists — iterate + check `cli_settlement_confirmed`
- D5: `compute_aggregate_metrics` takes `list[ParameterizedReplayResult]` (separate from old DataFrame-based function)
- D6: Custom `__hash__` on Scenario to handle dict/list fields
- D7: `market_probs` populated as `{str(strike): prob}` from MC result

## Implementation Steps

- [x] Step 1: `quant/monte_carlo.py` — Add `anchor_weight_multiplier` to MCParams + apply in `run_simulation`
- [x] Step 2: `backtesting/scenarios.py` (NEW) — Scenario dataclass + ReplayDataCache + all presets
- [x] Step 3: `backtesting/replay_engine.py` — Add ParameterizedReplayResult + ParameterizedReplayEngine
- [x] Step 4: `backtesting/metrics.py` — Add `compute_aggregate_metrics(list[ParameterizedReplayResult])`
- [x] Step 5: `ui/model_lab.py` (NEW) — Tab 6, Replay mode only
- [x] Step 6: `ui/app.py` — Add Tab 6 import + render call
- [x] Step 7: `tests/test_model_lab.py` (NEW) — All Phase L1 tests
- [x] Step 8: Verify — run new tests + regression tests

## Definition of Done
- [x] All Phase L1 tests pass
- [x] Old `test_backtesting.py` still passes (no regression)
- [x] UI import works cleanly
- [x] `ParameterizedReplayEngine` runs end-to-end with synthetic data
