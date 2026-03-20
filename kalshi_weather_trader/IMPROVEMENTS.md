# IMPROVEMENTS.md — Kalshi Weather Trading System
# Created: March 19, 2026
# Source: Full system code review across all major components

This document is the living backlog for incremental improvements.
Items are organized by tier (model correctness → data integrity →
execution → UI), then by implementation priority within each tier.
New findings from deep sub-agent reviews are integrated below the
original roadmap in each tier.

---

## Tier 1 — Model Correctness (Systematic Bias Sources)

### 1. Theta calibration fits AR(1) on raw temperatures, not NWP departures
**File:** `calibration/calibrator.py` — `calibrate_theta()` lines 370–414

**The bug:** `hourly_temps = temps[::12]` then AR(1) on `hourly_temps` directly.
Raw hourly KBOS temperatures have very high autocorrelation (φ ≈ 0.97–0.99) due to
the strong diurnal cycle, not mean-reversion. This makes `theta = -ln(φ)/dt` near
zero, which means the OU paths in the simulation barely revert toward the NWP
attractor — they drift almost as random walks. The OU process in monte_carlo.py
models *departures from the NWP curve*, so theta should be calibrated on
`T_t - nwp_curve[hour_idx]` (residuals), not on `T_t` itself.

**Fix:** For each hourly reading, look up the corresponding NWP blended forecast
value and compute the departure. Fit AR(1) on departures. This will typically yield
φ ≈ 0.7–0.85 → theta ≈ 0.16–0.36/hr, which is physically meaningful mean-reversion
speed for intraday temperature anomalies.

**Impact:** High. Underestimated theta → simulation paths too wide → probabilities
near 50% → all edges below threshold → system never trades OR overconfident edges
if sigma is also inflated.

---

### 2. Sigma calibration includes the diurnal trend signal
**File:** `quant/monte_carlo.py` — `estimate_sigma_from_historical()` lines 534–598

**The bug:** Uses `(dT)^2 / dt` for each 5-min ASOS diff, where dT includes both
the predictable NWP-forecast warming trend and true diffusion noise. During morning
warm-up (7 AM–2 PM), each 5-min interval has a systematic positive dT of
~0.3–0.6°F/hr, inflating sigma by conflating the trend with volatility.

The OU process models `dT = theta*(mu_t - T_t)*dt + sigma*sqrt(dt)*Z`. The sigma
in this equation is the *noise amplitude of the residual*, not the total temperature
change amplitude. Estimating sigma from raw diffs over-counts the trend.

**Fix:** For each 5-min interval, subtract the NWP-predicted change over that
interval before computing contributions. `dT_residual = dT_observed - (nwp[t+dt] -
nwp[t])`. Then `sigma = sqrt(mean(dT_residual^2 / dt))`. Requires passing a blended
NWP curve to `estimate_sigma_from_historical()` — currently it only takes readings.

**Related:** Sigma is also pooled across the entire diurnal cycle. Boston morning
(pre-dawn) has volatility ~0.5°F/hr; afternoon peak is ~2°F/hr. Pooling yields a
blended ~1.5°F/sqrt-hr that underestimates afternoon risk and overestimates morning
risk. Stratifying sigma by hour-of-day is a further enhancement.

**Impact:** High. Inflated sigma → distribution too wide → model underconfident at
near-certain strikes → misses edges.

---

### 3. Kalman bias resets to 0.0 on every new trading day (no warm-start)
**File:** `quant/kalman_filter.py` — `load_or_initialize_filter()` lines 240–289

**The bug:** On each new `target_date`, if no system_state row exists yet,
`KalmanFilter(initial_temp=current_asos_temp, initial_bias=0.0)` is called. The
previous day's converged bias (which could be +2°F or −1.5°F for a persistent NWP
systematic error) is completely discarded. The bias must reconverge from scratch each
morning, which takes ~2–4 hours of ASOS observations at R=0.6. During that
reconvergence window, `nwp_anchor_offset` is larger than it should be (because bias
hasn't yet absorbed the systematic error) and the simulation is off-target.

**Fix:** After settlement/rollover, carry forward yesterday's `kalman_bias_estimate`
as the initial bias for the new day. Temperature should still reset to current ASOS
(the diurnal cycle makes yesterday's Kalman temp useless for today), but bias is
slow-moving and should persist. The covariance can also be partially preserved (e.g.,
use yesterday's final P with a slight inflation: `P_new = P_yesterday * 1.2` instead
of identity, to reflect 12+ hours of unobserved time).

**Impact:** High. Each trading morning the model runs with stale bias, inflating NWP
anchor offset, until ~10 AM. These are often the highest-volume trading hours.

---

### 4. Brier scoring uses fixed synthetic Gaussian, not model probability distributions
**File:** `calibration/calibrator.py` — `_brier_score_for_model()` lines 39–103

**The bug:** For each past day, uses `1 - norm.cdf(official_high, loc=predicted_high,
scale=2.0)` as the forecast probability, with a *fixed scale of 2.0°F for all
models*. This means model weights are determined entirely by which model's *point
forecast* was closer to the actual high — the fixed 2.0°F width affects all models
identically. The "outcome" is `1.0 if official_high >= predicted_high else 0.0`,
which scores models on whether they over- or under-forecast, not on how well
calibrated their probability distributions are.

**Fix:** Retrieve yesterday's MC result from `intraday_snapshots` and use the actual
stored CDF to compute `P(max >= official_high | model)`. Score = `(p_yes - 1)^2`
where outcome = 1 (the official high was observed). This uses the full distribution
and properly rewards both accuracy and calibration.

**Impact:** Medium-High. Current scoring still rank-orders models by point error,
which is correlated with distribution quality. But it can't distinguish a
systematically biased model vs one with higher noise, and ignores distributional info.

---

### 5. Kalman observation noise R=0.6 may be too high
**File:** `config/settings.py` — `kalman_r_obs`

**The issue:** ASOS sensors have accuracy ±0.5°F, suggesting R should be closer to
0.25°F². R=0.6 gives observations low weight in updates (smaller Kalman gains),
causing the filter to trust its own state evolution over fresh measurements. Bias
estimation becomes sluggish — observations that contradict NWP don't immediately pull
the bias estimate. This was increased recently (commit f661ca9) to dampen noise, but
may be overcorrecting.

**Note:** R was deliberately increased to fix a prior bug. Any reduction should be
tested carefully against historical data first.

---

### 6. Kalman state transition uses identity matrix — no diurnal physics
**File:** `quant/kalman_filter.py` — `predict()` method

**The issue:** The state transition F is the 2×2 identity matrix. The Kalman filter
predicts "next temperature = current temperature." All temperature dynamics come from
NWP deltas added as a control input. If NWP is wrong about the warming rate, the
filter can't self-correct the rate of change independently. The predict step accepts
NWP warming/cooling delta immediately with no rate-limiting or physical constraints.
A large NWP spike (e.g., "temperature rises 10°F next hour") is accepted without
dampening.

---

### 7. OU default parameters may be too conservative for Boston
**File:** `config/settings.py` — `ou_theta`, `ou_sigma`

**The issue:** `theta=0.1` means half-life of ln(2)/0.1 ≈ 7 hours — slow
mean-reversion. Boston max temperature typically peaks by 3 PM; below that, reversion
is rapid. A better default would be theta=0.3–0.5/hr (half-life 1.4–2.3 hours). These
are overridden by calibration, but calibration may not run on first startup, leaving
the system with physically implausible defaults.

---

## Tier 2 — Data Integrity & Robustness

### 8. Hard floor doesn't read the ASOS 6-hour maximum METAR field
**File:** `ingestion/asos_fetcher.py` — temperature parsing section

**The problem (documented in CLAUDE.md):** The ASOS sensor uses a 0.5°C persistence
filter. A true temperature peak of 40.1°F may not be reflected in the tabular reading
if the sensor hasn't crossed its threshold since the last 0.5°C step. The METAR
6-hour maximum temperature field (`maxT6` in NWS GeoJSON) captures intraday peaks
including sub-threshold spikes, and is the more reliable source for the hard floor.

The current hard floor is updated from the tabular `temperature` field only, meaning
the hard floor may be 0.2–0.4°F below the true intraday maximum near a bucket
boundary. At a boundary (e.g., 39.5°F), this difference could affect the YES/NO
probability by 2–5%.

**Fix:** In the ASOS parser, also extract `maxT6` (or `maxT24`) from the GeoJSON
properties. On each fetch, call `update_hard_floor()` with
`max(tabular_temp, max6h_temp)` after converting both to °F.

---

### 9. `_normalize_market()` has no input guard on yes_bid_dollars / yes_ask_dollars
**File:** `ingestion/kalshi_fetcher.py` — `_normalize_market()`

**The problem (CLAUDE.md blocking issue #3):** If Kalshi changes their API response
to return integers in [0,100] instead of floats in [0,1], all downstream edge
calculations inflate 100x silently.

**Fix:**
```python
if yes_bid_dollars > 1.0 or yes_ask_dollars > 1.0:
    # Assume integers — divide by 100
    yes_bid_dollars /= 100.0
    yes_ask_dollars /= 100.0
```
Or raise with a clear error. Either way, this should be caught before reaching edge
computation.

---

### 10. Partition sum tolerance of 10% is too loose relative to 5% edge threshold
**File:** `quant/monte_carlo.py` — `compute_normalized_market_probs()` line 512

**The problem:** The function normalizes probabilities only if `|sum - 1.0| <= 0.10`.
A 10% partition error means the normalized probabilities can each shift by up to ~5%.
Since the edge threshold is 5%, a trade just above the threshold could flip below (or
vice versa) from normalization noise alone. A well-formed Kalshi market should
partition exactly, so a >3% deviation indicates a structural problem (gap in market
listings) that should halt trading, not be silently corrected.

**Fix:** Tighten to `<= 0.05` for normalization and log at ERROR level. Above 5%
deviation, log the gap details and optionally skip the trade for that evaluation cycle.

---

### 11. NWP blended curve silently truncates when models have different horizons
**File:** `ingestion/nwp_fetcher.py` — lines 328–391

**The issue:** If HRRR has 18 hours but GFS has 24 hours, the per-hour loop breaks
when no models contribute data for that hour. Hours 19–24 are silently dropped from
the blend. If the afternoon peak falls in that window (e.g., HRRR only covers through
noon), the MC simulation's attractor curve is cut short and the OU paths drift freely
past HRRR's horizon.

**Fix:** For each hour, blend using only models that have data at that hour (already
done per-hour), but continue through the longest available model's horizon. This
requires detecting the fallback and logging it clearly.

---

### 12. NWS day bounds use fixed UTC-5 (EST), not DST-aware Eastern Time
**File:** `ingestion/asos_fetcher.py` — `get_nws_day_bounds()` (referenced throughout)

**The issue:** The US observes EDT (UTC-4) from mid-March to early November. If
`get_nws_day_bounds()` uses UTC-5 fixed offset, all ASOS readings are attributed to
the wrong hour during EDT months, causing hard-floor update failures and potential
cross-day contamination. This is a systemic issue affecting ~8 months of trading.

**Fix:** Use `pytz.timezone("America/New_York")` for all ET-based day boundary
calculations. Cross-check against the hour_offset DST logic already in MCParams.

---

### 13. Hard floor corrupted during post-6PM rollover gap
**Files:** `ingestion/asos_fetcher.py`, `scheduler/orchestrator.py`

**The issue:** After 6 PM ET rollover, `target_date` is tomorrow, but ASOS reads
continue every 5 minutes with today's sensor data. The ASOS fetch job calls
`update_hard_floor()` with today's afternoon peak temperature applied to *tomorrow's*
market row. Tomorrow starts its trading day with today's maximum as its hard floor,
causing systematic short-probability bias until tomorrow's real data arrives.

**Status:** Partially addressed in March 17 update (rolling floor reset). Verify the
fix covers the full 6 PM–midnight window including DST edge cases.

---

### 14. IEM fallback CSV timestamp parsing is brittle
**File:** `ingestion/asos_fetcher.py` lines 205–296

**The issue:** `_fetch_iem_current()` parses `valid_raw` as ISO format using
`valid_raw.replace(" ", "T") + "+00:00"`. This assumes a single space separator. If
IEM changes to tab-separated or multi-space, the ISO parse fails silently (ValueError
caught but function returns None). IEM fallback occasionally returns None even when
data is available, causing ASOS fetch to fail.

---

### 15. Hard floor `update_hard_floor()` has a race condition
**File:** `db/db_manager.py` lines 357–425

**The issue:** The function performs TWO separate database operations: a SELECT to
check if the row exists (line 375), then a separate UPDATE (lines 393–405). Between
these two statements, another concurrent process could insert or update the row,
violating atomicity.

**CLAUDE.md states** hard floor updates must use single-statement SQL:
```sql
UPDATE markets SET current_max_observed = GREATEST(current_max_observed, :val)
WHERE market_id = :id
```
Verify that no preceding SELECT exists. Atomicity depends on single statement with no
read-modify-write cycle.

---

### 16. No database indexes on critical query paths
**File:** `db/db_manager.py` — ORM definitions lines 80–211

**The issue:** Several high-frequency query paths lack indexes:
- `asos_readings`: no index on `observation_time_utc` alone for range queries
- `nwp_forecasts`: no index on `(target_date, fetched_at_utc)`
- `intraday_snapshots`: no index on `(target_date, snapshot_time_utc)`
- `trade_logs`: no index on `(target_date, executed_at_utc)`

After 2–4 weeks of operation (millions of readings), full-table scans will
materially slow the UI and calibrator. This should be addressed before extended
operation, not after.

---

### 17. Settlement confirmation job may calibrate against wrong date
**File:** `scheduler/orchestrator.py` line 512

**The issue:** `job_confirm_settlement()` runs at 10:05 AM ET. By that time,
`get_target_date()` has already rolled to today's date. If line 512 calls
`run_full_calibration(get_target_date())`, it calibrates today's state rather than
yesterday's. Yesterday's Brier score and drift adjustments computed from the
preliminary ASOS high (not the official NWS value) are never updated.

**Fix:** `job_confirm_settlement()` should explicitly pass `yesterday` (computed as
`calendar_date - timedelta(days=1)`) to `run_full_calibration()`.

---

### 18. NWS CLI regex patterns unverified against real pages (BLOCKING)
**File:** `ingestion/nws_cli_fetcher.py` lines 68–153

**The issue (CLAUDE.md blocking issue #2):** `fetch_official_daily_high()` patterns
("CLIMATE SUMMARY FOR", "MAXIMUM\s+") have never been run against a real NWS page.
If the page uses "MAXIMUM TEMPERATURE (°F)" instead of "(F)", or if NWS changes
format, the regex silently fails and returns None, propagating a stale or missing
settlement value without alert. The 10:05 AM settlement job depends entirely on this
working.

---

## Tier 3 — Execution & Audit Quality

### 19. Kelly contract sizing floors at 1 even when raw Kelly < 1
**File:** `execution/trader.py` — `compute_kelly_contracts()` line 76

**The problem:** `contracts = max(1, min(int(raw_contracts), max_contracts))`.
When raw Kelly says 0.3 contracts (e.g., edge just above threshold, small max_size),
this forces a trade of 1 contract. The floor at 1 means the system always trades at
least 1 contract whenever any edge exists above the threshold, regardless of how small
the Kelly-optimal position is. This is not catastrophic ($0.55 risk at 55¢ ask) but
it defeats fractional Kelly's risk management purpose.

**Fix:** Return `None` when `int(raw_contracts) < 1` instead of forcing to 1.
Kelly = 0.3 contracts genuinely means "this edge isn't large enough to justify even
the minimum position."

---

### 20. `_log_no_trade` always logs action="BUY_YES" and markets[0]
**File:** `execution/trader.py` — `_log_no_trade()` lines 438–491

**The problem:** When no trade occurs, the no-trade log records `action="BUY_YES"`
and uses `markets[0]` regardless of which market actually had the best edge. If the
near-miss was a BUY_NO at a different strike, the audit trail is wrong. Over many
trading days this corrupts the historical record used to evaluate model performance.

**Fix:** Pass `best_action`, `best_strike`, `best_edge`, and the best `market` dict
from `evaluate_and_trade()` to `_log_no_trade()`. If no market had positive edge at
all, fall back to current behavior with a note.

---

### 21. Trade log stores insufficient MC context for post-hoc analysis
**File:** `execution/trader.py` — trade log `notes` field, line 431

**The problem:** The notes field only stores `"MC n_paths=X, hard_floor=Y"`. To
understand why a trade was placed or avoided on a given day requires knowing T0,
sigma, theta, hour_offset, drift_adj, bias, and is_future_day at execution time.
Without these, reproducing the exact MC run that produced fair_value_prob is
impossible.

**Fix:** Expand the notes field (or add a `mc_params_json` column to trade_logs) to
capture the full MCParams state. JSON-encode T0, sigma, theta, hour_offset, drift_adj,
bias, n_paths, hard_floor, is_future_day at execution time.

---

### 22. Kelly formula crashes when ask_decimal = 1.0
**File:** `execution/trader.py` lines 64–87

**The issue:** Line 64 guards against `ask_decimal <= 0.0`, but there's no guard
against `ask_decimal = 1.0`, which makes `b = (1.0/1.0) - 1 = 0`. Line 69 then
divides by zero: `kelly = (p*0 - (1-p)) / 0`. This is caught by the outer
try/except and returns None, but the failure is silent. If a market bid/ask collapses
to 0/100, trade evaluation fails without explanation.

---

### 23. Position tracking is long-exposure only
**File:** `execution/trader.py` lines 199–209

**The issue (CLAUDE.md known issue #13):** `get_positions()` reduces Kelly by long
exposure but not short (NO) positions. A short position of −50 contracts becomes 50,
incorrectly reducing Kelly sizing and causing double-sizing on the next YES order.
This only matters if the system ever trades both YES and NO on the same strike, but
the code path doesn't guard against it.

---

### 24. MCParams constructed independently in 4 places (tech debt)
**Files:** `execution/trader.py`, `ui/app.py` (Tab 1), `ui/app.py` (Stage 3),
`calibration/calibrator.py`

**The issue (CLAUDE.md known issue #5):** If `hour_offset` or `drift_adj` logic
is updated in one place, it silently diverges in the others. The UI may display
different probability estimates than what the trader is actually computing.

**Fix:** Extract MCParams construction into a single shared function in a utility
module (e.g., `quant/mc_params_builder.py`) and import from all four locations.

---

### 25. Snapshot doesn't store NWP curve used in MC
**File:** `calibration/calibrator.py` — `record_snapshot()`

**The issue:** Snapshots record current state and MC results but not the NWP curve
used in MC. If NWP is updated between snapshots, you can't replay the exact MC from a
snapshot's fair-value estimate. Post-hoc debugging of trade decisions is significantly
harder.

**Fix:** Store `nwp_curve` and all MCParams fields in the snapshot record.

---

### 26. No explicit transaction boundaries for multi-step DB operations
**File:** `db/db_manager.py` — all CRUD functions

**The issue:** Operations like "upsert market + insert snapshot + log trade" are
three separate transactions. If the process crashes between them, the DB is left in
an inconsistent state (e.g., a trade is logged but the market state isn't updated).
Critical multi-step writes should be wrapped in explicit transactions.

---

## Tier 4 — UI Transparency

### 27. No data staleness indicators anywhere in the UI
**File:** `kalshi_weather_trader/ui/app.py` — all dashboard tabs

**The problem:** The dashboard shows current ASOS temperature, Kalman estimate, and
NWP forecasts with no indication of how old they are. A user watching the app at
3 PM can't tell if the ASOS was fetched 2 minutes ago or is 35 minutes stale (the
staleness threshold). During an NWS API outage, the system will keep displaying the
last-seen value indefinitely with no visual warning.

**Fix:** On Tab 1 (Trading Desk), display "Last ASOS: X min ago" and color it yellow
if > 15 min, red if > 30 min. Similarly for NWP ("Last NWP: X min ago") and Kalshi
prices. These timestamps are already stored in the DB
(`asos_readings.observation_time_utc`, `nwp_forecasts.fetched_at_utc`).

---

### 28. Stage 4 histogram uses synthetic normal distribution, not actual paths_max
**File:** `kalshi_weather_trader/ui/app.py` — Stage 4 histogram section, lines 2103–2105

**The problem (CLAUDE.md known issue #7):** The histogram is generated with
`np.random.normal(mean_max, std_max, 5000)` which produces a symmetric, unbounded
normal. The actual paths_max distribution is left-truncated at hard_floor (cannot go
below the observed maximum) and right-skewed. Near a boundary strike, the synthetic
distribution can show non-trivial probability mass below hard_floor (physically
impossible), misrepresenting model confidence.

The histogram title says "Simulated Daily Max" but it is a post-hoc Gaussian
approximation, not the real simulation output.

**Fix:** Pass actual `paths_max` percentile data from the snapshot through to the
histogram, or clearly label the chart as "fitted normal approximation." Better: run a
lightweight MC in the UI to generate fresh paths_max for visualization only.

---

### 29. Kill switch state can become out of sync if market row not found
**File:** `ui/app.py` lines 186–210

**The issue:** `auto_trade_enabled = market.auto_trade_enabled if market else True`.
If the market row doesn't exist, the UI shows "TRADING ENABLED" but trading may be
halted in the scheduler. The fallback `True` is the dangerous direction for a missing
row.

---

### 30. Kalman divergence warning uses hard-coded threshold (3.0°F)
**File:** `ui/app.py` line 1139

**The issue:** `abs(state.kalman_temp_estimate - asos_val) > 3.0` fires a warning
at a fixed 3°F regardless of season or time of day. On a 20°F winter day, 3°F is a
15% deviation; on an 80°F summer day, it's 3.75%. The threshold should scale with
Kalman uncertainty (P diagonal) or at minimum be configurable in settings.

---

### 31. Recent trades table has no legend for "Dry Run" column
**File:** `ui/app.py` lines 446–468

**The issue:** A checkmark in Dry Run is clear, but the absence is ambiguous — does
it mean the order was filled, rejected, or pending? No legend or tooltip explains the
column semantics.

---

### 32. Snapshot history table doesn't show whether each snapshot led to a trade
**File:** `ui/app.py` lines 914–936

**The issue:** Each snapshot captures a model decision, but there's no "Trade?" column
showing whether that decision resulted in an order. Users must cross-reference with
the Recent Trades tab manually.

---

## Summary Table

| # | Description | Tier | Files | Priority |
|---|---|---|---|---|
| 1 | Theta: AR(1) on raw temps vs NWP departures | Model | `calibrator.py` | Immediate |
| 2 | Sigma: includes diurnal trend + not stratified by hour | Model | `monte_carlo.py` | High |
| 3 | Kalman bias cold-starts at 0.0 daily | Model | `kalman_filter.py` | Immediate |
| 4 | Brier scoring uses fixed Gaussian, not MC distributions | Model | `calibrator.py` | High |
| 5 | Kalman R=0.6 may be too high for fast bias convergence | Model | `settings.py` | Medium |
| 6 | Kalman F=identity: no diurnal physics in state transition | Model | `kalman_filter.py` | Low |
| 7 | OU default theta=0.1 too conservative before calibration | Model | `settings.py` | Medium |
| 8 | Hard floor misses ASOS 6-hour max METAR field | Data | `asos_fetcher.py` | High |
| 9 | `_normalize_market()` no input guard (BLOCKING) | Data | `kalshi_fetcher.py` | Immediate |
| 10 | Partition sum tolerance 10% too loose vs 5% edge | Data | `monte_carlo.py` | Medium |
| 11 | NWP blend silently truncates shorter model horizons | Data | `nwp_fetcher.py` | High |
| 12 | NWS day bounds use EST fixed offset, not DST-aware | Data | `asos_fetcher.py` | High |
| 13 | Hard floor corrupted in post-6PM rollover gap | Data | `asos_fetcher.py`, `orchestrator.py` | High |
| 14 | IEM CSV timestamp parsing brittle | Data | `asos_fetcher.py` | Medium |
| 15 | `update_hard_floor()` race condition (BLOCKING verify) | Data | `db_manager.py` | Immediate |
| 16 | No DB indexes on critical query paths | Data | `db_manager.py` | High |
| 17 | Settlement calibration uses wrong date | Data | `orchestrator.py` | High |
| 18 | NWS CLI regex unverified on real pages (BLOCKING) | Data | `nws_cli_fetcher.py` | Immediate |
| 19 | Kelly floors at 1 contract when raw < 1 | Execution | `trader.py` | Medium |
| 20 | `_log_no_trade` logs wrong action/market | Execution | `trader.py` | Immediate |
| 21 | Trade log missing full MCParams context | Execution | `trader.py`, `db/schema` | Medium |
| 22 | Kelly formula crashes when ask_decimal = 1.0 | Execution | `trader.py` | Medium |
| 23 | Position tracking is long-exposure only | Execution | `trader.py` | Medium |
| 24 | MCParams constructed in 4 independent places | Execution | `trader.py`, `app.py` (x2), `calibrator.py` | High |
| 25 | Snapshot missing NWP curve used in MC | Execution | `calibrator.py` | Low |
| 26 | No transaction boundaries for multi-step DB writes | Execution | `db_manager.py` | Medium |
| 27 | No staleness indicators in UI | UI | `app.py` | High |
| 28 | Stage 4 histogram uses synthetic normal | UI | `app.py` | Medium |
| 29 | Kill switch defaults to True when market row missing | UI | `app.py` | Medium |
| 30 | Kalman divergence warning threshold hard-coded | UI | `app.py` | Low |
| 31 | Dry Run column has no legend | UI | `app.py` | Low |
| 32 | Snapshot table doesn't cross-reference trades | UI | `app.py` | Low |

---

## Recommended Implementation Order

### Immediate (before any live trades)
- **#9** (`_normalize_market` input guard) — 10-line fix, prevents silent 100× inflation
- **#3** (Kalman bias warm-start) — highest impact per line-of-code, fixes cold-start daily
- **#20** (`_log_no_trade` correctness) — cheap fix, corrupts audit trail every cycle
- **#15** (verify `update_hard_floor` atomicity) — confirm no SELECT precedes UPDATE
- **#18** (NWS CLI regex) — run against a real NWS page before first 10:05 AM job

### High value, moderate effort
- **#1** (theta calibration on NWP departures) — requires NWP curve in calibration loop
- **#8** (ASOS 6-hour max field) — read `maxT6` from NWS GeoJSON
- **#12** (DST-aware day bounds) — systemic fix affecting 8 months of trading
- **#13** (post-rollover hard floor corruption) — verify March 17 fix is complete
- **#16** (DB indexes) — add before extended operation (weeks)
- **#17** (settlement calibration date) — pass `yesterday` explicitly
- **#27** (staleness indicators) — query already in DB, purely UI work

### Medium effort, significant accuracy gain
- **#2** (sigma: remove diurnal trend) — requires NWP curve plumbing into estimate_sigma
- **#4** (Brier scoring improvement) — needs design decision on scoring event definition
- **#11** (NWP blend horizon extension) — blend per-hour using whatever models have data
- **#24** (MCParams consolidation) — extract to shared utility module
- **#19** (Kelly min contracts) — 1-line change with behavioral impact

### Lower priority / cleanup
- **#10** (partition tolerance tightening) — conservative tightening
- **#21** (trade log MC params) — schema change required
- **#22** (Kelly ask=1.0 guard) — edge case, already caught by outer try/except
- **#23** (short position tracking) — only matters if system ever trades both directions
- **#26** (transaction boundaries) — wrap multi-step writes
- **#28** (histogram fix) — UI only, cosmetic but misleading near boundaries
- **#5**, **#6**, **#7** (Kalman R, F, OU defaults) — requires careful calibration
- **#25**, **#29**, **#30**, **#31**, **#32** (minor UI / transparency cleanup)
