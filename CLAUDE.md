Here is a fully updated and condensed CLAUDE.md that accurately reflects the current state of the system:

```markdown
# CLAUDE.md — Kalshi Weather Trading System
# Last Updated: March 15, 2026

You are a senior quantitative developer and software engineer 
working on an existing, functionally complete automated 
weather-trading system targeting the Kalshi Daily Maximum 
Temperature market for Boston Logan Airport (KBOS). All 7 phases 
have been built. You are in maintenance, debugging, and 
improvement mode — not greenfield development.

Read this file completely before touching any code.

---

## System Status: FUNCTIONALLY COMPLETE — NOT YET VALIDATED

All 7 phases are built and running. The system has never completed
a full end-to-end validated cycle with live data from a cold 
start. Do not assume any module works correctly until confirmed 
against live observations.

---

## Tech Stack (Final — Do Not Suggest Alternatives)

- Database: Replit native PostgreSQL via DATABASE_URL env variable
- ORM: SQLAlchemy with psycopg2-binary
- UI: Streamlit with Plotly charts
- Scheduler: APScheduler BackgroundScheduler running as background
  thread inside Streamlit process (not a separate process)
- HTTP client: httpx with tenacity retry logic
- Validation: Pydantic v2
- Logging: structlog exclusively — never print()
- Python: 3.11+

---

## Project Structure

kalshi_weather_trader/
├── CLAUDE.md
├── ARCHITECTURE.md
├── requirements.txt
├── .env
├── config/
│   └── settings.py          # pydantic-settings, all OU/Kalman/
│                            # trading params as constants
├── db/
│   ├── schema.sql           # PostgreSQL table definitions
│   ├── db_manager.py        # All read/write helpers
│   └── schemas.py           # ALL Pydantic models — import from
│                            # here only, never redefine elsewhere
├── ingestion/
│   ├── asos_fetcher.py      # NWS 5-min + IEM staleness fallback
│   ├── nwp_fetcher.py       # Open-Meteo HRRR/GFS/ECMWF
│   ├── kalshi_fetcher.py    # RSA-PSS auth, market queries,
│   │                        # positions
│   └── nws_cli_fetcher.py   # NWS Climate Summary for settlement
├── quant/
│   ├── kalman_filter.py     # 2D Kalman [temp, bias], Joseph form
│   └── monte_carlo.py       # OU simulation, NWP anchor fix,
│                            # CDF interpolation
├── execution/
│   └── trader.py            # Kill switch → MC → Kelly → orders
├── calibration/
│   └── calibrator.py        # Brier scores, 7-day drift window,
│                            # model weights, snapshots
├── scheduler/
│   └── orchestrator.py      # APScheduler jobs, startup catch-up,
│                            # two-phase settlement
├── ui/
│   └── app.py               # 4-tab Streamlit dashboard
└── tests/
    ├── test_kalman.py
    ├── test_monte_carlo.py   # 21 tests passing, includes
    │                        # NWP anchor tests
    └── test_ingestion.py

---

## Database Tables

markets, nwp_forecasts, asos_readings, system_state,
intraday_snapshots, trade_logs

Critical: kalshi_strike columns are NUMERIC(5,1) — NOT integer.
Live Kalshi strikes are decimals (e.g. 44.5°F).
_migrate_kalshi_strike_columns() runs idempotently on every 
startup to ensure this.

Hard floor updates use single-statement SQL:
UPDATE ... SET col = GREATEST(col, :val)
Never use a preceding SELECT — atomicity depends on this.

---

## Key Mathematical Specifications

### Kalman Filter
- 2D state vector: [T_t (true temp), B_t (model bias)]
- Update step: every 5-minute ASOS reading
- Predict step: every hourly NWP delta
- Joseph form covariance update for numerical stability
- Q_temp=0.1, Q_bias=0.05, R=0.3 (in settings.py)
- Cold-start: sigma/theta calibration meaningless for first ~30
  days of operation. No warm-start logic exists yet.

### Monte Carlo — NWP Anchor Fix (Critical)
- Process: Ornstein-Uhlenbeck (NOT geometric Brownian motion)
- Core equation: dT = theta*(mu_t - T_t)*dt + sigma*sqrt(dt)*Z
- dt = 5/60 hours

NWP ANCHOR OFFSET (implemented — do not remove):
  nwp_anchor_offset = T0 - nwp_curve[hour_offset]
  mu_t = nwp_curve[hour_idx] + nwp_anchor_offset + bias + 
         drift_adj

This fix resolves sigma inflation (~2°F overshoot) caused by the
OU attractor pulling paths from T0 toward the absolute NWP level
at step 0. The offset anchors step-0 attractor to T0 exactly and
follows NWP rate-of-change thereafter. As Kalman bias warms up
over weeks, nwp_anchor_offset trends toward zero naturally.

- Hard floor: paths_max array initialized at current_max_observed
- Vectorized: pre-generate full Z matrix before loop
- Returns full distribution dict including percentiles

### hour_offset — ALWAYS UTC HOUR
hour_offset into nwp_curve must be UTC hour, NOT Eastern hour.
Using Eastern hour causes a systematic ~5-hour bias in all 
probability estimates. This was a confirmed bug that has been 
fixed. Any new code touching hour_offset must use 
datetime.now(timezone.utc).hour.

### CDF and Strike Pricing
- Settlement boundaries at half-integers (39.5°F, 40.5°F etc.)
  because NWS rounds to nearest integer
- Use _interpolate_cdf() + compute_normalized_market_probs()
  with partition gap detection and normalization
- Do NOT use simple .get(key, default) on CDF dict

### Position Sizing (Fractional Kelly at 25%)
  b = (1/ask_decimal) - 1
  kelly = (p*b - (1-p)) / b
  contracts = min(0.25*kelly*MAX_SIZE / (ask*100), MAX_SIZE)
  
evaluate_and_trade() fetches existing positions via 
get_positions() and reduces Kelly contracts by current long 
exposure before sizing.

---

## Architectural Deviations from Original Spec

These are final decisions — do not revert them:

| What Spec Said | What Was Built |
|---|---|
| Separate Trading Engine workflow | APScheduler background thread inside Streamlit |
| gfs_seamless as GFS source | gfs_global (pure GFS) — seamless was identical to HRRR near-term |
| final_official_high from ASOS max only | Two-phase: ASOS preliminary 7 PM → NWS CLI authoritative 10:05 AM |
| SmallInteger for kalshi_strike | NUMERIC(5,1) — live strikes are decimals |
| Eastern hour as nwp_curve index | UTC hour — Eastern caused ~5-hour systematic bias |
| Integer CDF strike boundaries | Half-integer boundaries (39.5°F etc.) |
| RSA PKCS1v15 signing | RSA-PSS with MGF1(SHA256) + DIGEST_LENGTH salt |
| No position tracking | get_positions() reduces Kelly by current exposure |
| No startup recovery | Hard floor catch-up + missed calibration + CLI confirmation |
| Drift calibrated from yesterday only | 7-day rolling window, pooled errors |
| Stage 6 in original UI plan | Deferred — not yet built |

---

## Settlement Architecture (Two-Phase)

Phase 1 — 7:00 PM ET (job_check_settlement):
  Uses max(ASOS readings) as preliminary final_official_high
  Sets market_status='settled', kills auto-trading
  Source tagged as "asos_preliminary" in logs

Phase 2 — 10:05 AM ET next day (job_confirm_settlement):
  Calls fetch_official_daily_high(yesterday) from NWS CLI
  Overwrites preliminary value with authoritative NWS figure
  Triggers run_full_calibration() after confirmation
  Uses calendar date, NOT get_target_date() (which has already
  rolled to tomorrow)

Startup catch-up: on restart, checks if yesterday's 
final_official_high is None and attempts one CLI fetch.

---

## ASOS Data — Critical Behaviors to Know

The ASOS sensor uses a 0.5°C persistence filter before reporting.
Temperature only changes in display when reading differs from 
previous by ≥0.5°C. This produces "sticky" values:
  2.0°C=35.6°F, 2.5°C=36.5°F, 3.0°C=37.4°F, 3.9°C=39.2°F, etc.

These sticky values are separated by 0.9°F gaps in Fahrenheit.
Whole-number Fahrenheit values (39°F, 40°F, 41°F) almost never 
appear in the tabular feed because they are not clean Celsius 
conversions.

The 6-hour maximum field in METARs is encoded separately from the
tabular display and captures the true intraday peak including
sub-threshold spikes. The hard floor should read from this field,
NOT just the tabular air temperature column.

The daily maximum in the NWS CLI can exceed the highest tabular
value by ~0.2–0.4°F on average due to the persistence filter 
causing the true peak to fall between threshold crossings. Near 
strike boundaries this is meaningful.

---

## Kalshi API Notes

Authentication: RSA-PSS with MGF1(SHA256) and DIGEST_LENGTH salt
  (NOT PKCS1v15 — this was a confirmed bug that is fixed)

_normalize_market() converts yes_bid_dollars/yes_ask_dollars 
(floats in [0,1]) to cent fields for all downstream code.
WARNING: No input guard exists. If Kalshi returns integers in 
[0,100] instead of floats in [0,1], all edge calculations 
silently inflate 100×. Add validation before going live.

Market query uses 3-strategy fallback:
  1. /markets?series_ticker=KXHIGHTBOS + client-side filter
  2. /markets?event_ticker=KXHIGHTBOS-{date}
  3. /events/KXHIGHTBOS-{date}/markets

Strike extraction: floor_strike field directly from API response.
extract_strike_from_ticker() retained as fallback only.

KALSHI_ENV=demo is validated but cosmetic — it is never used to 
select the API URL. URL always comes from kalshi_api_base_url 
directly. Do not rely on this setting for environment switching.

---

## Non-Negotiable Coding Rules

- Pydantic models defined ONLY in db/schemas.py — never redefine
- structlog for ALL logging — never print()
- tenacity retry on ALL external API calls (3 retries, exp backoff)
- All secrets from environment variables — never hardcode
- SQLAlchemy for ALL database operations — never raw psycopg2
- Type annotations on every function
- Full docstrings: Args, Returns, Raises on every function
- try/except on every DB write and external API call
- Timestamps: store as UTC datetime objects, never strings
- Display conversion to US/Eastern at render time only
- Temperatures: Fahrenheit floats, one decimal precision
- DRY_RUN env variable must be checked before every real order
- Kill switch (auto_trade_enabled) must be checked before every
  trade evaluation — not just order placement

---

## MCParams Construction — Duplication Warning

MCParams is independently constructed in THREE places:
  1. trader.py (evaluate_and_trade)
  2. ui/app.py Tab 1 edge table
  3. ui/app.py Stage 3 Model Transparency

If hour_offset or drift_adj logic is updated in one place it will
silently diverge in the others. Before modifying MCParams 
construction anywhere, search all three locations and update 
them consistently. This is a known tech debt item.

---

## Open Issues — Prioritized

### BLOCKING (do not go live until resolved)

1. No end-to-end validated cycle
   Full fetch → Kalman update → MC → trade eval → snapshot has
   never been confirmed from cold start with live data.
   To trigger manually: Calibration tab → Fetch All NWP Models,
   then watch scheduler logs for ASOS fetch completion.

2. NWS CLI regex unverified
   fetch_official_daily_high() patterns (CLIMATE SUMMARY FOR,
   MAXIMUM\s+) have never been run against a real NWS page.
   The 10:05 AM settlement job depends on this working.
   Verify before the first 10:05 AM firing.

3. _normalize_market() has no input guard
   Assumes yes_bid_dollars/yes_ask_dollars are in [0,1].
   If Kalshi changes field format, all edge calculations inflate
   100× silently. Add validation: assert 0 <= value <= 1.

4. Hard floor atomicity unconfirmed
   Verify db_manager.update_hard_floor() uses exactly:
   UPDATE markets SET current_max_observed = 
     GREATEST(current_max_observed, :val) WHERE market_id = :id
   No preceding SELECT. Atomicity depends on single statement.

### HIGH PRIORITY

5. MCParams constructed in 3 independent places (see above)

6. NWP anchor fix unvalidated against live data
   The fix hasn't been confirmed against a real trading day.
   Monitor mean_max vs actual final high for first few days.
   Watch for overcorrection on rising-temperature mornings where
   T0 is genuinely below NWP (fog burn-off, sea breeze).

### MEDIUM PRIORITY

7. Stage 4 histogram uses synthetic normal distribution
   numpy.random.normal(mean_max, std_max, 5000) approximation
   does not reflect true hard-floor truncation shape from actual
   paths_max array. Pass actual paths_max through to fix.

8. Blended forecast truncates to shortest model curve
   If HRRR=18h and GFS=24h, blend is cut to 18h. Should blend
   per-hour using whatever models have data at that hour.

9. Kalman cold-start no warning
   sigma/theta calibration meaningless for first ~30 days.
   No UI warning, no warm-start period logic. System silently
   uses uncalibrated defaults.

10. Settlement preliminary high depends on ASOS completeness
    job_check_settlement() at 7 PM uses max(ASOS readings).
    If ASOS was offline during peak hours, preliminary value
    may be lower than true peak.

11. Position tracking is long-exposure only
    get_positions() reduces Kelly by long exposure but not
    short (NO) positions. Works for current single-direction
    strategy only.

### LOW PRIORITY / CLEANUP

12. Stage 6 Historical Calibration Performance — not built
    Needs: multi-day Brier score time series, date-picker 
    replay, weight convergence chart.

13. KALSHI_ENV=demo is cosmetic — wire it or remove it.

14. Startup catch-up equality check is a fragile heuristic
    (final_official_high == current_max_observed) false-positives
    when CLI value happens to match ASOS exactly. Harmless but
    conceptually wrong.

15. Test coverage gaps
    Zero tests for: kalshi_fetcher.py, orchestrator.py, app.py,
    nws_cli_fetcher.py, db_manager.py.
    21 tests passing in test_kalman.py, test_monte_carlo.py,
    test_ingestion.py.

---

## What Has Been Built March 16 2025 10:34 AM

- [x] Phase 1: Config, schemas, db_manager
- [x] Phase 2: ASOS + NWP + Kalshi + NWS CLI fetchers
- [x] Phase 3: Kalman filter + Monte Carlo (NWP anchor fix applied)
- [x] Phase 4: Calibrator (7-day drift window) + snapshot manager
- [x] Phase 5: Execution engine + trader + position tracking
- [x] Phase 6: Streamlit — Trading Desk, Visualizer, Calibration,
              Model Transparency (Stages 1–5; Stage 6 deferred)
- [x] Phase 7: Orchestrator + two-phase settlement + startup 
              catch-up sequence
- [ ] Stage 6 Historical Calibration Performance (deferred)
- [ ] End-to-end live cycle validation
- [ ] NWS CLI regex verification against real page
```
Changes Made This Session                           

1. UTC → ET hour fix (app.py — Tab 1, Stage 2, Stage
 3)                                               

The nwp_curve is ET-indexed (Open-Meteo is called
with timezone: America/New_York). Three places in
the UI were indexing it with
datetime.now(timezone.utc).hour instead of the ET
hour, causing a systematic ~4-hour index shift in
March (EDT = UTC−4).

- Tab 1 "NWP Blended Now" metric: now_hour_utc →
now_hour_et
- Stage 2 blended NWP display: now_hour_utc_s2 →
now_hour_et_s2
- Stage 3 MCParams construction: hour_utc_s3 →
hour_et_s3; parameter table label "Hour Offset
(UTC)" → "Hour Offset (ET)"

2. Stage 3 day_fraction fix (app.py)

Was using 1.0 - hour_offset_s3 / 24.0 which assumes
the trading day ends at midnight ET. The NWS day
runs midnight EST → midnight EST (1 AM EDT → 1 AM
EDT), so this was 1 hour short. Replaced with
get_remaining_day_fraction() — the same NWS-pinned
function used by trader.py.

3. NWP anchor offset — time-weighted formula
(monte_carlo.py)

Was: nwp_anchor_offset = T0 - nwp_curve[hour_offset]
 — full gap applied permanently regardless of time
of day.

Now: nwp_anchor_offset = (T0 -
nwp_curve[hour_offset]) * anchor_weight

Where anchor_weight = 1 - hours_to_peak /
peak_hour_idx:
- peak_hour_idx = argmax(nwp_curve) — the ET hour of
 the forecast peak
- Weight = 0 at the start of the day (far from
peak), 1 at/past the peak
- Fallback weight = 1.0 when peak_hour_idx == 0
(flat or declining curve)

Rationale: The original full offset projected the
current morning gap permanently onto the afternoon
peak, overcorrecting when the gap is simply due to
normal morning warming. The weighted formula treats
early-day gaps as low-confidence signals that grow
in weight as you approach the peak and have more
sustained evidence.

4. Tests updated (test_monte_carlo.py)

Rewrote TestNWPAnchor class. Renamed both original
tests, added a third
(test_rising_nwp_at_peak_full_anchor) that
explicitly validates weight=1.0 behavior at the peak
 hour. Total tests: 50 → 52, all passing.

---
Deviations from CLAUDE.md

Critical contradiction — hour_offset timezone

CLAUDE.md currently says:
▎ ### hour_offset — ALWAYS UTC HOUR
▎ hour_offset into nwp_curve must be UTC hour, NOT
Eastern hour. Using Eastern hour causes a systematic
 ~5-hour bias in all probability estimates. This was
 a confirmed bug that has been fixed.

And in the deviations table:
▎ Eastern hour as nwp_curve index | UTC hour —
Eastern caused ~5-hour systematic bias

This is now wrong. Open-Meteo is called with
"timezone": "America/New_York", making nwp_curve
ET-indexed. The UTC-hour fix documented in CLAUDE.md
 was itself a misdiagnosis — it happened to work
differently or the original symptom was in a
different code path. We have confirmed via the
−13.34°F anchor offset diagnostic that UTC hour was
the bug and ET hour is correct. CLAUDE.md needs this
 section updated.

NWP anchor offset formula

CLAUDE.md documents:
▎ nwp_anchor_offset = T0 - nwp_curve[hour_offset]
▎ mu_t = nwp_curve[hour_idx] + nwp_anchor_offset +
bias + drift_adj

The actual formula is now:
▎ nwp_anchor_offset = (T0 - nwp_curve[hour_offset])
* anchor_weight
▎ where anchor_weight = 1 - hours_to_peak /
peak_hour_idx

The section also says "do not remove" — the concept
is preserved, the implementation is modified.

---
Known Issues / To-Dos

Blocking (do not go live until resolved)

1. No end-to-end validated cycle — full fetch →
Kalman → MC → trade eval → snapshot has never been
confirmed from cold start with live data.
2. NWS CLI regex unverified —
fetch_official_daily_high() regex patterns have
never run against a real NWS page. The 10:05 AM
settlement job depends on this.
3. _normalize_market() has no input guard — assumes
yes_bid/ask_dollars are floats in [0,1]. If Kalshi
returns integers in [0,100], edge calculations
inflate 100× silently.
4. Hard floor atomicity unconfirmed — verify
update_hard_floor() is a single UPDATE ... SET col =
 GREATEST(col, :val) statement with no preceding
SELECT.

High Priority

5. MCParams constructed in 3 independent places —
trader.py, app.py Tab 1, app.py Stage 3. The anchor
weight logic lives inside run_simulation() so the
formula change is consistent automatically, but
hour_offset and drift_adj construction logic can
still silently diverge between the three sites.
6. Weighted anchor formula unvalidated against live
data — monitor mean_max vs. actual settlement for
the first week. The anchor_weight ramp (linear in
hours-to-peak) was chosen as reasonable but is not
empirically calibrated. Watch for overcorrection on
fast-warming mornings.
7. CLAUDE.md requires updating — the hour_offset
timezone section and the anchor offset formula are
now incorrect as documented.

Medium Priority

8. Kalman cold-starts every new trading day — bias =
 0.0 on every new day regardless of yesterday's
converged state. No warm-start from prior day's
final Kalman state. This keeps the anchor offset
active longer than necessary each day during the
early-operation period.
9. First-startup hard floor gap — if the app starts
for the first time today, no ASOS readings are
stored yet. The hard-floor catch-up query returns
empty and current_max_observed starts at −999 until
the first live fetch. If today's true max occurred
before startup, it will be missed.
10. Stage 4 histogram uses synthetic normal —
numpy.random.normal(mean_max, std_max, 5000) instead
 of actual paths_max array. Doesn't reflect
hard-floor truncation shape.
11. NWP blending truncates to shortest model — fixed
 in get_nwp_curve() (blends per-hour using available
 models), but get_blended_forecast()
(predicted_daily_high) still uses per-model max, not
 per-hour blending.
12. Kalman cold-start no UI warning — no indication
to the user that bias is uncalibrated for the first
~30 days.

Low Priority / Cleanup

13. Stage 6 Historical Calibration Performance — not
 built. Needs multi-day Brier score time series,
date-picker replay, weight convergence chart.
14. KALSHI_ENV=demo is cosmetic — never used to
select the API URL. Wire it or remove it.
15. Test coverage gaps — zero tests for
kalshi_fetcher.py, orchestrator.py, app.py,
nws_cli_fetcher.py, db_manager.py.
16. Startup catch-up equality check is fragile —
final_official_high == current_max_observed
heuristic produces false positives when CLI value
happens to match ASOS exactly.