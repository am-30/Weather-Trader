Here is a fully updated and condensed CLAUDE.md that accurately reflects the current state of the system:

```markdown
# CLAUDE.md — Kalshi Weather Trading System
# Last Updated: March 16, 2026

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
│   ├── asos_fetcher.py      # IEM bulk gap-fill (primary), AVWX METAR
│   │                        # (secondary), NWS latest (last resort +
│   │                        # max6h_f); 2-min schedule, 4-min API rate limit
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
- Update step: on each new ASOS reading (scheduler fires every 2 min,
  API rate-limited to ≤every 4 min; METARs arrive every ~5–30 min)
- Predict step: every hourly NWP delta
- Joseph form covariance update for numerical stability
- Q_temp=0.1, Q_bias=0.05, R=0.6 (in settings.py)
- Cold-start: sigma/theta calibration meaningless for first ~30
  days of operation. No warm-start logic exists yet.

### Monte Carlo — NWP Anchor Fix (Critical)
- Process: Ornstein-Uhlenbeck (NOT geometric Brownian motion)
- Core equation: dT = theta*(mu_t - T_t)*dt + sigma*sqrt(dt)*Z
- dt = 5/60 hours

NWP ANCHOR OFFSET (implemented — do not remove):
  anchor_weight = 1 - hours_to_peak / peak_hour_idx
    (peak_hour_idx = argmax(nwp_curve); fallback = 1.0 when == 0)
  nwp_anchor_offset = (T0 - nwp_curve[hour_offset]) * anchor_weight
  mu_t = nwp_curve[hour_idx] + nwp_anchor_offset + bias +
         drift_adj

anchor_weight ramps from 0 (far from peak, NWP curve dominates)
to 1 (at/past peak, attractor fully anchored to T0). This prevents
the original full-offset from permanently projecting an early-
morning T0 gap onto the afternoon peak. As Kalman bias warms up
over weeks, nwp_anchor_offset trends toward zero naturally.

- Hard floor: paths_max array initialized at current_max_observed
- Vectorized: pre-generate full Z matrix before loop
- Returns full distribution dict including percentiles

### hour_offset — ALWAYS ET HOUR (same-day) or 0 (stitched)
For same-day simulations, hour_offset is the current Eastern Time
hour. Open-Meteo is called with timezone="America/New_York", so
nwp_curve is ET-indexed from midnight ET. Using UTC hour causes
a systematic ~4-hour index shift (EDT = UTC−4). Any new code
touching hour_offset must use:
datetime.now(timezone.utc).astimezone(pytz.timezone("America/New_York")).hour

For the post-6 PM stitched curve, hour_offset=0 always (index 0 of
the stitched array IS the current wall-clock time). Do not apply the
ET-hour value as an index into a stitched curve.

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
| UTC hour as nwp_curve index | ET hour — nwp_curve is ET-indexed (Open-Meteo called with timezone=America/New_York) |
| Integer CDF strike boundaries | Half-integer boundaries (39.5°F etc.) |
| RSA PKCS1v15 signing | RSA-PSS with MGF1(SHA256) + DIGEST_LENGTH salt |
| No position tracking | get_positions() reduces Kelly by current exposure |
| No startup recovery | Hard floor catch-up + missed calibration + CLI confirmation |
| Drift calibrated from yesterday only | 7-day rolling window, pooled errors |
| Stage 6 in original UI plan | Built — NWP accuracy chart, weight history, calibration scatter, snapshot replay |
| Brier score uses latest NWP forecast | Uses first fetch in [10 AM, 1 PM) ET window via `get_morning_nwp_forecasts()` — latest fetch introduces lookback bias from intraday model revisions |
| Scheduler guard via globals() | Stored on `sys` module (`sys._kalshi_scheduler_started`) — globals() is reset by Streamlit on every script rerun; sys persists for the process lifetime |

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

### ASOS Fetch Architecture (as of session 5)

fetch_current_observation() tries three sources in order:
  1. IEM Mesonet bulk gap-fill (PRIMARY)
     _fetch_iem_since(last_stored_timestamp) — one CSV request
     returns ALL readings since the last one in the DB. Zero gaps
     even if the scheduler was down or IEM was momentarily slow.
     IEM ingests NOAA data 1–3 min faster than the NWS public API.

  2. Aviation Weather Center METAR JSON (SECONDARY)
     _fetch_aviationweather_metar() — hits aviationweather.gov
     /api/data/metar. KBOS issues SPECI (special) METARs on any
     significant condition change, making this a useful gap-filler
     when IEM returns nothing for the current tick.

  3. NWS /observations/latest (LAST RESORT)
     _fetch_nws_latest() — retained because it is the only source
     for max6h_f (6-hour ASOS max, captures sub-threshold peaks
     suppressed by the 0.5°C persistence filter).

Rate-limit guard: _last_asos_fetch_utc module-level variable.
If last API call < asos_min_fetch_interval_minutes (default 4 min)
ago, returns cached DB reading without touching any API.
Scheduler fires every 2 min; real API calls ≤ every 4 min.
asos_min_fetch_interval_minutes and aviationweather_api_base_url
are overridable via env vars.

Hard floor is updated in a loop over all new readings, not just
the latest. max6h_f only available on the NWS fallback path.

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

## MCParams Construction

All four construction sites (trader.py, app.py x2, calibrator.py)
call the single shared factory: `quant/mc_params_builder.py` →
`build_mc_params()`. This was consolidated in session 4.

### Post-6 PM rollover — stitched NWP curve (session 6)

After the 6 PM ET rollover (`target_date > now_et.date()`),
`build_mc_params()` uses a stitched NWP curve instead of starting
from midnight tomorrow:

  bridge   = today_curve[current_et_hour : 24]  (tonight → 11 PM)
  full_day = tomorrow_curve[0 : 24]             (midnight → 11 PM tomorrow)
  stitched = bridge + full_day

  hour_offset = 0  (index 0 = current wall-clock time in stitched curve)
  is_future_day = False  (anchor offset active; T0 physically valid)
  day_fraction_remaining = len(stitched) / 24.0  (~1.17–1.25)

This allows the OU simulation to correctly price overnight temperature
peaks (e.g., daily high at 1 AM when a warm front passes) that would
be missed if the simulation jumped straight to midnight tomorrow.

Fallback: if today's NWP is not yet in the DB, reverts to the prior
behaviour (tomorrow's curve, hour_offset=1/0, is_future_day=True).

At rollover, `job_rollover_check()` re-fetches today's NWP (in
addition to pre-fetching tomorrow's) so the bridge hours reflect the
latest model run.

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

5. [RESOLVED] MCParams consolidated in quant/mc_params_builder.py
   (session 4); stitched overnight curve added (session 6)

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

10. Kalman bias cold-starts at 0.0 every new trading day
    bias resets to 0.0 on each daily rollover regardless of
    yesterday's converged state. No warm-start from prior day's
    final Kalman state. Keeps anchor offset active longer than
    necessary and slows effective bias convergence.

11. First-startup hard floor gap
    If the app starts for the first time mid-day with no ASOS
    readings yet in DB, hard-floor catch-up returns empty and
    current_max_observed stays at −999 until the first live
    fetch. If the true daily max occurred before startup it
    will be missed entirely.

12. Settlement preliminary high depends on ASOS completeness
    job_check_settlement() at 7 PM uses max(ASOS readings).
    If ASOS was offline during peak hours, preliminary value
    may be lower than true peak.

13. Position tracking is long-exposure only
    get_positions() reduces Kelly by long exposure but not
    short (NO) positions. Works for current single-direction
    strategy only.

### LOW PRIORITY / CLEANUP

14. KALSHI_ENV=demo is cosmetic — wire it or remove it.

15. Startup catch-up equality check is a fragile heuristic
    (final_official_high == current_max_observed) false-positives
    when CLI value happens to match ASOS exactly. Harmless but
    conceptually wrong.

16. Test coverage gaps
    Zero tests for: kalshi_fetcher.py, orchestrator.py, app.py,
    nws_cli_fetcher.py, db_manager.py.
    53 tests passing in test_kalman.py, test_monte_carlo.py,
    test_ingestion.py.

---

## What Has Been Built (as of March 16, 2026)

- [x] Phase 1: Config, schemas, db_manager
- [x] Phase 2: ASOS + NWP + Kalshi + NWS CLI fetchers
- [x] Phase 3: Kalman filter + Monte Carlo (NWP anchor fix applied)
- [x] Phase 4: Calibrator (7-day drift window) + snapshot manager
- [x] Phase 5: Execution engine + trader + position tracking
- [x] Phase 6: Streamlit — Trading Desk, Visualizer, Calibration,
              Model Transparency (Stages 1–6 complete)
- [x] Phase 7: Orchestrator + two-phase settlement + startup
              catch-up sequence
- [ ] End-to-end live cycle validation
- [ ] NWS CLI regex verification against real page
```
March 17 Updates
- Fixed hard floor bug where once trader roller over to tracking the market for the next day at 6 PM, the highest temp observed in the current day was being set as the hard floor for that ensuing day. The hard floor value is also now being rounded down to account for oddities in how the NWS rounds their max recorded temps.