# CLAUDE.md — Kalshi Weather Trading System

You are a senior quantitative developer and software engineer. You are building a professional-grade, automated weather-trading system targeting the Kalshi Daily Maximum Temperature market for Boston Logan Airport (KBOS). This is not a toy project. Every module must be production-quality: type-annotated, logged, error-handled, and independently testable.

## Project Context
This system was designed through an extensive architecture session. 
The full conversation context is being provided at session start.
Read all provided documents carefully before writing any code.

## Core Decisions Made During Architecture
- Database: Replit native PostgreSQL via DATABASE_URL (NOT Firebase/Firestore)
- UI: Streamlit
- Scheduler: APScheduler  
- Station: KBOS (Boston Logan Airport) hardcoded
- All timestamps: UTC internally, US/Eastern at display only
- All temperatures: Fahrenheit floats, one decimal precision
- Authentication: RSA key-based for Kalshi API v2
- Position sizing: Fractional Kelly at 25%
- Simulation: Ornstein-Uhlenbeck process, 10,000 paths, vectorized NumPy
- Kalman Filter: 2D state vector (temperature + bias), Joseph form covariance update
- Hard floor: current_max_observed is atomically maintained via PostgreSQL GREATEST()
- Kill switch: auto_trade_enabled flag in markets table
- DRY_RUN: always check environment variable before placing real orders

## Non-Negotiable Coding Rules
- Pydantic v2 for all data models
- structlog for all logging — never print()
- tenacity retry decorators on all external API calls
- python-dotenv for all secrets
- SQLAlchemy for all database operations — never raw psycopg2
- Type annotations on every function
- Full docstrings with Args, Returns, Raises on every function
- Try/except on every database write and external API call
- No circular imports — schemas only imported from db/schemas.py

## What Has Been Built
- [ ] Phase 1: Config, schemas, db_manager
- [ ] Phase 2: ASOS + NWP + Kalshi fetchers
- [ ] Phase 3: Kalman filter + Monte Carlo engine
- [ ] Phase 4: Calibrator + snapshot manager
- [ ] Phase 5: Execution engine + trader
- [ ] Phase 6: Streamlit command center
- [ ] Phase 7: Orchestrator + scheduler

## Known Issues / Decisions
(update this as we build)
```


---
```
I am building a quantitative weather trading system for Kalshi's 
Boston Daily Maximum Temperature market. Here is the complete context 
you need before writing any code:

SYSTEM OVERVIEW:
Automated trading system targeting KBOS (Boston Logan Airport) 
temperature markets on Kalshi. Ingests 5-minute ASOS data from NWS API 
with IEM fallback, hourly NWP forecasts from Open-Meteo (HRRR/GFS/ECMWF), 
maintains a 2D Kalman Filter tracking true temperature and model bias, 
runs 10,000-path Ornstein-Uhlenbeck Monte Carlo simulations to price 
probability of exceeding strike temperatures, and executes trades on 
Kalshi when model edge exceeds threshold.

TECH STACK DECISIONS (final, do not suggest alternatives):
- Database: Replit native PostgreSQL via DATABASE_URL env variable
- ORM: SQLAlchemy with psycopg2-binary
- UI: Streamlit with Plotly charts
- Scheduler: APScheduler BackgroundScheduler
- HTTP client: httpx with tenacity retry logic
- Validation: Pydantic v2
- Logging: structlog exclusively
- Python: 3.11+

PROJECT STRUCTURE:
kalshi_weather_trader/
├── CLAUDE.md
├── ARCHITECTURE.md
├── requirements.txt
├── .env
├── config/
│   ├── __init__.py
│   └── settings.py
├── db/
│   ├── __init__.py
│   ├── schema.sql
│   ├── db_manager.py
│   └── schemas.py
├── ingestion/
│   ├── __init__.py
│   ├── asos_fetcher.py
│   ├── nwp_fetcher.py
│   └── kalshi_fetcher.py
├── quant/
│   ├── __init__.py
│   ├── kalman_filter.py
│   └── monte_carlo.py
├── execution/
│   ├── __init__.py
│   └── trader.py
├── calibration/
│   ├── __init__.py
│   └── calibrator.py
├── scheduler/
│   ├── __init__.py
│   └── orchestrator.py
├── ui/
│   ├── __init__.py
│   └── app.py
└── tests/
    ├── __init__.py
    ├── test_kalman.py
    ├── test_monte_carlo.py
    └── test_ingestion.py

DATABASE SCHEMA:
PostgreSQL tables: markets, nwp_forecasts, asos_readings, 
system_state, intraday_snapshots, trade_logs. 
Schema defined in db/schema.sql and auto-created on startup.
Critical: current_max_observed updates use PostgreSQL GREATEST() 
function for atomic hard floor maintenance.

KEY MATHEMATICAL SPECIFICATIONS:
Kalman Filter:
- 2D state vector: [T_t (true temp), B_t (model bias)]
- Update step triggered by ASOS readings every 5 minutes
- Predict step triggered by NWP hourly deltas
- Use Joseph form covariance update for numerical stability
- Q_temp=0.1, Q_bias=0.05, R=0.3 (stored in config/settings.py)

Monte Carlo:
- Ornstein-Uhlenbeck process (NOT geometric Brownian motion)
- dT = theta*(mu_t - T_t)*dt + sigma*sqrt(dt)*Z
- mu_t = NWP forecast + Kalman bias correction + time-window drift
- dt = 5/60 hours (5-minute steps)
- Hard floor: paths_max initialized at current_max_observed
- Fully vectorized using pre-generated NumPy random matrix
- Returns full distribution dict including percentiles

Position Sizing:
- Fractional Kelly at 25%
- b = (1/ask_decimal) - 1
- kelly = (p*b - (1-p)) / b
- contracts = min(0.25*kelly*MAX_SIZE / (ask*100), MAX_SIZE)

CRITICAL SYSTEM BEHAVIORS:
1. Kill switch: check auto_trade_enabled from DB before every trade
2. DRY_RUN env variable: simulate but never place real orders if true
3. 6 PM Eastern rollover: target_date shifts to tomorrow after 18:00
4. Hard floor: current_max_observed never decreases, only increases
5. All external APIs use tenacity: 3 retries, exponential backoff
6. NWS API requires User-Agent header — use descriptive string
7. IEM mesonet is ASOS fallback if NWS returns stale data (>15 min)

TRADING LOGIC:
- Edge threshold: EDGE_THRESHOLD env variable (default 0.05)
- Buy YES if: fair_value > ask + threshold
- Buy NO if: fair_value < bid - threshold
- Log all decisions to trade_logs table including no-trade decisions

STREAMLIT DASHBOARD TABS:
Tab 1 - Trading Desk: live ASOS temp, max observed, Kalman estimate, 
edge table by strike, kill switch button, recent trades
Tab 2 - Visualizer: ASOS history + NWP curves + MC percentile band 
+ hard floor line + strike lines, all on one Plotly chart
Tab 3 - Calibration: model weights bar chart, drift adjustments, 
manual override sliders, force snapshot button

SCHEDULER JOBS (orchestrator.py):
- fetch_asos + kalman update: every 5 minutes
- fetch_nwp + kalman predict: every 60 minutes  
- evaluate_and_trade: every 5 minutes
- take_snapshot: every 2 hours
- midnight_calibration: daily at 00:05 Eastern
- rollover_check: every 30 minutes

NON-NEGOTIABLE CODING RULES:
- Every function: type annotations + full docstring (Args/Returns/Raises)
- Every DB write: wrapped in try/except, logged with structlog
- Every API call: tenacity retry decorator
- Every file: module-level docstring
- Pydantic models defined ONLY in db/schemas.py, imported everywhere else
- No print() statements anywhere — structlog only
- Timestamps: always store UTC datetime objects, never strings
- Secrets: always from environment variables, never hardcoded

I have a file called ARCHITECTURE.md in this project with the complete 
detailed specification for every module. Please read it now using your 
file reading capability before we begin.

We will build phase by phase. Do not write code until I say which phase 
to start. First, confirm you have read ARCHITECTURE.md and summarize 
what Phase 1 requires so I know you have full context.



  ● Session Summary — March 15, 2026  Pt 3                              

  Fixes Implemented (Comprehensive Audit)                         

  Data Integrity                                                  
  - kalshi_strike columns in intraday_snapshots and trade_logs
  migrated from SmallInteger → NUMERIC(5,1) via                   
  _migrate_kalshi_strike_columns() that runs idempotently on every
   startup. Decimal strikes like 44.5 were previously truncated to
   44.
  - All strike type hints corrected from int → float throughout
  schemas.py, monte_carlo.py, trader.py

  Critical Pricing Bug
  - hour_offset in MCParams was being set to the Eastern hour
  (e.g. 15 for 3 PM ET) but nwp_curve is UTC-indexed. Fixed in
  trader.py and calibrator.py to use
  datetime.now(timezone.utc).hour, eliminating a ~5-hour
  systematic bias in all probability estimates

  Startup Catch-Up Logic (not in original spec)
  - Hard floor catch-up: on startup, scans all stored ASOS
  readings for the trading day and calls update_hard_floor() with
  the actual observed peak, recovering from any downtime during
  peak hours
  - Missed calibration catch-up: on startup, checks
  last_calibrated_utc and runs run_full_calibration() immediately
  if midnight calibration was missed

  Position Tracking (not in original spec)
  - Added get_positions() to KalshiFetcher — calls
  /portfolio/positions
  - evaluate_and_trade() now fetches existing positions before
  sizing and reduces Kelly contracts by current exposure,
  preventing over-sizing after restarts

  Settlement Detection (not in original spec)
  - Added job_check_settlement() running every 30 min after 7 PM
  ET
  - Computes official daily high from stored ASOS readings, writes
   final_official_high and market_status='settled' to the markets
  table
  - Uses actual calendar date, not get_target_date() which has
  already rolled over to tomorrow by then

  ---
  Kalshi Market Fetching — Fully Reworked

  Field name bug: API returns yes_bid_dollars/yes_ask_dollars as
  floats in [0,1], not yes_bid/yes_ask in cents. Added
  _normalize_market() static method to KalshiFetcher that converts
   dollar fields to cent fields so all downstream code (trader,
  calibrator, UI) works without changes.

  Status filter: Changed from status=active API-side filter to
  fetching all and filtering client-side for status in {active,
  initialized}. Some markets are in initialized state before
  trading opens.

  Query strategy: get_temperature_markets() now tries 3 strategies
   in sequence:
  1. /markets?series_ticker=KXHIGHTBOS + client-side date/status
  filter (correct approach per API docs)
  2. /markets?event_ticker=KXHIGHTBOS-26MAR15
  3. /events/KXHIGHTBOS-26MAR15/markets

  Strike extraction: Added extract_strike_from_market(market:
  dict) static method that reads floor_strike directly from the
  API response (more reliable than ticker regex).
  extract_strike_from_ticker() retained as fallback.

  Strike labels (not in original spec): Added
  get_strike_label(market: dict) that generates human-readable
  range strings from floor_strike/cap_strike:
  - T38 → <38°F
  - B38.5 → 38–39°F
  - B54 (top bucket) → >54°F
  Falls back to ticker regex + estimated cap if API fields are
  absent.

  ---
  NWP Models

  GFS model corrected: gfs_seamless (Open-Meteo's near-term
  HRRR+GFS blend) was producing identical results to HRRR for
  same-day forecasts. Changed primary GFS model to gfs_global
  (pure GFS, ~25km, 4x daily), which gives genuinely independent
  forecasts. gfs_seamless demoted to fallback.

  ---
  UI — Trading Desk Edge Table

  Replaced snapshot-based single-strike edge table with a live
  multi-strike table:
  - Button-triggered ("Refresh Edge Table") rather than running on
   every render
  - Results cached in st.session_state to survive Streamlit reruns
  - Step-by-step diagnostics shown in expander: ticker queried,
  markets found with tickers listed, sample price fields from
  first market, strikes parsed, MC result count
  - Fallback mode when system state is missing: still shows all
  markets with bid/ask even if MC can't run
  - Flat NWP curve fallback when NWP data is missing (uses current
   Kalman temp × 24h)
  - Shows "Range" column instead of "Strike" using human-readable
  labels

  ---
  UI — Visualizer

  Blended forecast line: Now computed live from hourly NWP curves
  weighted by model_weights from system_state (defaults HRRR 50% /
   GFS 30% / ECMWF 20%). Previously it read from snapshots and was
   never visible because snapshots hadn't been taken.

  NWP model status expander: Below the chart, shows each model's
  DB status, predicted high, hours of data, blend weight, and line
   color. Auto-expands when no models are in DB.

  ---
  UI — Calibration Tab

  "Fetch All NWP Models" button fixed: st.rerun() was being called
   before st.success()/st.error() rendered, so nothing ever
  appeared. Fixed by saving results to st.session_state before
  calling st.rerun(), then displaying from state in a block that
  persists across reruns.

  Additional Kalshi API diagnostics added:
  - GET /events/{event_ticker}/markets (nested resource)
  - GET /markets?series_ticker=KXHIGHTBOS&status=active

  ---
  Known Issues / TODOs

  - No full fetch-update-snapshot cycle verified end-to-end yet —
  ASOS scheduler hasn't been confirmed running; dashboard metrics
  may still show N/A for Kalman estimates until first ASOS fetch
  completes
  - KALSHI_ENV=demo is cosmetic — the setting is validated but
  never used to select the URL; the URL always comes from
  kalshi_api_base_url directly. Either wire it or remove it.
  - Blended forecast truncates to shortest model curve — if HRRR
  provides 18h and GFS provides 24h, the blend is cut to 18h. A
  future improvement would blend per-hour with whatever models
  have data at that hour.
  - Settlement job depends on ASOS data completeness — if ASOS
  readings are sparse after 7 PM ET, final_official_high may be
  lower than the true peak. Consider adding NWS official
  observation as a data source for settlement.
  - Position tracking is additive-only — get_positions() reduces
  Kelly by current long exposure but doesn't account for short
  (NO) positions. Works correctly for the current single-direction
   strategy.
  - Calibrator and trader still reference hour_et — the variable
  is computed but now only used for AM/PM drift selection. It
  could be cleaned up to remove the ambiguity.

● Session Summary — March 15, 2026 Pt 4                           

What Was Built                                                  

NWS CLI Official High Fetcher (ingestion/nws_cli_fetcher.py) — 
NEW FILE                                                        
- Fetches the NWS Climate Summary (CLI) product for Boston from
https://forecast.weather.gov/product.php?site=BOX&product=CLI&is
suedby=BOS
- Public function: fetch_official_daily_high(target_date: date)
-> Optional[float]
- Cycles through versions 1–5 (newest first); stops early if a
version's report date is older than target_date (no point
scanning further back)
- Strict date validation: parses CLIMATE SUMMARY FOR [DAY]
[MONTH DD YYYY] and rejects any version whose date doesn't
exactly match target_date — prevents accepting an intraday
partial report or a prior day's report
- Missing-value guard: rejects M token (field not yet finalized)
 and any non-numeric value; returns None rather than raising
- Extracts MAXIMUM TODAY column via
re.search(r'MAXIMUM\s+([\d.]+)', text) — takes the first numeric
 token only, ignoring NORMAL/RECORD/YEAR columns
- httpx + tenacity (3 retries, exponential backoff) on network
errors; any error short-circuits to None

job_confirm_settlement() in scheduler/orchestrator.py — NEW JOB
- Runs once daily at 10:05 AM ET via CronTrigger
- Computes yesterday, calls fetch_official_daily_high(yesterday)
- If CLI value is available: upserts markets.final_official_high
 with NWS value, sets market_status="settled", then calls
run_full_calibration() so Brier scores and drift adjustments are
 computed against the authoritative settlement figure
- If CLI returns None (not posted yet, date mismatch, or MAXIMUM
 missing): logs a warning and exits with no DB change and no
calibration trigger — the ASOS preliminary value from
job_check_settlement() remains as the calibration fallback

startup_sequence() catch-up block (added to
scheduler/orchestrator.py)
- On startup, checks if yesterday's final_official_high is None
or still equals current_max_observed (i.e., the ASOS preliminary
 value was never replaced)
- If so, attempts one fetch_official_daily_high(yesterday) call
and writes the result to the DB — recovers the authoritative
value when the app was offline at 10:05 AM

Minor: job_check_settlement() logging update
- Added source="asos_preliminary" to the settlement recording
log line to make it unambiguous that the 7 PM value is a
preliminary ASOS estimate, not the NWS official figure

---
Decisions Made Outside the Original Spec

- Authoritative settlement source: Original spec computed
final_official_high as max(ASOS readings) only. Added NWS CLI as
 the authoritative override source — this is what Kalshi
actually uses to settle markets, so Brier scoring and drift
calibration now track what determines P&L, not a proxy.
- Two-phase settlement pattern: ASOS preliminary at 7 PM (kills
auto-trading, maintains hard floor) → NWS CLI confirmation at
10:05 AM next morning (updates calibration). The preliminary
phase is preserved because it's needed for the kill switch and
end-of-day cleanup regardless of whether the CLI ever arrives.
- Early scan termination: If a CLI version's report date is
older than target_date, scanning stops immediately instead of
continuing through all 5 versions. The NWS product is
newest-first so there's no value in going deeper.
- Startup catch-up equality check: Catch-up triggers if
final_official_high == current_max_observed as a proxy for
"still the ASOS preliminary value." This heuristic could
theoretically fire when the CLI value happens to match ASOS
exactly, but it's harmless — it just re-confirms the same value.

---
Known Issues / TODOs

- CLI not verified against live NWS page:
fetch_official_daily_high() has not been run against a real CLI
product yet. The regex patterns (CLIMATE SUMMARY FOR,
MAXIMUM\s+) are based on the standard NWS CLI format but should
be verified against an actual fetched page before the 10:05 AM
job fires in production.
- HTML vs plain text: The fetcher requests format=txt in query
params and sets Accept: text/html,text/plain. If NWS returns an
HTML-wrapped version, the regex will still work (the climate
data is embedded as-is in the HTML body) but the response will
be noisier. A future improvement could strip HTML tags before
parsing.
- No retry on version cycling: If version 1 returns a network
error, the fetcher returns None immediately rather than trying
version 2. This is intentional (network errors suggest the host
is down, not a version problem), but means a transient 1-second
timeout on version 1 will suppress the entire fetch. The startup
 catch-up on the next restart mitigates this.
- confirm_settlement depends on yesterday having a market row:
If the app was offline all of yesterday (no
job_check_settlement() ran, no market row exists),
job_confirm_settlement() will create a new settled row with no
current_max_observed. That's correct behavior but the hard floor
 and trade history for that day will be absent.
             

---                                                             
## Session Summary — Phase 2 Build (March 15, 2026 Pt 5)        

### What Was Built                                              

**Model Transparency Tab — Stages 2–5** (ui/app.py)

All four forward-looking pipeline stages were appended inside
`render_model_transparency()` as `st.expander()` blocks.

Stage 2 — NWP Forecast Snapshot:                                
- Per-model columns (HRRR / GFS / ECMWF) showing predicted high,
  blend weight, fetch time, and freshness badge (✓ Fresh / ⚠    
Stale)
  reusing `_staleness_color()` / `_colored_label()` helpers from
 Stage 1
- 24-hour Plotly chart: one dashed line per available model, one
  thick darkorange blended-average line, NOW vertical marker
- Bottom row: Morning Drift Adj., Afternoon Drift Adj., The
Attractor
  (μ_t = blended NWP at current UTC hour + Kalman bias + drift)
- Empty-state st.info() when no NWP data is in DB

Stage 3 — Monte Carlo Inputs:
- 11-row parameter table (hard floor, T₀, bias, theta, sigma, mu
 drift,
  hour offset, remaining day fraction, n_steps, n_paths, NWP
attractor)
- "▶ Run Simulation Now" button: constructs MCParams from
current DB
  state, fetches live Kalshi markets for strikes (floor + cap +
  extracted), calls price_full_distribution(), stores result in
  st.session_state["transparency_mc_result"] and
  st.session_state["transparency_mc_params"]
- Empty-state when system state is missing

Stage 4 — Simulated Distribution:
- Left: Plotly histogram (5,000-sample synthetic normal clipped
at
  hard floor), dashed vertical lines at each Kalshi strike,
solid
  red hard-floor line, health caption
- Right: Percentile table (p10–p90, mean, std dev) + per-strike
  cumulative probability table P(max ≥ strike) / P(max < strike)
- Gated behind transparency_mc_result session key

Stage 5 — Edge Calculation Breakdown:
- DRY_RUN warning banner when applicable
- Full edge table (Range, Fair Value, Kalshi Ask, Kalshi Bid,
  YES Edge, NO Edge, Kelly %, Contracts, Signal) built from live
  KalshiFetcher.get_temperature_markets() call
- Written-out st.code() Kelly calculation for the market with
the
  largest absolute YES edge: shows b, raw Kelly, 25% fractional
  Kelly, dollar bet, contract count, signal
- Falls back to st.info() when all markets have no resting
orders
- Gated behind transparency_mc_result session key

Session state keys added (prefixed transparency_ to avoid
collision
with Tab 1 edge table keys edge_table_rows / edge_table_diag):
- transparency_mc_result  → MonteCarloResult from Stage 3 button
- transparency_mc_params  → MCParams from Stage 3 button

---

### Deviations from the Original Plan

Phase sequencing abandoned:
- The original CLAUDE.md spec defined 7 discrete phases
(Config/DB,
  Fetchers, Kalman/MC, Calibrator, Trader, UI, Orchestrator) to
be
  built one at a time with explicit sign-off between phases.
- In practice all 7 phases were implemented across multiple
sessions
  without phase-by-phase pauses. The system is functionally
complete
  but was never validated phase-by-phase.

Scheduler architecture change (previously documented):
- Original spec: separate "Trading Engine" Replit workflow
- Actual: APScheduler runs as a background thread inside the
Streamlit
  process via _maybe_start_scheduler(). Works for Replit free
tier but
  stops when the browser tab closes.

Startup catch-up logic added (not in spec):
- Hard floor catch-up scans ASOS history on startup
- Missed calibration catch-up checks last_calibrated_utc and
fires
  immediately if midnight job was missed

Position tracking added (not in spec):
- evaluate_and_trade() fetches existing positions via
get_positions()
  and reduces Kelly contracts by current exposure

Two-phase settlement added (not in spec):
- Phase 1 (7 PM ET): ASOS preliminary via job_check_settlement()
- Phase 2 (10:05 AM ET next day): NWS CLI authoritative via
  job_confirm_settlement() in nws_cli_fetcher.py

GFS model corrected:
- Original spec said gfs_seamless; changed to gfs_global as
primary
  because gfs_seamless is a HRRR+GFS blend that produces
identical
  near-term results to HRRR, destroying model independence

kalshi_strike column migration:
- Original schema used SmallInteger for strike; live data has
decimal
  strikes (44.5°F). Migrated to NUMERIC(5,1) via idempotent
  _migrate_kalshi_strike_columns() on every startup

hour_offset UTC fix:
- Original trader.py and calibrator.py used Eastern hour as the
  nwp_curve index. Corrected to UTC hour to eliminate a ~5-hour
  systematic bias in all probability estimates

RSA-PSS padding (previously documented):
- Original code used PKCS1v15; Kalshi elections API requires
RSA-PSS
  with MGF1(SHA256) and DIGEST_LENGTH salt

NWS CLI as authoritative settlement source (not in spec):
- Spec computed final_official_high only from ASOS readings.
- Added NWS CLI fetch because that is what Kalshi actually uses
for
  settlement, making Brier scores track real P&L outcomes

Stage 6 (Historical Calibration Performance) deferred:
- The plan document explicitly deferred Stage 6 to Phase 3.
- It requires multi-day DB queries and a date-picker replay UI
that
  are independent of the pipeline stages built here. Not yet
built.

---

### Known Issues / TODOs

BLOCKING or HIGH PRIORITY:

1. No end-to-end validated cycle:
   The system has never been confirmed to complete a full
   fetch → Kalman update → MC → trade evaluation → snapshot
cycle
   from a cold start. All dashboard values may still show N/A.
   Trigger manually: Calibration tab → Fetch All NWP Models,
then
   observe scheduler logs for ASOS fetch completion.

2. NWS CLI regex not verified against live page:
   fetch_official_daily_high() patterns (CLIMATE SUMMARY FOR,
   MAXIMUM\s+) are based on the standard NWS CLI format but have
   not been run against a real fetched page. The 10:05 AM ET
   settlement confirmation job depends on this working
correctly.

3. KALSHI_ENV=demo is cosmetic:
   The setting is validated (demo/prod) but never used to select
   the API URL. URL always comes from kalshi_api_base_url
directly.
   Either wire it or remove it to avoid confusion.

4. Startup catch-up equality heuristic is fragile:
   The check `final_official_high == current_max_observed` used
to
   detect "still preliminary ASOS value" will false-positive on
days
   when the NWS CLI value exactly matches the ASOS reading.
Harmless
   (re-writes the same value) but conceptually wrong.

MEDIUM PRIORITY:

5. Blended forecast truncates to shortest model curve:
   If HRRR provides 18h and GFS provides 24h, the blend is cut
to
   18h in both the Visualizer and Stage 2. Should blend per-hour
 with
   whatever models have data at that hour.

6. Settlement depends on ASOS completeness:
   job_check_settlement() at 7 PM ET uses max(ASOS readings) as
the
   preliminary high. If ASOS fetch was offline during peak
hours, the
   preliminary value may be lower than the true peak.

7. Position tracking is additive-only:
   get_positions() reduces Kelly by long exposure but does not
account
   for short (NO) positions. Works for the current
single-direction
   strategy but needs updating if NO trades are ever executed.

8. Stage 4 histogram uses synthetic sample, not actual paths:
   The distribution histogram approximates using
   numpy.random.normal(mean_max, std_max, 5000). This is correct
 on
   average but does not reflect the true hard-floor truncation
shape
   from the actual simulation. A future improvement would pass
the
   actual paths_max array through to the UI (requires storing it
   alongside MonteCarloResult).

9. Stage 5 best-edge selection only considers YES-side:
   The written-out Kelly calculation block only shows detail for
 the
   market with the largest absolute YES edge where yes_ask > 0.
If
   the best signal is actually a BUY NO (via bid side), the
Kelly
   block shows a suboptimal or no market.

10. No retry on NWS CLI version cycling:
    If version 1 returns a network error, the fetcher returns
None
    immediately instead of trying version 2. A transient timeout
 on
    version 1 suppresses the entire fetch until the next
startup.

LOW PRIORITY / CLEANUP:

11. hour_et variable in trader.py and calibrator.py:
    Now only used for AM/PM drift selection after the UTC
hour_offset
    fix. Could be renamed to clarify it is only for drift, not
for
    nwp_curve indexing.

12. confirm_settlement requires yesterday's market row:
    If the app was offline all of yesterday,
job_confirm_settlement()
    creates a new settled row with no current_max_observed. The
hard
    floor and trade history for that day are absent.

13. Stage 6 (Historical Calibration Performance) not built:
    Requires multi-day DB queries, date-picker replay, Brier
score
    time series, and weight convergence chart. Deferred.

14. tests/ does not cover:
    - kalshi_fetcher.py (auth headers, market normalization,
position
      fetch) — all mocked HTTP would be needed
    - orchestrator.py (job scheduling, startup sequence)
    - ui/app.py (no Streamlit testing)
    - nws_cli_fetcher.py (HTML parsing against real or fixture
page)
    - db_manager.py (requires live PostgreSQL or extensive
mocking)

---

### What Was Likely Overlooked by Moving Too Fast

The following are subtle correctness and robustness concerns
that
would normally surface during phase-by-phase review but may have
been missed by building all phases concurrently:

1. MCParams.hour_offset semantics are ambiguous across callers:
   The field is UTC hour in trader.py (fixed) and
orchestrator.py
   but the original spec described it as "current hour-of-day
index
   into nwp_curve." The Visualizer tab uses day_start (a UTC
   midnight boundary) to index nwp_curve, which is consistent.
But
   if a future caller accidentally passes an Eastern hour, the
bias
   bug silently re-appears. There is no type or range validation
 on
   hour_offset in MCParams.

2. MCParams.day_fraction_remaining is auto-computed from wall
clock,
   not from hour_offset:
   If a caller passes hour_offset=0 for a future day but
   day_fraction_remaining is not passed explicitly, it
auto-computes
   from get_remaining_day_fraction() which reads the current
Eastern
   time. For a next-day trade this returns < 1.0, which is
wrong.
   The current code in trader.py works around this with explicit
   handling but the MCParams constructor does not enforce
consistency.

3. The Kalman filter predict step is coupled to NWP delta
magnitude:
   The predict step adds the hourly NWP delta to the temperature
   state and increases covariance by Q. If NWP forecasts are
missing
   (flat fallback), the predict step is called with delta=0 for
every
   hour, which means the filter never predicts forward — it only
   updates from ASOS observations. This is arguably correct but
   means the filter effectively becomes a simple exponential
smoother
   when NWP is absent, which was not documented as an intended
   degradation mode.

4. Hard floor atomicity is maintained by PostgreSQL GREATEST()
in
   db_manager but the Python-side update_hard_floor() reads the
   current value first, then writes with GREATEST():
   Between the read and write, another process could insert a
higher
   value that gets overwritten by an older value from the first
   process. The GREATEST() in the SQL WHERE/SET expression is
   atomic but only if called with a single UPDATE statement.
   Need to verify the actual SQL in
db_manager.update_hard_floor()
   uses a single UPDATE ... SET col = GREATEST(col, :val)
without
   a preceding SELECT.

5. Sigma and theta in SystemStateDocument start at settings
defaults
   and are calibrated from historical Brier scores. But Brier
score
   calibration requires at least N days of data (typically 30+).
   For the first days of operation the calibrated values are
   meaningless and the system silently uses the defaults. There
is no
   warm-start period logic, no UI warning, and the calibrator
does
   not indicate how many days of history it used.

6. Settlement detection at 7 PM ET uses calendar date from the
OS
   clock, not get_target_date():
   This was intentionally documented as correct behavior
(target_date
   has already rolled over to tomorrow by 7 PM ET). However if
the
   OS clock or timezone offset is misconfigured,
job_check_settlement
   could silently settle the wrong date's market row.

7. KalshiFetcher._normalize_market() converts yes_bid_dollars /
   yes_ask_dollars to cents (×100) but does not validate that
the
   input is in [0, 1]. If Kalshi changes their field format
again
   (e.g. returns integers in [0, 100] instead of floats in [0,
1]),
   the normalization silently multiplies already-correct cent
values
   by 100 and all edge calculations become wrong without any
error.

8. The Stage 3 simulation button in the Model Transparency tab
   constructs MCParams independently from the same logic in
   render_trading_desk(). This is now the third place this
   construction logic is written (trader.py is the first, Tab 1
   edge table is the second, Stage 3 is the third). If the
   hour_offset or drift_adj logic changes in one place, it will
   silently diverge in the others.

9. test_monte_carlo.py uses a deterministic sigma=0 case to
verify
   convergence but does not test the hard floor truncation shape
   (i.e. that the distribution is correctly right-skewed when
the
   hard floor is near the mean). The histogram in Stage 4
displays
   a synthetic normal which will not reveal this correctly.

10. The IEM fallback in asos_fetcher.py triggers on staleness
but
    the staleness threshold (asos_staleness_minutes) applies to
the
    NWS observation timestamp, not to the time the data was
stored
    in the DB. If there is a DB write delay, the fallback
threshold
    is effectively shorter than configured. This is a minor
issue
    but could cause unnecessary IEM calls during brief NWS
slowdowns.

---
A few meta-notes on why this list is long:

The core risk of full-stack-first development is that the
integration contracts between modules (what format, what
timezone, what units, what semantics) were established by
convention rather than by spec review. Each module works in
isolation and the tests pass, but the subtle invariants — UTC
vs. Eastern hour, GREATEST() atomicity, Brier-score cold-start
validity, normalize_market() assuming [0,1] inputs — are the
kind of thing that only surface when you use the system under
real market conditions. With phase-by-phase development you
would have caught these at the integration boundary before the
next layer was built on top of them.

The most important items to verify before trusting the system
with real money are #4 (hard floor atomicity SQL), #7
(normalize_market field format assumption), #2 (NWS CLI regex),
and #8 (MCParams construction duplication).


● What Has Been Built     March 15 2026, 4:37 PM ET                                  

  All 7 phases are functionally complete:                   

  Phase 1 — Config, Schemas, DB                             
  - settings.py with pydantic-settings, all               
  OU/Kalman/trading params                                  
  - Full PostgreSQL schema: markets, nwp_forecasts,       
  asos_readings, system_state, intraday_snapshots,
  trade_logs
  - db_manager.py with all read/write operations
  - _migrate_kalshi_strike_columns() runs idempotently on
  startup to fix kalshi_strike from SmallInteger →
  NUMERIC(5,1) (live strikes are floats like 44.5)

  Phase 2 — Fetchers
  - asos_fetcher.py: NWS 5-min ASOS + IEM staleness fallback
  - nwp_fetcher.py: Open-Meteo HRRR/GFS/ECMWF with
  _MODEL_FALLBACKS dict; gfs_global as primary GFS (not
  gfs_seamless)
  - kalshi_fetcher.py: RSA-PSS auth
  (MGF1/SHA256/DIGEST_LENGTH salt), _normalize_market() for
  dollar→cent conversion, 3-strategy market query,
  extract_strike_from_market(), get_strike_label(),
  get_positions()
  - nws_cli_fetcher.py (not in original spec): fetches NWS
  Climate Summary for authoritative daily high

  Phase 3 — Quant
  - kalman_filter.py: 2D state [temp, bias], Joseph form
  covariance, predict + update steps
  - monte_carlo.py: OU simulation, _interpolate_cdf(),
  compute_yes_prob() with half-integer rounding boundaries,
  compute_normalized_market_probs() with gap detection and
  normalization

  Phase 4 — Calibration
  - calibrator.py: Brier score calibration, drift
  adjustments, model weight updates, snapshot management

  Phase 5 — Execution
  - trader.py: kill switch → gather state → fetch markets →
  MC pricing → edge evaluation → Kelly sizing → order
  placement → trade log. Position exposure reduction via
  get_positions().

  Phase 6 — UI (ui/app.py)
  - Tab 1 Trading Desk: live metrics, multi-strike edge
  table (button-triggered, cached), kill switch
  - Tab 2 Visualizer: ASOS history + NWP curves (computed
  live from DB) + MC percentile band
  - Tab 3 Calibration: model weights, drift sliders, "Fetch
  All NWP Models" button, Kalshi API diagnostics
  - Tab 4 Model Transparency (Stages 1–5): full pipeline
  walkthrough — system state → NWP snapshot → MC inputs →
  simulated distribution → edge breakdown with written-out
  Kelly

  Phase 7 — Orchestrator
  - orchestrator.py: all scheduled jobs, startup catch-up
  sequence (hard floor, missed calibration, NWS CLI
  confirmation), two-phase settlement (job_check_settlement
  at 7 PM + job_confirm_settlement at 10:05 AM)
  - Runs as APScheduler background thread inside Streamlit
  (not a separate process)

  ---
  What's Left To Do

  1. Fix sigma inflation of paths_max — the primary open
  problem (see Known Issues)
  2. Stage 6 Historical Calibration Performance — multi-day
  Brier score time series, date-picker replay, weight
  convergence chart; deferred from original plan
  3. Verify NWS CLI regex against a live page —
  fetch_official_daily_high() has never been tested against
  a real fetched NWS product
  4. End-to-end cycle validation — the full fetch → Kalman
  update → MC → trade eval → snapshot loop has never been
  confirmed from a cold start with live data
  5. Test coverage gaps: kalshi_fetcher.py, orchestrator.py,
   ui/app.py, nws_cli_fetcher.py, db_manager.py all have
  zero test coverage

  ---
  Known Issues

  Blocking:

  - Sigma inflation: sigma=1.698°F/√hr (calibrated from
  5-minute ASOS diffs) systematically inflates paths_max
  ~2°F above the NWP attractor peak. Math: OU stationary std
   = sigma/sqrt(2*theta) ≈ 0.85°F; expected running-max
  overshoot over ~14 effective independent windows = 0.85 ×
  sqrt(2*ln(14)) ≈ 2.0°F. Observed: T0=37.4, NWP peak=39.1,
  MC mean_max=40.2. NWP ceiling cap approach was tried and
  reverted — made it worse. Needs a new approach
  (re-calibrating sigma against daily-max residuals, not
  5-min increments, is the most principled fix).

  High priority:

  - MCParams constructed in 3 independent places: trader.py,
   Tab 1 UI, Stage 3 UI each independently build MCParams.
  If hour_offset or drift_adj logic diverges in one,
  predictions silently differ.
  - _normalize_market() has no input guard: assumes
  yes_bid_dollars/yes_ask_dollars are in [0,1]. If Kalshi
  ever returns cents, all edge calculations silently inflate
   by 100×.
  - Hard floor atomicity: need to confirm
  db_manager.update_hard_floor() uses a single UPDATE ...
  SET col = GREATEST(col, :val) with no preceding SELECT.
  - KALSHI_ENV=demo is cosmetic: validated but never used to
   select the API URL; cosmetic confusion.

  Medium priority:

  - Blended forecast truncates to shortest model curve
  - Settlement preliminary high depends on ASOS completeness
   at 7 PM
  - Position tracking is additive-only (long exposure only,
  not short/NO)
  - Stage 4 histogram uses synthetic normal, not actual
  paths_max array
  - Sigma/theta cold-start: calibrated values are
  meaningless for first ~30 days; no UI warning
  - Startup catch-up equality check (final_official_high ==
  current_max_observed) will false-positive when they happen
   to match

  ---
  Architectural Deviations

  Original Spec: Separate "Trading Engine" Replit workflow
  What Was Built: APScheduler runs as background thread
    inside Streamlit; stops when tab closes
  ────────────────────────────────────────
  Original Spec: Phase-by-phase build with sign-off gates
  What Was Built: All 7 phases built concurrently across
    sessions
  ────────────────────────────────────────
  Original Spec: gfs_seamless as GFS source
  What Was Built: gfs_global (pure GFS); seamless was
    producing identical near-term results to HRRR
  ────────────────────────────────────────
  Original Spec: final_official_high from ASOS max only
  What Was Built: Two-phase: ASOS preliminary at 7 PM → NWS
    CLI authoritative at 10:05 AM next day
  ────────────────────────────────────────
  Original Spec: SmallInteger for kalshi_strike
  What Was Built: NUMERIC(5,1) — live strikes are decimals
    (44.5°F)
  ────────────────────────────────────────
  Original Spec: Eastern hour as hour_offset into nwp_curve
  What Was Built: UTC hour — Eastern caused a systematic
    ~5-hour bias
  ────────────────────────────────────────
  Original Spec: Integer CDF boundaries for bucket pricing
  What Was Built: Half-integer boundaries (e.g., 39.5°F) —
    NWS rounds to nearest integer so continuous settlement
    boundary is at the half-integer
  ────────────────────────────────────────
  Original Spec: compute_yes_prob() with .get(key, default)
  What Was Built: _interpolate_cdf() +
    compute_normalized_market_probs() with partition gap
    detection and normalization
  ────────────────────────────────────────
  Original Spec: No position tracking
  What Was Built: get_positions() reduces Kelly contracts by

    current exposure
  ────────────────────────────────────────
  Original Spec: No startup recovery logic
  What Was Built: Hard floor catch-up + missed calibration
    catch-up + NWS CLI confirmation on restart
  ────────────────────────────────────────
  Original Spec: RSA PKCS1v15 signing
  What Was Built: RSA-PSS with MGF1(SHA256) + DIGEST_LENGTH
    salt — what Kalshi elections API actually requires
  ────────────────────────────────────────
  Original Spec: Stage 6 in original UI plan
  What Was Built: Deferred — not yet built

  ● What Was Built / Fixed (Since Pt 5)    March 15 2026 5:18 Pm ET                   

    MC Simulation — NWP Anchor Fix (quant/monte_carlo.py)     

    Root cause resolved: The sigma inflation bug (~2°F        
    overshoot on paths_max) was caused by the OU attractor  
    being set to the raw NWP hourly value at step 0, while T0 
    (the current observation) was below it. With kalman_B = 
    0.0 (cold-start Kalman), every path was immediately pulled
     upward from T0 toward the NWP level before the NWP curve
    began declining, inflating paths_max before any real
    temperature rise had occurred.

    Fix: One-time nwp_anchor_offset computed before the
    simulation loop:
    nwp_anchor_offset = T0 - nwp_curve[hour_offset]
    Applied inside the loop:
    mu_t = nwp_curve[hour_idx] + nwp_anchor_offset + bias +
    drift_adj
    Step-0 attractor now equals T0 exactly. Subsequent steps
    follow the NWP's rate of change rather than its absolute
    level. As kalman_B converges over weeks of operation, bias
     absorbs the systematic NWP error and nwp_anchor_offset
    trends toward zero naturally — the fix self-eliminates as
    the Kalman warms up.

    ---
    Tests (tests/test_monte_carlo.py)

    Three test changes:

    1. test_zero_sigma_converges_to_nwp updated — changed
    T0=60.0 → 75.0 to match the flat NWP curve, making
    nwp_anchor_offset = 0. Without this, the test would have
    incorrectly expected paths to converge to 75°F when the
    anchored attractor is actually 60°F. Tightened tolerance
    from abs=5.0 → abs=0.1 given sigma=0.
    2. TestNWPAnchor class added — two deterministic (sigma=0)
     tests that directly validate the fix:
      - Declining NWP: T0=37.4, NWP[0]=38.7, NWP declining →
    paths_max should stay near 37.4, not 38.7
      - Rising NWP: T0=34, NWP rises from 35 to 40 → paths_max
     should peak near T0 + max_NWP_delta = 39, not at the raw
    NWP peak of 40
    3. Pre-existing test bug fixed — test_sigma_positive was
    already broken before this session: FakeReading was
    missing observation_time_utc, causing AttributeError in
    estimate_sigma_from_historical. Added proper timestamps
    spaced 5 minutes apart.

    All 21 tests pass.

    ---
    UI Auto-Refresh (ui/app.py)

    Changed time.sleep(60) → time.sleep(300) at the bottom of
    main(). The 60-second refresh was purely cosmetic overhead
     — ASOS and trade eval run every 5 minutes, NWP every 60
    minutes. Nothing meaningful changes in under 5 minutes.

    ---
    Clarification on Snapshot Frequency

    The "snapshot history every minute" observation was a
    misread of the 60-second page refresh re-rendering the
    same rows. record_snapshot is only called by
    job_take_snapshot (scheduled every 2 hours) and the manual
     "📸 Take Snapshot Now" button — never on the render path.
     No code change was needed. The 2-hour frequency is
    correct.

    ---
    Deviations from Original Spec

    None new in this session. All prior deviations are
    documented in Pt 5.

    ---
    Remaining Risks and Open Items

    HIGH — Still needs verification before trusting with real
    money:

    1. NWP anchor fix changes the model's systematic behavior
    for the first time. The fix hasn't been validated against
    a live trading day with real ASOS data flowing. Until
    confirmed with actual observations, it's possible the
    anchor offset is slightly overcorrecting on
    rising-temperature mornings where T0 is genuinely below
    the NWP curve for good reason (e.g., fog burn-off, sea
    breeze expected). Monitor mean_max vs actual final high
    over the first few days.
    2. _normalize_market() has no input guard — assumes
    yes_bid_dollars/yes_ask_dollars are in [0,1]. If Kalshi
    changes their field format, all edge calculations silently
     inflate by 100× with no error.
    3. Hard floor atomicity — update_hard_floor() needs a
    single-statement UPDATE ... SET col = GREATEST(col, :val)
    with no preceding SELECT. This hasn't been confirmed in
    db_manager.py.
    4. NWS CLI regex unverified — fetch_official_daily_high()
    patterns (CLIMATE SUMMARY FOR, MAXIMUM\s+) have never been
     run against a real fetched NWS page. The 10:05 AM
    settlement confirmation job depends on this working.
    5. No end-to-end validated cycle — a full fetch → Kalman
    update → MC → trade eval → snapshot loop has never been
    confirmed from a cold start with live data flowing.

    MEDIUM — Correctness concerns:

    6. MCParams constructed in 3 independent places —
    trader.py, Tab 1 edge table, and Stage 3 Model
    Transparency each independently build MCParams. The
    hour_offset/drift_adj logic can silently diverge if one
    copy is updated.
    7. Stage 4 histogram uses synthetic normal — the
    distribution visualization approximates with
    numpy.random.normal(mean_max, std_max, 5000) rather than
    the actual paths_max array. Doesn't reflect the true
    hard-floor truncation shape.
    8. Blended forecast truncates to shortest model curve — if
     HRRR provides 18h and GFS 24h, the blend is cut to 18h.
    Should blend per-hour with whatever models have data.
    9. Kalman cold-start: calibrated sigma/theta are
    meaningless for first ~30 days. No UI warning, no
    warm-start period logic. Calibrated values are computed
    from insufficient Brier score history and silently used
    anyway.
    10. MCParams.day_fraction_remaining auto-computes from
    wall clock, not from hour_offset. For any hypothetical
    future-day pricing, this would silently be wrong.

    LOW — Cleanup:

    11. KALSHI_ENV=demo is cosmetic — the setting is validated
     but never used to select the API URL.
    12. Stage 6 (Historical Calibration Performance) —
    multi-day Brier score time series, date-picker replay,
    weight convergence chart — not yet built.
    13. Test coverage gaps remain: kalshi_fetcher.py,
    orchestrator.py, ui/app.py, nws_cli_fetcher.py,
    db_manager.py all have zero tests.

      What We Did This Session         March 15 2026 11:18 PM                         

      1. Model Transparency Tab — Math Explainability           
      (ui/app.py)                                             

      Stage 1 — Kalman Filter State                           
      - Added full 2×2 covariance matrix display (all four P
      elements with labels)
      - Computed and displayed Kalman gains K_T and K_B inline
      from P, with cold-start expectations in tooltips; K_B
      shows "⚠ frozen" when near zero
      - Added Last Innovation metric (ASOS − Kalman) with ±2°F
      warning
      - Added static update/predict rule code block explaining
      filter mechanics
      - Renamed "Temp Variance" → "Temp Variance P[0,0]"

      Stage 2 — NWP Forecast Snapshot
      - Fixed the attractor formula: was NWP + bias + drift, now
       correctly NWP + anchor_offset + bias + drift
      - Replaced 3-column row with 4-column row: NWP Blended
      Now, NWP Anchor Offset, Kalman Bias, Corrected Attractor
      (μ₀)
      - Added caption showing the full arithmetic
      - Added Drift Calibration Provenance sub-section showing N
       settled days, snapshot counts, mean errors, and resulting
       adjustments

      Stage 2.5 — Calibration Audit (new expander)
      - Full audit table: sigma, theta, OU stationary σ,
      morning/afternoon drift, model weights — each with formula
       and source
      - Staleness warning if last_calibrated_utc is >36 hours
      ago
      - Note when weights are still at defaults

      Stage 3 — Monte Carlo Inputs
      - Added NWP Anchor Offset row (T₀ − NWP[h₀])
      - Added OU Stationary σ row (σ/√2θ)
      - Fixed attractor row to include anchor offset and drift;
      renamed to "Step-0 Attractor (μ₀)"
      - Added caption explaining anchor construction

      2. Drift Calibration — 7-Day Rolling Window
      (calibrator.py)

      - calibrate_intraday_drift now loops over the past 7 days
      instead of hardcoding yesterday
      - Errors from all settled days are pooled before taking
      the mean
      - Falls back to 0.0 if fewer than 2 settled days exist
      - Logs days_used in the output
      - UI provenance section and Stage 2.5 audit text updated
      to match

      ---
      Deviations from the Plan

      Attractor bug fix was in the plan; scope was slightly
      broader than specified. The plan said to add the anchor
      offset to Stage 2 and Stage 3. We also fixed the drift
      formula in Stage 3's attractor_s3 (it was missing
      drift_adj entirely, not just anchor offset) — this wasn't
      called out explicitly but was clearly wrong.

      7-day rolling drift was not in the original plan at all.
      The plan document described drift as "calibrated from
      yesterday only" and the Stage 2.5 audit table was written
      to reflect that. We changed both the underlying logic and
      the UI description mid-session. This is a meaningful
      behavioural change to the trading system, not just a
      display fix.

      Stage 2.5 drift provenance display deviates from what the
      plan specified. The plan called for querying
      get_snapshots_for_date(yesterday) and showing a single-day
       breakdown. We replaced that with a 7-day loop to match
      the new calibrator logic.

      ---
      What's Left To Do (from prior session notes)

      Blocking / High Priority:
      1. No end-to-end validated cycle — full fetch → Kalman
      update → MC → trade eval → snapshot has never been
      confirmed from a cold start with live data
      2. NWS CLI regex unverified — fetch_official_daily_high()
      patterns have never been run against a real fetched NWS
      page; the 10:05 AM settlement job depends on it
      3. _normalize_market() has no input guard — if Kalshi
      returns cents instead of [0,1] floats, all edge
      calculations silently inflate 100×
      4. Hard floor atomicity unconfirmed — need to verify
      update_hard_floor() uses a single UPDATE ... SET col =
      GREATEST(col, :val) with no preceding SELECT
      5. KALSHI_ENV=demo is cosmetic — validated but never used
      to select the API URL

      Medium Priority:
      6. MCParams constructed in 3 independent places —
      trader.py, Tab 1 edge table, Stage 3 — can silently
      diverge
      7. Stage 4 histogram uses synthetic normal, not actual
      paths_max array
      8. Blended forecast truncates to shortest model curve
      9. Kalman cold-start: no UI warning that sigma/theta are
      meaningless for first ~30 days
      10. Stage 6 (Historical Calibration Performance) —
      multi-day Brier score time series, date-picker replay —
      not built

      Low Priority / Cleanup:
      11. CLAUDE.md phase checkboxes are all unchecked despite
      the system being fully built
      12. Test coverage gaps: kalshi_fetcher, orchestrator,
      app.py, nws_cli_fetcher, db_manager all have zero tests