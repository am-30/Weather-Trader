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

  What Was Built / Fixed Today, March 15 1:00 AM EDT                                    

  Infrastructure fixes:                                           
  - Added PYTHONPATH=/home/runner/workspace to the Streamlit
  command in .replit — fixes ModuleNotFoundError: No module named
  'kalshi_weather_trader' that occurred because Streamlit adds the
   script's own directory to sys.path, not the workspace root
  - Reverted unnecessary proxy code (http-proxy-middleware,
  WebSocket upgrade handler) that had been added to
  artifacts/api-server/src/app.ts and index.ts during earlier
  debugging

  Scheduler architecture change (not in original spec):
  - Original spec had the scheduler running as a separate "Trading
   Engine" workflow in Replit
  - Changed to run APScheduler as a background thread inside the
  Streamlit process (_maybe_start_scheduler() in ui/app.py)
  - Reason: Replit free tier doesn't support persistent background
   workflows; this approach runs exactly when the app is open and
  stops when it closes
  - The orchestrator.py and build_scheduler() are unchanged and
  still work as a standalone process if needed later

  Kalshi ticker corrected:
  - Original code used KXHIGHNEW and HIGHBOS as event ticker
  prefixes — both wrong
  - Correct series ticker is KXHIGHTBOS (e.g. KXHIGHTBOS-26MAR15)
  - Date format %y%b%d (e.g. 26MAR15) was confirmed correct from
  the actual market URL

  Kalshi API domain migration:
  - Kalshi has migrated their API from
  https://trading-api.kalshi.com/trade-api/v2 to
  https://api.elections.kalshi.com/trade-api/v2
  - Default URL updated in config/settings.py
  - User added KALSHI_API_BASE_URL=https://api.elections.kals
  hi.com/trade-api/v2 in Replit Secrets to override any cached
  value, but still not working

  Diagnostic tool added:
  - "Test Kalshi Connection" button added to the Calibration tab
  in ui/app.py
  - Tests key loading, balance endpoint, and market search with
  raw response output
  - Useful for debugging auth issues without checking logs

  ---
  Known Issues / TODOs

  BLOCKING — Kalshi authentication returning HTTP 401:
  - Despite correct URL, correct ticker, and RSA key loading
  successfully, all API calls return 401
  - Two signing formats tested: with /trade-api/v2 path prefix and
   without — both fail
  - KALSHI_API_BASE_URL secret may not be taking effect
  (diagnostic still showed old URL on last run — user needs to
  confirm secret is set and workflow fully restarted)
  - Root cause still unconfirmed: could be wrong signing message
  format for the new api.elections.kalshi.com endpoint, or secret
  not applied
  - Next step: Confirm the diagnostic shows the new URL after
  setting the secret; if still 401, check Kalshi's migration docs
  at api.elections.kalshi.com for any auth format changes

  Balance endpoint type error:
  - get_balance() throws '<' not supported between instances of
  'str' and 'int'
  - Likely the response JSON has "balance" as a string instead of
  int, or the response structure changed with the new API domain
  - Fix: inspect the raw balance response once auth is working;
  may need int(data.get("balance", 0)) instead of relying on the
  API returning an int

  KALSHI_ENV is a no-op label:
  - kalshi_env setting in settings.py is validated (demo/prod) but
   never used to select the API URL
  - The URL is always taken from kalshi_api_base_url directly
  - Either wire kalshi_env to automatically set the URL, or remove
   it to avoid confusion

  No market data flowing yet:
  - System has not successfully completed a full
  fetch-update-snapshot cycle
  - All dashboard values show N/A

 What Was Built / Fixed Today, March 15 2026 Pt 2

  Kalshi Authentication — Fully Resolved

  RSA padding scheme wrong (root cause of all 401s):
  - Original code used padding.PKCS1v15(). Kalshi's elections API
  requires RSA-PSS with MGF1(SHA256) and DIGEST_LENGTH salt.
  - Fix: updated _get_auth_headers() in kalshi_fetcher.py

  Signing path confirmed: /trade-api/v2/portfolio/balance (with
  base path prefix) is correct. Without-prefix fails. Both were
  tested via diagnostic.

  Market status field: Kalshi uses status=active, not status=open.
   Changed get_temperature_markets() to use active. The
  status=open filter silently returned empty results, causing the
  system to believe no markets existed.

  Ticker format confirmed from live API:
  - Format: KXHIGHTBOS-26MAR15-T38, KXHIGHTBOS-26MAR15-B44.5
  - Strikes are floats, not ints (B44.5, B38.5)
  - Rewrote extract_strike_from_ticker() with regex
  -[TB](\d+(?:\.\d+)?)$; return type changed from int to float

  get_temperature_markets() simplified: removed fallback to bare
  KXHIGHTBOS series ticker search. Only the date-specific event
  ticker (e.g. KXHIGHTBOS-26MAR15) is searched, which is the
  correct Kalshi API pattern.

  ---
  before_sleep_log TypeError — Fixed Across All Fetchers

  All three fetchers (kalshi_fetcher.py, nwp_fetcher.py,
  asos_fetcher.py) had before_sleep_log(logger, "warning") which
  passes a string where tenacity expects an integer log level.
  This caused a '<' not supported between instances of 'str' and
  'int' TypeError whenever a retry was triggered. Removed
  before_sleep_log from all retry decorators.

  Also in kalshi_fetcher.py: changed retry_if_exception_type to
  retry_if_exception(_is_retryable) so 4xx HTTP errors (including
  401) fail immediately without retrying. Only 5xx and network
  errors retry.

  ---
  yes_bid / yes_ask Null Safety

  Live markets currently show yes_bid=None, yes_ask=None (no
  resting orders). Changed all reads from .get("yes_bid", 0) to
  .get("yes_bid") or 0 in trader.py and calibrator.py. Markets
  with no liquidity are skipped for trading; no crash.

  ---
  GFS Fallback Model Names (nwp_fetcher.py)

  Added _MODEL_FALLBACKS dict — if gfs_seamless fails, tries
  gfs_global; if ecmwf_ifs025 fails, tries ecmwf_ifs04.
  _fetch_model() loops through candidates, logs warning per
  failure, returns partial data (< 24 hrs) instead of None.

  get_nwp_curve() changed from min() to max() for curve length,
  with per-hour model filtering so a shorter model's data doesn't
  truncate a longer one.

  A "🌤️  Fetch All NWP Models" button was added to the Calibration
  tab for manual triggering.

  ---
  Calibration Tab Diagnostic Improvements

  - Diagnostic now tests both signing path formats for balance
  side-by-side
  - HTTP 401 responses now show Kalshi's full error body in the UI
   (was previously swallowed)
  - Direct event lookup (GET /events/KXHIGHTBOS-26MAR15),
  markets-without-status-filter, and events-by-series searches
  added to expose real API state

  ---
  Tests Updated

  test_ingestion.py strike extraction tests updated from old
  fictional ticker formats (KXHIGHNEW-2025-0615T70) to the
  confirmed live format (KXHIGHTBOS-26MAR15-T38,
  KXHIGHTBOS-26MAR15-B44.5).

  ---
  Known Issues / TODOs

  GFS line not appearing in Visualizer:
  The NWP curve changes require GFS data to be in the database.
  The Visualizer reads from DB, not live API. Steps to debug:
  1. Go to Calibration tab → click "🌤️  Fetch All NWP Models" —
  this will show exactly which models succeeded and how many hours
   of data each returned
  2. If GFS still fails (both gfs_seamless and gfs_global), paste
  the error — Open-Meteo may have renamed the GFS model identifier
   again
  3. If GFS succeeds in the fetch but still doesn't appear in the
  chart, the bug is in how the Visualizer reads NWP data from the
  DB, not in the fetcher

  Markets have no liquidity (yes_bid=None, yes_ask=None):
  6 active markets exist for today (T38, T45, B38.5, B40.5, B42.5,
   B44.5) but none have resting orders. The system will correctly
  skip them for execution until a market maker posts orders. This
  may be normal for early-morning hours or thin markets.

  KALSHI_ENV=demo label:
  Config shows env: demo but this is a cosmetic no-op — the URL
  (api.elections.kalshi.com) is production. Set KALSHI_ENV=prod in
   Replit Secrets to remove confusion. Does not affect any
  behavior.

  No full fetch-update-snapshot cycle completed yet:
  All dashboard values are likely still N/A. Auth is now working;
  the next step is triggering the scheduler (or manually fetching
  NWP + ASOS) to populate the DB so the dashboard shows live data.

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