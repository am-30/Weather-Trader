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