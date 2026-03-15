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