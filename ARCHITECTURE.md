This is a serious build. I'm going to rewrite this from scratch as a true senior-level system specification — not just a synthesis of the two responses, but a hardened, production-quality prompt document that closes the gaps, specifies the mathematics precisely, handles edge cases, and gives the AI zero room to improvise incorrectly. The two original responses were good conceptually but left too many critical implementation details unspecified, which is exactly where AI coding tools hallucinate or cut corners.

---

# MASTER SYSTEM ARCHITECTURE & BUILD SPECIFICATION
## Kalshi Boston Temperature Trading System — v3.0
### For AI Coding Agent: Full Implementation Brief

---

## PREAMBLE FOR AI AGENT — READ BEFORE WRITING ANY CODE

You are a senior quantitative developer and software engineer. You are building a professional-grade, automated weather-trading system targeting the Kalshi Daily Maximum Temperature market for Boston Logan Airport (KBOS). This is not a toy project. Every module must be production-quality: type-annotated, logged, error-handled, and independently testable.

**Non-negotiable development rules:**
- Every file must have a module-level docstring explaining its purpose
- Every function must have a full docstring with Args, Returns, and Raises sections
- Every external API call must be wrapped in try/except with exponential backoff retry logic (max 3 retries)
- Every database write must be wrapped in try/except and logged
- Use `structlog` for all logging — never use print statements
- Use `python-dotenv` for all secrets — never hardcode credentials
- Use Pydantic v2 for all data models/validation
- All timestamps must be stored as UTC and converted to US/Eastern only at display time
- All temperatures are stored internally as Fahrenheit floats with one decimal place of precision
- Code must be compatible with Python 3.11+

**You will be fed this document in phases. When you receive a phase, write only the code for that phase. Do not anticipate future phases. Do not combine phases. Acknowledge the architecture and ask for confirmation before writing.**

---

## SECTION 0: FULL PROJECT STRUCTURE

Before writing any code, create the following directory structure exactly. Every file listed here will be populated in subsequent phases.

```
kalshi_weather_trader/
├── .env                          # Secrets (never committed to git)
├── .env.example                  # Template for secrets
├── .gitignore
├── requirements.txt
├── README.md
│
├── config/
│   ├── __init__.py
│   └── settings.py               # All constants, config, and target date logic
│
├── db/
│   ├── __init__.py
│   ├── db_manager.py             # Firebase Admin init and generic CRUD helpers
│   └── schemas.py                # All Pydantic models for every Firestore collection
│
├── ingestion/
│   ├── __init__.py
│   ├── asos_fetcher.py           # KBOS 5-minute ASOS data from NWS + IEM backup
│   ├── nwp_fetcher.py            # NWP model forecasts from Open-Meteo
│   └── kalshi_fetcher.py         # Kalshi market data and bid/ask polling
│
├── quant/
│   ├── __init__.py
│   ├── kalman_filter.py          # Cascaded 2D Kalman Filter implementation
│   └── monte_carlo.py            # Receding Horizon Monte Carlo simulation engine
│
├── execution/
│   ├── __init__.py
│   └── trader.py                 # Kalshi order execution logic and kill switch
│
├── calibration/
│   ├── __init__.py
│   └── calibrator.py             # Intraday snapshot calibrator and weight updater
│
├── scheduler/
│   ├── __init__.py
│   └── orchestrator.py           # APScheduler jobs wiring all components together
│
├── ui/
│   ├── __init__.py
│   └── app.py                    # Streamlit Command Center
│
└── tests/
    ├── __init__.py
    ├── test_kalman.py
    ├── test_monte_carlo.py
    └── test_ingestion.py
```

---

## SECTION 1: ENVIRONMENT SETUP

### requirements.txt — specify these exact packages:

```
firebase-admin==6.5.0
pydantic==2.7.0
python-dotenv==1.0.1
requests==2.31.0
numpy==1.26.4
scipy==1.13.0
pandas==2.2.2
streamlit==1.35.0
plotly==5.22.0
apscheduler==3.10.4
structlog==24.1.0
tenacity==8.3.0
pytz==2024.1
httpx==0.27.0
```

### .env.example — define these variables:

```
# Firebase
FIREBASE_CREDENTIALS_PATH=./firebase_service_account.json
FIRESTORE_PROJECT_ID=your_project_id

# Kalshi API
KALSHI_API_KEY=your_api_key_here
KALSHI_API_SECRET=your_api_secret_here
KALSHI_BASE_URL=https://trading-api.kalshi.com/trade-api/v2
KALSHI_MARKET_TICKER=KXHIGHTBOS-26MAR14   # Update this daily or make dynamic

# System
LOG_LEVEL=INFO
DRY_RUN=true   # Set to false only when live trading is desired
EDGE_THRESHOLD=0.05
MAX_POSITION_SIZE_DOLLARS=50
SIMULATION_PATHS=10000
```

---

## PHASE 1: CONFIG & DATABASE SCHEMA

**Agent Task:** Build `config/settings.py` and `db/schemas.py` and `db/db_manager.py`.

---

### FILE: `config/settings.py`

This file is the single source of truth for all system-wide constants, configuration, and the critical target date logic.

**Implement the following:**

**Constants:**
```python
STATION_ID = "KBOS"
STATION_LAT = 42.3606
STATION_LON = -71.0097
TIMEZONE = "America/New_York"
ROLLOVER_HOUR = 18  # 6 PM Eastern — after this, shift target to tomorrow
NWS_BASE_URL = "https://api.weather.gov"
IEM_BASE_URL = "https://mesonet.agron.iastate.edu/json"
OPEN_METEO_BASE_URL = "https://api.open-meteo.com/v1"
KALSHI_MARKET_PREFIX = "KXHIGHTBOS"
```

**Function: `get_target_date() -> str`**
- Import `datetime`, `pytz`
- Get current time in `America/New_York`
- If current time hour >= 18 (ROLLOVER_HOUR), return tomorrow's date as `YYYY-MM-DD`
- If current time hour < 18, return today's date as `YYYY-MM-DD`
- This function must be called everywhere a target date is needed — it must never be hardcoded elsewhere

**Function: `get_trading_day_bounds(target_date: str) -> tuple[datetime, datetime]`**
- Returns the start and end of the NWS climatological observation window for the given date
- The official NWS daily max for Boston is computed over midnight-to-midnight **local standard time** (i.e., always EST, not EDT)
- Return: `(start_utc, end_utc)` — the UTC equivalents of midnight-to-midnight EST on the target date
- Note: Since Boston observes DST, midnight EST = 05:00 UTC in winter and 04:00 UTC during DST — account for this explicitly using pytz `localize()` not manual offset

**Function: `get_remaining_day_fraction() -> float`**
- Returns the fraction of the trading day (0.0 to 1.0) remaining as of now
- Used by the Monte Carlo to know how many time steps to simulate
- If before market open (6 AM Eastern), return 1.0
- If after 6 PM, return 0.0
- Otherwise, compute `(18:00 - now) / (18:00 - 06:00)`

---

### FILE: `db/schemas.py`

Define all Pydantic v2 models. These are the canonical data structures for everything stored in Firestore and passed between modules.

**Model: `MarketDocument`**
```python
market_id: str              # Firestore doc ID, e.g. "2026-03-14"
target_date: str            # YYYY-MM-DD
current_max_observed: float # Hard floor — highest confirmed ASOS reading today. Default: -999.0
market_status: str          # "open" | "closed" | "settled"
auto_trade_enabled: bool    # Kill switch flag. Default: True
final_official_high: Optional[float]  # Populated after NWS CLI drops next morning. Default: None
last_updated_utc: datetime
```

**Model: `NWPForecastDocument`**
```python
doc_id: str                        # "{date}_{model_name}_{fetched_at_hour}" e.g. "20260314_HRRR_10"
target_date: str
model_name: str                    # "HRRR" | "GFS" | "ECMWF"
fetched_at_utc: datetime           # When this forecast was pulled
forecast_valid_from_utc: datetime
hourly_temps: List[float]          # Indexed 0-23, representing hourly temps for the target date in °F
predicted_daily_high: float        # max(hourly_temps)
```

**Model: `ASOSReadingDocument`**
```python
doc_id: str              # "KBOS_{timestamp_utc_iso}"
station_id: str          # "KBOS"
observation_time_utc: datetime
temperature_f: float
dew_point_f: Optional[float]
wind_speed_mph: Optional[float]
raw_metar: Optional[str] # Store the raw METAR string for auditability
```

**Model: `SystemStateDocument`**
```python
doc_id: str              # "KBOS_{target_date}" e.g. "KBOS_2026-03-14"
target_date: str
# Kalman Filter State
kalman_temp_estimate: float       # Current best estimate of true temperature
kalman_bias_estimate: float       # Current estimated model bias (NWP - ASOS)
kalman_covariance: List[List[float]]  # 2x2 covariance matrix as nested list
# Model Weights (must sum to 1.0)
model_weights: Dict[str, float]   # e.g. {"HRRR": 0.5, "GFS": 0.3, "ECMWF": 0.2}
# Drift Parameters
mu_drift: float                   # Morning bias correction (learned from calibration)
theta_decay: float                # Mean-reversion speed for OU process
sigma_volatility: float           # Intraday temperature volatility estimate
# Calibration metadata
last_calibrated_utc: Optional[datetime]
morning_drift_adjustment: float   # Specific correction for 8AM-12PM window
afternoon_drift_adjustment: float # Specific correction for 12PM-5PM window
last_updated_utc: datetime
```

**Model: `IntradaySnapshotDocument`**
```python
snapshot_id: str           # "KBOS_20260314_1400" (KBOS_YYYYMMDD_HHMM Eastern)
target_date: str
snapshot_time_eastern: str # "14:00"
snapshot_time_utc: datetime
current_asos_temp_f: float
current_max_observed_f: float
# NWP state at snapshot time
hrrr_predicted_high: Optional[float]
gfs_predicted_high: Optional[float]
ecmwf_predicted_high: Optional[float]
blended_predicted_high: float      # Weighted blend using model_weights
# Kalman state at snapshot time
kalman_temp_estimate: float
kalman_bias_estimate: float
# Market state at snapshot time
kalshi_implied_prob_yes: Optional[float]   # Decimal: 0.0 to 1.0
kalshi_bid: Optional[float]
kalshi_ask: Optional[float]
kalshi_strike: Optional[int]
# Our model output at snapshot time
model_fair_value_prob: Optional[float]
model_edge: Optional[float]       # fair_value - ask (positive = buy signal)
is_forced: bool                   # True if triggered by "Force Snapshot" button
```

**Model: `TradeLogDocument`**
```python
trade_id: str              # UUID4
target_date: str
executed_at_utc: datetime
market_ticker: str
action: str                # "BUY_YES" | "BUY_NO" | "SELL_YES"
kalshi_strike: int
contracts: int
price_cents: int           # Kalshi prices in cents (0-100)
fair_value_prob: float
kalshi_implied_prob: float
edge_at_execution: float
dry_run: bool              # Was this a simulated trade or real?
order_id: Optional[str]    # Kalshi order ID if real
status: str                # "filled" | "cancelled" | "pending" | "error"
notes: Optional[str]
```

**Model: `KalshiMarketSnapshot`**  *(Not stored in DB — used internally for passing data between modules)*
```python
ticker: str
market_title: str
yes_bid: float    # Decimal probability
yes_ask: float
no_bid: float
no_ask: float
volume: int
open_interest: int
status: str
```

---

### FILE: `db/db_manager.py`

**Initialize Firebase Admin SDK:**
- Load credentials path from `.env` via `python-dotenv`
- Initialize with `firebase_admin.initialize_app()`
- Create a singleton `get_db()` function that returns the Firestore client — use a module-level variable to avoid re-initializing on every call
- Handle `ValueError` if already initialized (idempotent init)

**Implement these generic helper functions. Every function must log its operation:**

`write_document(collection: str, doc_id: str, data: dict, merge: bool = True) -> bool`
- Uses Firestore `.set(..., merge=merge)` 
- Wrapped in try/except, returns True on success, False on failure
- Log: `"Writing to Firestore" collection=collection doc_id=doc_id`

`read_document(collection: str, doc_id: str) -> Optional[dict]`
- Returns the document dict or None if not found
- Do NOT raise on missing doc — return None

`query_collection(collection: str, filters: List[tuple]) -> List[dict]`
- Accepts filters as list of `(field, operator, value)` tuples
- Example: `[("target_date", "==", "2026-03-14"), ("station_id", "==", "KBOS")]`
- Returns list of dicts

`update_field_if_greater(collection: str, doc_id: str, field: str, new_value: float) -> bool`
- Uses a Firestore transaction to atomically update `field` only if `new_value > current value`
- This is the **Hard Floor update** for `current_max_observed` — must be atomic to avoid race conditions
- Critical: if the document doesn't exist yet, create it with the new value

`batch_write(collection: str, documents: List[tuple[str, dict]]) -> bool`
- Takes list of `(doc_id, data)` tuples and commits as a single Firestore batch

---

## PHASE 2: DATA INGESTION

**Agent Task:** Build `ingestion/asos_fetcher.py`, `ingestion/nwp_fetcher.py`, and `ingestion/kalshi_fetcher.py`.

Import and use the schemas from `db/schemas.py` and helpers from `db/db_manager.py`. Import `get_target_date()` from `config/settings.py`.

---

### FILE: `ingestion/asos_fetcher.py`

**Class: `ASOSFetcher`**

This class fetches real-time KBOS observations. It uses the NWS API as the primary source and Iowa Environmental Mesonet (IEM) as a fallback.

**Method: `fetch_latest_observation() -> Optional[ASOSReadingDocument]`**

Primary source — NWS API:
- URL: `https://api.weather.gov/stations/KBOS/observations/latest`
- Headers: `{"User-Agent": "KalshiWeatherTrader/1.0 contact@youremail.com"}` — NWS requires a User-Agent
- Parse the GeoJSON response. The temperature is at `properties.temperature.value` in **Celsius** — convert to Fahrenheit using `(C * 9/5) + 32`
- The observation time is at `properties.timestamp` as ISO8601 UTC
- The raw METAR text is at `properties.rawMessage`
- Validate that the observation timestamp is within the last 15 minutes. If it's stale (older than 15 minutes), log a warning and fall through to the IEM backup
- Use `tenacity` for retry: `@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))`

Fallback source — IEM API:
- If NWS fails or returns stale data, call IEM:
- URL: `https://mesonet.agron.iastate.edu/json/current.py?station=KBOS&network=ASOS`
- Parse `data[0].tmpf` for temperature in Fahrenheit
- Parse `data[0].valid` for timestamp
- Log that fallback was used: `"Using IEM fallback for KBOS observation"`

After successful fetch:
- Construct an `ASOSReadingDocument`
- Call `db_manager.write_document("asos_readings", doc.doc_id, doc.model_dump())`
- Call `db_manager.update_field_if_greater("markets", target_date, "current_max_observed", temperature_f)`
- Return the document

**Method: `fetch_last_n_hours(hours: int = 24) -> List[ASOSReadingDocument]`**

Used for backcalculation and calibration. Call IEM historical API:
- URL: `https://mesonet.agron.iastate.edu/json/asos.py?station=KBOS&data=tmpf&hours={hours}&tz=UTC`
- Returns a list of observations
- Filter to only observations within `get_trading_day_bounds(get_target_date())`
- Store each observation using `batch_write`

**Method: `get_current_max_observed() -> float`**

- Query Firestore `markets` collection for today's `current_max_observed`
- Return the value, or -999.0 if not found

---

### FILE: `ingestion/nwp_fetcher.py`

**Class: `NWPFetcher`**

Fetches hourly temperature forecasts from Open-Meteo for Boston. Open-Meteo provides free access to HRRR, GFS, and ECMWF data.

**Important:** Open-Meteo uses model-specific API endpoints. Know these:
- HRRR: `https://api.open-meteo.com/v1/forecast?...&models=hrrr_conus`
- GFS: `https://api.open-meteo.com/v1/forecast?...&models=gfs_seamless`
- ECMWF IFS: `https://api.open-meteo.com/v1/forecast?...&models=ecmwf_ifs025`

**Method: `fetch_model_forecast(model_name: str) -> Optional[NWPForecastDocument]`**

- `model_name` must be one of: `"HRRR"`, `"GFS"`, `"ECMWF"`
- Map model names to Open-Meteo model strings internally
- Request parameters: `latitude=42.3606`, `longitude=-71.0097`, `hourly=temperature_2m`, `temperature_unit=fahrenheit`, `timezone=America/New_York`, `forecast_days=2`
- Parse the response: `hourly.time` and `hourly.temperature_2m` are parallel arrays
- Filter to only the hours on `get_target_date()` (24 values, one per hour, in local Eastern time)
- Compute `predicted_daily_high = max(hourly_temps)` 
- Construct `NWPForecastDocument` and write to Firestore
- Generate `doc_id` as `f"{target_date.replace('-','')}_{model_name}_{fetched_hour:02d}"` where `fetched_hour` is the current UTC hour

**Method: `fetch_all_models() -> Dict[str, NWPForecastDocument]`**

- Calls `fetch_model_forecast()` for HRRR, GFS, and ECMWF
- Returns dict keyed by model name
- Log warning for any model that fails, but do not fail the whole function — partial results are acceptable

**Method: `get_blended_forecast(target_date: str) -> float`**

- Reads the latest `NWPForecastDocument` for each model from Firestore for the given target_date
- Reads `model_weights` from `system_state` document
- Returns weighted average: `sum(weight_i * predicted_daily_high_i)`
- If a model's forecast is missing, redistribute its weight proportionally among available models

---

### FILE: `ingestion/kalshi_fetcher.py`

**Class: `KalshiFetcher`**

This class handles all read operations from the Kalshi API. The execution module handles writes (orders).

**Authentication:**
- Kalshi API v2 uses RSA key authentication. Load `KALSHI_API_KEY` and `KALSHI_API_SECRET` from `.env`
- The secret key is a PEM-encoded RSA private key. Use the `cryptography` library to sign requests
- Implement a private method `_get_auth_headers(method: str, path: str) -> dict` that generates the required `KALSHI-ACCESS-KEY`, `KALSHI-ACCESS-TIMESTAMP`, and `KALSHI-ACCESS-SIGNATURE` headers per Kalshi's v2 spec
- Note to AI: If the Kalshi API v2 auth spec changes, this method is the only thing that needs to change — isolate all auth logic here

**Method: `get_market_data(ticker: str) -> Optional[KalshiMarketSnapshot]`**

- GET `{KALSHI_BASE_URL}/markets/{ticker}`
- Parse and return a `KalshiMarketSnapshot`
- The yes_ask and yes_bid are returned as integers (cents). Convert to decimal: `value / 100`
- Retry with tenacity on 429 (rate limit) and 5xx errors
- On 404, log error and return None — do not raise

**Method: `get_all_boston_temp_markets() -> List[KalshiMarketSnapshot]`**

- GET `{KALSHI_BASE_URL}/markets?series_ticker=KXHIGHTBOS&status=open`
- Returns all currently open Boston temperature markets
- Used for dynamic market discovery instead of hardcoding the ticker

**Method: `get_market_orderbook(ticker: str) -> Optional[dict]`**

- GET `{KALSHI_BASE_URL}/markets/{ticker}/orderbook`
- Returns the full orderbook for market depth analysis
- Store raw response — parsing will be done in execution module

---

## PHASE 3: THE QUANT ENGINE

**Agent Task:** Build `quant/kalman_filter.py` and `quant/monte_carlo.py`.

---

### FILE: `quant/kalman_filter.py`

**Class: `KalmanUpdater`**

Implements a 2-dimensional Kalman Filter to track the true current temperature and the NWP model bias simultaneously.

**State Vector:**

The state vector `x` is 2x1:
```
x = [T_t,  # True current temperature estimate
     B_t]  # NWP model bias estimate (NWP prediction - ASOS truth)
```

**State Transition Matrix (F) — for the Predict Step:**

Temperature evolves toward the NWP forecast. Bias persists (random walk):
```python
F = np.array([[1.0, 0.0],   # T_t+1 = T_t + u (control input)
              [0.0, 1.0]])  # B_t+1 = B_t (bias is persistent)
```

**Control Input Vector (u) — the NWP model delta:**
```python
# u = hourly NWP temperature change * dt (where dt = 5/60 for 5-min step)
# This drives the temperature toward what the model expects
u = np.array([[nwp_delta * dt],
              [0.0]])
```

**Observation Matrix (H) — for the Update Step:**
```python
H = np.array([[1.0, 0.0]])  # We observe temperature directly (not bias)
```

**Process Noise Matrix (Q):**

Tune these values. They encode how much we trust the model vs. the data:
```python
# Q_temp: how much can temperature vary randomly in 5 minutes beyond NWP
# Q_bias: how fast can the model bias drift
Q = np.array([[0.1, 0.0],    # Q_temp — small, temperature is physically smooth
              [0.0, 0.05]])  # Q_bias — very small, bias is slow-moving
```

Store Q values as constants in `config/settings.py` so they can be tuned without code changes.

**Measurement Noise (R):**
```python
# R encodes ASOS sensor noise
R = np.array([[0.6]])  # ASOS 0.5°C persistence filter produces ~0.9°F steps; R=0.6 reflects this
```

Also store in `config/settings.py`.

**Method: `__init__(initial_temp: float, initial_bias: float, initial_covariance: List[List[float]])`**

- Initialize `self.x = np.array([[initial_temp], [initial_bias]])`
- Initialize `self.P = np.array(initial_covariance)` (2x2)
- If loading from Firestore (resuming after restart), pass the saved state

**Method: `predict(nwp_delta_f: float, dt_hours: float) -> None`**

The predict step runs every time a new NWP hourly forecast comes in or between observations.
```
# Prediction equations:
x_prior = F @ x + u
P_prior = F @ P @ F.T + Q
```
Update `self.x` and `self.P` in place.

**Method: `update(asos_temp_f: float) -> Tuple[float, float]`**

The update step runs every time a new ASOS 5-minute observation arrives.
```
# Innovation (residual)
z = np.array([[asos_temp_f]])
y = z - H @ self.x

# Innovation covariance
S = H @ self.P @ H.T + R

# Kalman Gain
K = self.P @ H.T @ np.linalg.inv(S)

# State update
self.x = self.x + K @ y

# Covariance update (Joseph form for numerical stability)
I = np.eye(2)
self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T
```
Return `(self.x[0,0], self.x[1,0])` — the updated temperature and bias estimates.

**Method: `get_state() -> dict`**

Returns the current state as a dict ready to be serialized to Firestore:
```python
return {
    "kalman_temp_estimate": float(self.x[0, 0]),
    "kalman_bias_estimate": float(self.x[1, 0]),
    "kalman_covariance": self.P.tolist()
}
```

**Function: `load_or_initialize_filter(target_date: str, current_asos_temp: float) -> KalmanUpdater`**

- Try to load existing state from Firestore `system_state` collection using doc_id `f"KBOS_{target_date}"`
- If found, initialize `KalmanUpdater` with the saved state
- If not found (new day), initialize with `initial_temp=current_asos_temp`, `initial_bias=0.0`, `initial_covariance=[[1.0, 0.0], [0.0, 1.0]]`

**Function: `sync_filter_to_db(kf: KalmanUpdater, target_date: str) -> bool`**

- Call `kf.get_state()` and merge into the `system_state` Firestore document
- Always merge (do not overwrite other fields like `model_weights`)
- Returns True on success

---

### FILE: `quant/monte_carlo.py`

**Class: `MonteCarloEngine`**

Implements the Receding Horizon Monte Carlo simulation. This is the core pricing engine.

**The Stochastic Process:**

We use an Ornstein-Uhlenbeck (OU) process for temperature, which models mean-reversion toward the NWP forecast. This is more realistic than GBM for temperature. The discretized OU SDE for a single time step is:

```
dT = theta * (mu_t - T_t) * dt + sigma * sqrt(dt) * Z

Where:
- T_t = current temperature (from Kalman estimate)
- mu_t = NWP forecast temperature at time t (the "attractor")
- theta = mean-reversion speed (from system_state.theta_decay)
- sigma = intraday volatility (from system_state.sigma_volatility)
- dt = 5/60 hours (5-minute step expressed in hours)
- Z ~ N(0,1)
```

Additionally, apply drift corrections from calibration:
```
effective_mu_t = mu_t + kalman_bias_correction + time_window_drift_adjustment
```

Where `time_window_drift_adjustment` is `morning_drift_adjustment` if hour < 12, else `afternoon_drift_adjustment`.

**Method: `__init__(n_paths: int = 10000)`**

Load `n_paths` from `config/settings.py` (from `.env` variable `SIMULATION_PATHS`).

**Method: `run_simulation(market_params: dict) -> dict`**

`market_params` must include:
- `current_max_observed`: float — the hard floor
- `current_temp_kalman`: float — Kalman estimate of current temperature
- `kalman_bias`: float — current bias correction
- `nwp_hourly_temps`: List[float] — the blended NWP temperature curve for remaining hours
- `theta`: float — mean-reversion speed
- `sigma`: float — volatility
- `mu_drift`: float — overall drift correction
- `morning_drift_adjustment`: float
- `afternoon_drift_adjustment`: float
- `strike_temp`: int — the specific Kalshi strike being priced
- `remaining_day_fraction`: float — from `config.get_remaining_day_fraction()`

**Simulation Loop — implement exactly as follows:**

```python
import numpy as np
from datetime import datetime
import pytz

def run_simulation(self, market_params: dict) -> dict:
    hard_floor = market_params["current_max_observed"]
    T0 = market_params["current_temp_kalman"]
    nwp_curve = np.array(market_params["nwp_hourly_temps"])  # length = remaining hours
    theta = market_params["theta"]
    sigma = market_params["sigma"]
    bias = market_params["kalman_bias"]
    strike = market_params["strike_temp"]
    n_paths = self.n_paths

    # Time setup
    dt = 5 / 60  # 5-minute steps in hours
    remaining_hours = max(market_params["remaining_day_fraction"] * 12, 0.1)
    n_steps = int(remaining_hours / dt)

    # If no time left, return current reality
    if n_steps <= 0:
        p_above = 1.0 if hard_floor >= strike else 0.0
        return {"p_above_strike": p_above, "expected_high": hard_floor, "paths_simulated": 0}

    # Initialize all paths at current Kalman temperature estimate
    paths_current_temp = np.full(n_paths, T0)
    paths_max_temp = np.full(n_paths, hard_floor)  # Initialize path maxima at hard floor

    # Pre-generate all random numbers at once for efficiency
    Z = np.random.standard_normal((n_steps, n_paths))

    for step in range(n_steps):
        # Map step to hour index for NWP attractor
        hour_idx = min(int(step * dt), len(nwp_curve) - 1)
        mu_t = nwp_curve[hour_idx] + bias  # Apply Kalman bias correction

        # Apply time-window drift adjustment
        current_sim_hour = datetime.now(pytz.timezone("America/New_York")).hour + step * dt
        drift_adj = market_params["morning_drift_adjustment"] if current_sim_hour < 12 else market_params["afternoon_drift_adjustment"]
        mu_t += drift_adj

        # OU update for all paths simultaneously (vectorized)
        dT = theta * (mu_t - paths_current_temp) * dt + sigma * np.sqrt(dt) * Z[step]
        paths_current_temp = paths_current_temp + dT

        # Update running maximum for each path
        paths_max_temp = np.maximum(paths_max_temp, paths_current_temp)

    # Final path maxima are the simulated daily highs
    # Hard floor is already embedded (initialized paths_max_temp at hard_floor)

    # Calculate probability above strike
    p_above = np.mean(paths_max_temp >= strike)

    # Calculate distribution statistics
    return {
        "p_above_strike": float(p_above),
        "p_below_strike": float(1.0 - p_above),
        "expected_high": float(np.mean(paths_max_temp)),
        "std_dev": float(np.std(paths_max_temp)),
        "percentile_5": float(np.percentile(paths_max_temp, 5)),
        "percentile_25": float(np.percentile(paths_max_temp, 25)),
        "percentile_50": float(np.percentile(paths_max_temp, 50)),
        "percentile_75": float(np.percentile(paths_max_temp, 75)),
        "percentile_95": float(np.percentile(paths_max_temp, 95)),
        "hard_floor_applied": float(hard_floor),
        "paths_simulated": n_paths,
        "steps_simulated": n_steps,
    }
```

**Method: `price_full_distribution(base_params: dict, strikes: List[int]) -> Dict[int, dict]`**

- Run `run_simulation` once with all paths, then compute P(max >= strike) for each strike from the same path set
- This avoids running 10,000 paths separately for each strike — generate paths once, evaluate multiple strikes
- Return dict keyed by strike: `{49: {...result_dict}, 50: {...result_dict}, ...}`

**Function: `estimate_sigma_from_historical(asos_readings: List[ASOSReadingDocument]) -> float`**

- Takes a list of recent ASOS readings
- Computes 5-minute temperature changes
- Returns the annualized standard deviation of those changes converted to hourly sigma
- Formula: `sigma_hourly = std(5min_changes) * sqrt(12)` (12 five-minute periods per hour)
- This is used to calibrate `sigma_volatility` in `system_state`

---

## PHASE 4: THE CALIBRATOR

**Agent Task:** Build `calibration/calibrator.py`.

---

### FILE: `calibration/calibrator.py`

**Class: `SystemCalibrator`**

This class learns from yesterday's data to improve today's predictions. It runs once daily at midnight.

**Method: `calibrate_model_weights(lookback_days: int = 14) -> Dict[str, float]`**

Uses Brier Score to evaluate each NWP model's recent performance and adjusts weights.

Brier Score for a binary outcome: `BS = (p - o)^2` where `p` is the predicted probability and `o` is 1 or 0.

- For each of the last `lookback_days` days:
  - Query Firestore `markets` for `final_official_high`
  - Query Firestore `nwp_models` for each model's `predicted_daily_high` for that day
  - For a range of binary questions (e.g., "was the high above 50?"), compute the Brier Score for each model's implied probability
  - Implied probability for a model: use a normal distribution CDF centered on `predicted_daily_high` with `sigma=2.0°F`

- Average Brier Score for each model over the lookback period

- Apply Softmax on inverse Brier Scores to get new weights:
  ```python
  inverse_scores = {m: 1.0 / (bs + 1e-6) for m, bs in brier_scores.items()}
  total = sum(inverse_scores.values())
  new_weights = {m: v / total for m, v in inverse_scores.items()}
  ```

- Write new weights to `system_state` in Firestore
- Return the new weights dict

**Method: `calibrate_intraday_drift(target_date: str) -> dict`**

Analyzes yesterday's `intraday_snapshots` to learn systematic morning/afternoon model biases.

- Query all snapshots for yesterday from `intraday_snapshots` collection
- Group snapshots by time window:
  - Morning window: 06:00–12:00 Eastern
  - Afternoon window: 12:00–18:00 Eastern
- For each window:
  - Compute: `error = final_official_high - blended_predicted_high` for each snapshot in the window
  - Average the errors across all snapshots in the window
- This average error IS the drift adjustment for that window
- If no final_official_high is available yet, skip calibration and log warning
- Update `system_state`: set `morning_drift_adjustment` and `afternoon_drift_adjustment`
- Return `{"morning_drift_adjustment": float, "afternoon_drift_adjustment": float}`

**Method: `calibrate_sigma(lookback_days: int = 7) -> float`**

- Calls `ASOSFetcher.fetch_last_n_hours(hours=lookback_days*24)` to get historical readings
- Calls `MonteCarloEngine.estimate_sigma_from_historical()` 
- Updates `system_state.sigma_volatility`
- Returns the new sigma

**Method: `calibrate_theta(lookback_days: int = 7) -> float`**

Estimates the mean-reversion speed by fitting historical ASOS data to the OU process.

- For each day in the lookback period, compute the hourly temperature departures from the NWP forecast
- Fit an AR(1) model to the departure series: `departure_t+1 = phi * departure_t + noise`
- Convert AR(1) coefficient to OU theta: `theta = -ln(phi) / dt` where `dt = 1.0` hour
- Update `system_state.theta_decay`
- Return the new theta estimate

**Method: `record_snapshot() -> bool`**

This is called both by the scheduler (every 2 hours automatically) and by the "Force Snapshot" button in the UI.

- Collect all current state: ASOS temp, current_max_observed, all NWP predicted highs, Kalman state, Kalshi bid/ask
- Construct `IntradaySnapshotDocument`
- Run `MonteCarloEngine.run_simulation()` to get current fair value
- Compute `model_edge = fair_value - kalshi_ask`
- Write to `intraday_snapshots` collection
- Return True on success

---

## PHASE 5: THE EXECUTION ENGINE

**Agent Task:** Build `execution/trader.py`.

---

### FILE: `execution/trader.py`

**Class: `KalshiTrader`**

All write operations to Kalshi (order placement, cancellation). Read operations remain in `kalshi_fetcher.py`.

**Critical:** Every method must check `auto_trade_enabled` from Firestore before executing any real trade. If `auto_trade_enabled == False`, the method must log the kill switch status and return without placing an order.

**Critical:** Every method must check `DRY_RUN` from `.env`. If `DRY_RUN=true`, simulate the trade logic but do not call the Kalshi API. Log `"DRY RUN — order would have been placed"` with full order details.

**Authentication:** Reuse the auth header logic from `KalshiFetcher` — import and use the same `_get_auth_headers` method. Do not duplicate auth code.

**Method: `place_limit_order(ticker: str, action: str, contracts: int, price_cents: int) -> Optional[str]`**

- `action`: `"buy"` for YES contracts, `"sell"` for YES contracts (which is buying NO)
- `contracts`: number of contracts (start with 1 for safety)
- `price_cents`: limit price in cents (0-100)
- POST to `{KALSHI_BASE_URL}/portfolio/orders`
- Body: `{"ticker": ticker, "action": action, "type": "limit", "count": contracts, "yes_price": price_cents}`
- On success, return the `order_id` string
- On any error (including 400, 403, 429), log the full response and return None

**Method: `cancel_order(order_id: str) -> bool`**

- DELETE `{KALSHI_BASE_URL}/portfolio/orders/{order_id}`

**Method: `get_open_positions() -> List[dict]`**

- GET `{KALSHI_BASE_URL}/portfolio/positions`
- Returns list of open positions for KXHIGHTBOS series only (filter by series ticker)

**Method: `evaluate_and_trade(target_date: str) -> Optional[TradeLogDocument]`**

This is the main trading decision function. Runs every 5 minutes.

```
LOGIC:
1. Check kill switch (auto_trade_enabled). If False, return None immediately.
2. Fetch KalshiMarketSnapshot from KalshiFetcher.
3. Build market_params for MonteCarloEngine:
   - Load system_state from Firestore
   - Load current_max_observed from markets
   - Load blended NWP curve from NWPFetcher
   - Get remaining_day_fraction from config
4. Run MonteCarloEngine.run_simulation()
5. Extract fair_value_prob from simulation result

6. POSITION SIZING (Kelly Criterion — fractional Kelly at 25%):
   edge = fair_value_prob - kalshi_ask_decimal
   If edge > 0 (buying YES):
       b = (1 / kalshi_ask_decimal) - 1  # decimal odds
       kelly_fraction = (fair_value_prob * b - (1 - fair_value_prob)) / b
       fractional_kelly = 0.25 * kelly_fraction
       dollar_bet = min(fractional_kelly * MAX_POSITION_SIZE_DOLLARS, MAX_POSITION_SIZE_DOLLARS)
       contracts = max(1, int(dollar_bet / (kalshi_ask_decimal * 100)))

7. TRADE DECISION:
   - If fair_value_prob > (kalshi_ask_decimal + EDGE_THRESHOLD):
       action = "BUY_YES"
       price_cents = int(kalshi_ask_decimal * 100)  # buy at the ask
   - Elif fair_value_prob < (kalshi_bid_decimal - EDGE_THRESHOLD):
       action = "BUY_NO"
       price_cents = int((1 - kalshi_bid_decimal) * 100)  # buy NO at its ask
   - Else:
       log "No edge — no trade" and return None

8. Execute: call place_limit_order()
9. Construct and write TradeLogDocument to Firestore
10. Return the TradeLogDocument
```

---

## PHASE 6: THE STREAMLIT COMMAND CENTER

**Agent Task:** Build `ui/app.py`.

This is a multi-tab professional dashboard. Use Plotly for all charts. Import all backend modules directly.

---

### FILE: `ui/app.py`

**Page Config:**
```python
st.set_page_config(
    page_title="KBOS Temperature Trading Desk",
    layout="wide",
    initial_sidebar_state="expanded"
)
```

**Sidebar:**
- Display current Eastern time
- Display current target date from `get_target_date()`
- Display "6 PM ROLLOVER ACTIVE" warning in red if target date is tomorrow
- Display system health status: green checkmarks or red X for each data source (ASOS last updated within 10 mins, NWP last updated within 2 hours, Kalshi data live)
- Link to Kalshi market page

**Tab 1: Trading Desk (Real-Time)**

Layout: Two columns.

Left column:
- Large metric: Current ASOS Temperature at KBOS (auto-refresh every 60 seconds using `st.rerun()`)
- Large metric: Today's Maximum Observed (the hard floor)
- Large metric: Current Kalman Estimate
- Small metrics: Kalman Bias, Model Sigma, Theta

Right column:
- **THE KILL SWITCH:** A large red `st.button("⛔ KILL SWITCH — HALT ALL TRADING")` that writes `{"auto_trade_enabled": False}` to the markets Firestore document. Show current status ("Trading: ACTIVE ✅" or "Trading: HALTED ⛔")
- A green `st.button("▶ RESUME TRADING")` that sets `auto_trade_enabled` back to True

Below both columns:
- A `st.dataframe` table showing the edge analysis:
  - Columns: `Strike | Model Fair Value | Kalshi Ask | Kalshi Bid | Edge (YES) | Edge (NO) | Signal`
  - Color: If `Edge (YES) > EDGE_THRESHOLD`, highlight the row green. If `Edge (NO) > EDGE_THRESHOLD`, highlight orange.
- A "Run Simulation Now" button that triggers a fresh Monte Carlo run and refreshes this table

Recent trades section:
- Last 10 entries from `trade_logs` Firestore collection displayed as a table
- Columns: `Time | Action | Strike | Contracts | Price | Fair Value | Edge | DryRun | Status`

**Tab 2: The Visualizer**

Single full-width Plotly chart with:
- X-axis: Time (today from midnight to midnight Eastern)
- **Trace 1 (solid blue line):** Historical ASOS readings so far today — pulled from `asos_readings` collection filtered to today
- **Trace 2 (dashed lines, one per model):** NWP hourly temperature curves — HRRR in orange, GFS in green, ECMWF in purple
- **Trace 3 (solid orange line):** Blended NWP forecast (weighted average)
- **Trace 4 (horizontal dashed red line):** The current Hard Floor (current_max_observed)
- **Trace 5 (shaded band):** Monte Carlo 25th–75th percentile envelope for the simulated path distribution
- **Trace 6 (horizontal dashed lines):** Kalshi strike prices as labeled horizontal markers
- Show a vertical line at "NOW"
- Chart title: `f"KBOS Temperature — {get_target_date()} | Hard Floor: {current_max_observed}°F"`

Below chart:
- A second Plotly chart: Kalshi implied probability vs Model fair value over time today (from snapshots)
- X-axis: time, Y-axis: probability (0 to 1)
- Two lines: Kalshi implied (from snapshots) and Model fair value (from snapshots)

**Tab 3: Calibration & Manual Overrides**

Row 1: Calibration Status
- Show last calibration time, current model weights as a bar chart, current drift adjustments

Row 2: Manual Overrides (use with care)
- `st.slider("mu_drift override", min_value=-5.0, max_value=5.0, step=0.1, value=current_mu)` — changes Firestore on slider release
- `st.slider("kalman_bias override", -5.0, 5.0, 0.1, current_bias)` — same
- `st.slider("sigma override", 0.1, 5.0, 0.1, current_sigma)`
- A confirmation checkbox before writing: "I understand this will affect live trading immediately"

Row 3: Snapshot Controls
- `st.button("📸 Force Snapshot Now")` — calls `calibrator.record_snapshot()`, shows success/failure
- A table of today's snapshots from `intraday_snapshots` collection
- Columns: `Time | ASOS Temp | Max Observed | Blended NWP High | Kalshi Ask | Fair Value | Edge | Forced`

Row 4: Model Recalibration
- `st.button("🔄 Run Full Calibration Now")` — calls all calibrator methods and shows updated values

**Auto-refresh:** Use `st.empty()` and a 60-second sleep loop in a thread, or use `st.rerun()` with `time.sleep(60)` at the bottom of the script to auto-refresh the dashboard.

---

## PHASE 7: THE ORCHESTRATOR

**Agent Task:** Build `scheduler/orchestrator.py`.

---

### FILE: `scheduler/orchestrator.py`

**Uses APScheduler with `BackgroundScheduler`.**

This is the main entry point when running the system. Running `python scheduler/orchestrator.py` starts the full system including the data fetchers, quant engine, and trading logic. The Streamlit UI is run separately (`streamlit run ui/app.py`).

**Define these scheduled jobs:**

| Job Name | Function | Interval | Notes |
|---|---|---|---|
| `fetch_asos` | `ASOSFetcher.fetch_latest_observation()` then `KalmanUpdater.update()` then `sync_filter_to_db()` | Every 5 minutes | Core loop |
| `fetch_nwp` | `NWPFetcher.fetch_all_models()` then `KalmanUpdater.predict()` | Every 60 minutes | Model refresh |
| `evaluate_trade` | `KalshiTrader.evaluate_and_trade()` | Every 5 minutes | Trading loop |
| `take_snapshot` | `SystemCalibrator.record_snapshot()` | Every 2 hours | Calibration data |
| `midnight_calibration` | `SystemCalibrator.calibrate_model_weights()` + `calibrate_intraday_drift()` + `calibrate_sigma()` + `calibrate_theta()` | Daily at 00:05 Eastern | Full recalibration |
| `rollover_check` | `check_and_handle_rollover()` | Every 30 minutes | Handles 6 PM target date shift |

**Function: `check_and_handle_rollover()`**

- Call `get_target_date()` to get the current target
- Compare to the last known target date stored in a local variable
- If the target date changed (i.e., it just rolled over to tomorrow), log the rollover event, write a snapshot for the closing day, and initialize a new `markets` document for tomorrow with `current_max_observed=-999.0`

**Startup sequence in `main()`:**
1. Load `.env`
2. Initialize Firebase
3. Run `ASOSFetcher.fetch_latest_observation()` immediately on startup
4. Run `NWPFetcher.fetch_all_models()` immediately on startup
5. Initialize Kalman filter via `load_or_initialize_filter()`
6. Start the scheduler
7. Log: `"Orchestrator running. Trading system active."`
8. Keep the main thread alive with a `while True: time.sleep(60)` loop with graceful keyboard interrupt handling

---

## PHASE 8: FINAL INTEGRATION CHECKLIST

Before considering any phase complete, the AI must verify all of the following. Include this as a comment block at the top of each file:

```python
# INTEGRATION CHECKLIST FOR THIS MODULE:
# [ ] All imports resolved — no circular imports
# [ ] All Pydantic models imported from db/schemas.py (not redefined here)
# [ ] All DB writes use db_manager helpers (not raw Firebase calls)
# [ ] All external API calls have tenacity retry wrappers
# [ ] All secrets loaded from .env via python-dotenv
# [ ] All timestamps stored as UTC datetime objects, not strings
# [ ] structlog used for all logging, not print()
# [ ] Module has been tested with a mock Firestore (use mock_firestore library)
```

---

## HOW TO FEED THIS TO THE AI — EXACT PROTOCOL

**Do not deviate from this protocol.** AI coding tools hallucinate more when given large prompts without checkpoints.

1. **Start a fresh conversation.** Paste only this text as the first message:

> *"I am going to give you a Master Architecture Document in phases. Before I give you Phase 1, confirm that you have read the Preamble, understand the project directory structure in Section 0, and understand the requirements.txt. Do not write any code yet. Just confirm."*

2. After confirmation, paste **Phase 1** (Config + Schema + DB Manager). Say:
> *"Write the code for Phase 1 only. Include all files in Section 0 that belong to Phase 1. Do not skip the INTEGRATION CHECKLIST. Do not begin Phase 2."*

3. Read the output carefully. Check: Are all Pydantic models present? Are Firebase imports correct? Is `get_target_date()` implemented with pytz correctly? If anything is wrong, correct it before proceeding.

4. For each subsequent phase, say:
> *"Phase 1 is approved. Here is Phase 2. Ensure it imports schemas from db/schemas.py and helpers from db/db_manager.py. Do not redefine any Pydantic models. Write Phase 2 only."*

5. After Phase 5 (Streamlit UI), say:
> *"Now write orchestrator.py (Phase 7). Wire all modules together. Import only from the modules already built. Do not rewrite any existing logic."*

6. Final integration step:
> *"All phases are complete. Now write a requirements.txt, a .env.example, a .gitignore, and a README.md with exact setup instructions for running the orchestrator and the Streamlit UI simultaneously."*

---

## CRITICAL GAPS FILLED VS. THE ORIGINAL SPEC

For completeness, here is what this v3.0 spec adds beyond the two original responses that you must not let the AI omit:

- **Kalshi API v2 RSA authentication** is specified — the originals hand-waved this entirely
- **Joseph form covariance update** in the Kalman filter for numerical stability — naive implementations diverge
- **Vectorized Monte Carlo** using pre-generated random matrix — 10,000 paths in a single numpy operation, not a Python loop
- **Fractional Kelly position sizing** — the originals had a flat edge threshold with no sizing logic
- **IEM fallback** for ASOS data — NWS API goes down regularly
- **Atomic Firestore transaction** for `current_max_observed` — a race condition here would break the hard floor guarantee
- **Full `intraday_snapshots` schema** with `model_edge` and `is_forced` fields
- **`get_trading_day_bounds()`** with correct DST handling — this is where wrong implementations lose resolution arguments
- **`calibrate_theta()`** via AR(1) fitting — the original just said "store theta" with no derivation
- **Sigma estimation from historical data** instead of a hardcoded constant
- **Full kill switch** including a resume button — the original only had a one-way kill
- **Dry run mode** at `.env` level so the system never accidentally live-trades
- **Modular auth** in a single method in `kalshi_fetcher.py` that `trader.py` imports rather than duplicates