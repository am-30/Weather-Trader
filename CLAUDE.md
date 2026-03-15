# CLAUDE.md — Kalshi Weather Trading System

You are a senior quantitative developer and software engineer. You are building a professional-grade, automated weather-trading system targeting the Kalshi Daily Maximum Temperature market for Boston Logan Airport (KBOS). This is not a toy project. Every module must be production-quality: type-annotated, logged, error-handled, and independently testable.

## Project Context

This system was designed through an extensive architecture session.
Station: **KBOS (Boston Logan Airport)**, hardcoded throughout.

## Core Architecture Decisions (final — do not suggest alternatives)

- **Database**: Replit native PostgreSQL via `DATABASE_URL` env variable
- **ORM**: SQLAlchemy with psycopg2-binary (never raw psycopg2)
- **UI**: Streamlit with Plotly charts
- **Scheduler**: APScheduler `BackgroundScheduler` (runs inside Streamlit process)
- **HTTP client**: httpx with tenacity retry logic
- **Validation**: Pydantic v2
- **Logging**: structlog exclusively (never `print()`)
- **Python**: 3.11+
- **Authentication**: RSA-PSS with MGF1(SHA256) + DIGEST_LENGTH salt (Kalshi elections API)
- **Position sizing**: Fractional Kelly at 25%
- **Simulation**: Ornstein-Uhlenbeck process, 10,000 paths, vectorized NumPy
- **Kalman Filter**: 2D state vector [temperature + bias], Joseph form covariance update
- **Hard floor**: `current_max_observed` atomically maintained via PostgreSQL `GREATEST()`
- **Kill switch**: `auto_trade_enabled` flag in `markets` table
- **Dry run**: `DRY_RUN` env variable — always check before placing real orders

## Non-Negotiable Coding Rules

- Pydantic v2 for all data models (defined ONLY in `db/schemas.py`, imported everywhere else)
- structlog for all logging — never `print()`
- tenacity retry decorators on all external API calls
- python-dotenv for all secrets — never hardcode credentials
- SQLAlchemy for all database operations
- Type annotations on every function
- Full docstrings with Args, Returns, Raises on every function
- try/except on every database write and external API call
- No circular imports
- Timestamps: always store UTC `datetime` objects, never strings
- Temperatures: Fahrenheit floats, one decimal precision

## Project Structure

```
kalshi_weather_trader/
├── config/settings.py          # pydantic-settings, OU/Kalman params, intervals
├── db/
│   ├── schema.sql              # PostgreSQL DDL
│   ├── db_manager.py           # all DB read/write operations
│   └── schemas.py              # ALL Pydantic models live here
├── ingestion/
│   ├── asos_fetcher.py         # NWS 5-min ASOS + IEM fallback
│   ├── nwp_fetcher.py          # Open-Meteo HRRR/GFS/ECMWF hourly forecasts
│   ├── kalshi_fetcher.py       # Kalshi API v2 (markets, orders, positions)
│   └── nws_cli_fetcher.py      # NWS Climate Summary for authoritative daily high
├── quant/
│   ├── kalman_filter.py        # 2D Kalman filter (temp + bias)
│   └── monte_carlo.py          # OU simulation, CDF pricing, normalization
├── execution/trader.py         # Kelly sizing + order execution
├── calibration/calibrator.py   # Brier score calibration, drift adjustment
├── scheduler/orchestrator.py   # APScheduler jobs + startup sequence
├── ui/app.py                   # Streamlit dashboard (4 tabs + Model Transparency)
└── tests/
    ├── test_kalman.py
    ├── test_monte_carlo.py
    └── test_ingestion.py
```

## Database Schema

Tables: `markets`, `nwp_forecasts`, `asos_readings`, `system_state`, `intraday_snapshots`, `trade_logs`.

Critical notes:
- `kalshi_strike` columns use `NUMERIC(5,1)` — migrated from `SmallInteger` via `_migrate_kalshi_strike_columns()` on every startup. Live strikes are floats (e.g., 44.5°F).
- Hard floor updates use a single `UPDATE ... SET col = GREATEST(col, :val)` — atomic.

## Key Mathematical Specifications

### Kalman Filter
- 2D state: `[T_t (true temp), B_t (model bias)]`
- Predict: triggered by hourly NWP delta
- Update: triggered by ASOS reading every 5 minutes
- Joseph form covariance update for numerical stability
- `Q_temp=0.1`, `Q_bias=0.05`, `R=0.3`

### Monte Carlo (Ornstein-Uhlenbeck)
- `dT = theta * (mu_t - T_t) * dt + sigma * sqrt(dt) * Z`
- `mu_t = NWP_forecast[hour_offset] + Kalman_bias + drift_adj`
- `dt = 5/60` hours (5-minute steps)
- Hard floor: `paths_max` initialized at `current_max_observed` and enforced every step
- Fully vectorized: pre-generated `(n_steps, n_paths)` NumPy random matrix
- `hour_offset` is **UTC hour** — NOT Eastern hour (fixing this was a critical bug)

### Kalshi Bucket Semantics (CONFIRMED from live API)
B-ticker caps are **INCLUSIVE**. Ticker `B38.5` → `floor=38, cap=39` covers `{38°F, 39°F}`.

NWS reports daily max in whole degrees Fahrenheit. The continuous settlement boundary between `{38,39}` and `{40,41}` is at **39.5°F** (half-integer), not 40.0°F. A continuous MC path value of 39.6°F rounds to 40°F → B40.5 bucket.

MC CDF must be evaluated at half-integer boundaries:
- Bottom bucket `T38` (cap=38): `P = 1 − CDF(37.5)`
- Middle bucket `B38.5` (floor=38, cap=39): `P = CDF(37.5) − CDF(39.5)`
- Top bucket `T46` (floor=46): `P = CDF(45.5)`

Constants in `monte_carlo.py`:
```python
_TEMP_RESOLUTION = 1.0   # °F — NWS integer resolution
_HALF_STEP = 0.5         # °F — rounding boundary offset
```

### Probability Normalization
`compute_normalized_market_probs()` in `monte_carlo.py` is the single canonical source for per-market P(YES). It:
1. Calls `_interpolate_cdf()` for each boundary (linear interpolation on discrete MC CDF)
2. Detects partition gaps (checks `cap[i] + 1 == floor[i+1]` for middle buckets)
3. Normalizes if `|Σ − 1.0| ≤ 0.10`; logs ERROR if gap exceeds 10%
4. Returns `(ticker → prob, sum_raw, gaps_list)`

**All three call sites** — `trader.py`, Tab 1 UI, Stage 5 UI — must use this function. Do not call `compute_yes_prob()` directly in new code.

### Position Sizing
- `b = (1/ask_decimal) − 1`
- `kelly = (p*b − (1−p)) / b`
- `contracts = min(floor(0.25 * kelly * MAX_SIZE / (ask * 100)), MAX_SIZE)`

## Scheduler Jobs

| Job | Interval | Notes |
|-----|----------|-------|
| fetch_asos + kalman update | Every 5 min | |
| fetch_nwp + kalman predict | Every 60 min | |
| evaluate_and_trade | Every 5 min | kill switch checked first |
| take_snapshot | Every 2 hours | |
| midnight_calibration | Daily 00:05 ET | Brier score calibration |
| rollover_check | Every 30 min | 6 PM ET target_date rollover |
| job_check_settlement | Every 30 min after 7 PM ET | ASOS preliminary high |
| job_confirm_settlement | Daily 10:05 AM ET | NWS CLI authoritative high |

APScheduler runs as a background thread inside the Streamlit process (`_maybe_start_scheduler()`). Stops when the browser tab closes.

## Trading Logic

- Edge threshold: `settings.edge_threshold` (default 0.05)
- Buy YES if: `fair_value − ask > threshold`
- Buy NO if: `bid − fair_value > threshold`
- Re-checks kill switch immediately before order submission
- Reduces Kelly contracts by existing position exposure (from `get_positions()`)
- All decisions (including no-trade) logged to `trade_logs`

## Kalshi API Integration

Live API confirmed details:
- **Ticker format**: `KXHIGHTBOS-26MAR15-T38`, `KXHIGHTBOS-26MAR15-B44.5`
- **Auth**: RSA-PSS with MGF1(SHA256) + DIGEST_LENGTH salt (NOT PKCS1v15)
- **Signing path**: must include `/trade-api/v2` base prefix
- **Market status**: filter for `status in {active, initialized}` client-side
- **Price fields**: API returns `yes_bid_dollars`/`yes_ask_dollars` as floats in [0,1]. `_normalize_market()` converts to cent integers.
- **Strike extraction**: `extract_strike_from_market()` reads `floor_strike` from API response directly (more reliable than ticker regex)
- **Market query**: 3-strategy fallback: series_ticker → event_ticker → events/{event}/markets

## Settlement (Two-Phase)

1. **7 PM ET** (ASOS preliminary): `job_check_settlement()` writes `max(ASOS readings)` as preliminary `final_official_high`, sets `market_status='settled'`, disables auto-trading.
2. **10:05 AM ET next day** (NWS CLI authoritative): `job_confirm_settlement()` fetches `https://forecast.weather.gov/product.php?site=BOX&product=CLI&issuedby=BOS`, overwrites with official high, triggers calibration.

## Startup Sequence

On every startup:
1. Runs `_migrate_kalshi_strike_columns()` idempotently
2. Scans ASOS history for the current trading day → catches up hard floor if app was offline during peak hours
3. Checks `last_calibrated_utc` → runs calibration immediately if midnight job was missed
4. Checks if yesterday's `final_official_high` is still the ASOS preliminary → attempts NWS CLI fetch to confirm

## Architectural Deviations from Original Spec

| Original Spec | Actual Implementation |
|---|---|
| Separate "Trading Engine" Replit workflow | APScheduler inside Streamlit process |
| 7 phase-by-phase builds | All phases built concurrently, no phase gates |
| `gfs_seamless` as GFS model | `gfs_global` (pure GFS); seamless demoted to fallback — was producing identical near-term results to HRRR |
| `final_official_high` from ASOS only | Two-phase: ASOS preliminary → NWS CLI authoritative |
| `SmallInteger` for kalshi_strike | `NUMERIC(5,1)` — decimal strikes exist (44.5°F) |
| Eastern hour as `hour_offset` | **UTC hour** — Eastern caused ~5-hour systematic bias |
| `compute_yes_prob()` with `.get(key, default)` | `_interpolate_cdf()` + half-integer boundaries |
| Integer bucket boundaries | Half-integer boundaries (NWS rounding semantics) |
| Startup: no catch-up logic | Hard floor catch-up + missed calibration catch-up |
| No position tracking | `get_positions()` reduces Kelly by current exposure |

## Known Issues

### BLOCKING

**1. Sigma inflation of `paths_max` — UNRESOLVED**

`sigma=1.698°F/√hr` (calibrated from 5-minute ASOS data via sample std of consecutive differences) inflates `paths_max` by ~2°F above the NWP attractor peak over a typical afternoon simulation.

Math: OU stationary std = `sigma / sqrt(2*theta)` ≈ 0.85°F. Expected overshoot of running maximum over `N_eff ≈ 14` effective independent windows: `0.85 × sqrt(2 × ln(14)) ≈ 2.0°F`. This is inherent to the running maximum of a mean-reverting process — sigma controls intraday noise, not the daily-max distribution width.

Observed: T0=37.4°F, NWP daily max=39.1°F, hard_floor=37.4°F → MC mean_max=40.2°F (1.1°F above NWP peak).

**Root cause**: sigma is calibrated for 5-minute temperature increments, but `paths_max` inherits the cumulative extreme-value bias. The NWP ceiling cap approach was tried and reverted (made the problem worse / changed prediction direction incorrectly). A new approach is needed before the system can be trusted for real money.

**Options to evaluate**:
- Re-calibrate sigma as the std of the daily-max distribution residuals (vs. ASOS observed daily max), not intraday increments
- Use a separate "daily-max volatility" parameter derived from historical max-temp forecast errors
- Apply an analytical expected-overshoot correction when reporting `P(max ≥ strike)`

**2. NWS CLI regex not verified**

`fetch_official_daily_high()` patterns (`CLIMATE SUMMARY FOR`, `MAXIMUM\s+`) have not been run against a real fetched page. The 10:05 AM settlement job depends on this.

### HIGH PRIORITY

**3. MCParams construction duplicated in 3 places**

`trader.py`, Tab 1 edge table UI, and Stage 3 Model Transparency each independently build `MCParams`. If `hour_offset`, `drift_adj`, or `nwp_curve` logic diverges in one place, predictions silently differ. This is the highest architectural debt item.

**4. `_normalize_market()` field format assumption**

Assumes `yes_bid_dollars`/`yes_ask_dollars` are floats in [0,1]. If Kalshi changes field format to cents (integers in [0,100]), the ×100 conversion silently inflates all prices by 100× with no error. Should add a range guard: `if value > 1.0: raise ValueError(...)`.

**5. Hard floor SQL atomicity — verify**

`db_manager.update_hard_floor()` must use a single `UPDATE ... SET col = GREATEST(col, :val)` with no preceding SELECT. If there's a read-then-write pattern, the atomic guarantee is lost under concurrent writes.

### MEDIUM PRIORITY

**6. KALSHI_ENV=demo is cosmetic** — setting is validated but API URL always comes from `kalshi_api_base_url` directly. Either wire it or remove it.

**7. Blended forecast truncates to shortest model curve** — blend should be per-hour with whatever models have data at that hour, not cut to the shortest.

**8. Stage 4 histogram uses synthetic normal** — `numpy.random.normal(mean_max, std_max, 5000)` does not reflect hard-floor truncation shape. Should pass actual `paths_max` array through `MonteCarloResult`.

**9. Position tracking is additive-only** — Kelly reduction accounts for long (YES) exposure but not short (NO) positions.

**10. NWS CLI version cycling has no retry** — network error on version 1 suppresses entire fetch (intentional, but fragile on transient timeouts).

**11. Startup catch-up equality check is fragile** — `final_official_high == current_max_observed` as proxy for "still preliminary" will false-positive when they happen to match.

### LOW PRIORITY / CLEANUP

**12. Stage 6 (Historical Calibration Performance) not built** — requires multi-day DB queries, date-picker replay, Brier score time series, weight convergence chart.

**13. Sigma cold-start problem** — calibrated sigma/theta require 30+ days of Brier score history. For early operation the system silently uses `settings.ou_sigma`/`settings.ou_theta` defaults with no UI warning about cold-start validity.

**14. test coverage gaps**:
- `kalshi_fetcher.py` (auth headers, market normalization, positions)
- `orchestrator.py` (job scheduling, startup sequence)
- `ui/app.py` (no Streamlit testing)
- `nws_cli_fetcher.py` (HTML parsing against fixture page)
- `db_manager.py` (requires live PostgreSQL or extensive mocking)

**15. `hour_et` variable naming** — in `trader.py` and `calibrator.py`, `hour_et` is now only used for AM/PM drift selection (not for `nwp_curve` indexing). Should be renamed `_hour_et_for_drift` or similar to prevent confusion with `hour_offset`.

## What to Verify Before Using Real Money

In priority order:
1. **Sigma inflation fix** — system is currently overconfident on high-temperature outcomes
2. **Hard floor SQL atomicity** — verify `db_manager.update_hard_floor()` uses single UPDATE
3. **NWS CLI regex** — run `fetch_official_daily_high()` against a real fetched page
4. **`_normalize_market()` guard** — add input range validation for price fields
5. **End-to-end cycle** — confirm full fetch → Kalman update → MC → trade evaluation → snapshot completes from cold start with live data
