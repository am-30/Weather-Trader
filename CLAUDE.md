Here is a fully updated and condensed CLAUDE.md that accurately reflects the current state of the system:

```markdown
# CLAUDE.md — Kalshi Weather Trading System
# Last Updated: March 29, 2026

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
│   ├── asos_fetcher.py      # NWS time-series (primary), IEM bulk gap-fill
│   │                        # (secondary), AVWX METAR (tertiary), NWS latest
│   │                        # (last resort); 2-min schedule, 4-min API rate limit
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
    ├── test_monte_carlo.py   # 29 tests passing, includes
    │                        # NWP anchor + sigma cap tests
    ├── test_phase1.py       # 12 tests: Items 1.1/1.2/1.3 (Phase 1)
    ├── test_phase2.py       # 11 tests: Phase 2 (regime theta, exp
    │                        # weighting, model weights guard)
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
- 2D state vector: [dT_t (departure from NWP), B_t (model bias)]
  NOTE: was [T_t, B_t] before session 8 — state semantics changed
- Update step: on each new ASOS reading (scheduler fires every 2 min,
  API rate-limited to ≤every 4 min; METARs arrive every ~5–30 min)
- Predict step: every hourly NWP update
- Joseph form covariance update for numerical stability
- Q_temp=0.1, Q_bias=0.05, R=0.4 (in settings.py)
- NWP delta clamping: |nwp_delta| > kalman_max_nwp_delta (5°F/hr)
  is clamped before the predict step to guard against corrupt model
  spikes. Logs a warning when clamping fires.

H=[[1,1]] RESTRUCTURE (session 8 — do not revert):
  H = [[1.0, 1.0]]  (was [[1.0, 0.0]] — old form froze K[1] at ~0)
  z = asos_temp - nwp_current_hour  (departure observation)
  temperature = nwp_current_hour + dT + B  (NOT just nwp + dT)
  predict() does NOT shift dT — only inflates P and updates
    _nwp_current. Departure is stable across NWP hours.
  update() requires nwp_current_hour arg; job_fetch_asos_and_update()
    looks up blended NWP curve for current ET hour before calling it.
  load_or_initialize_filter() takes nwp_at_load_time; reconstructs
    dT = stored_kalman_temp - nwp_at_load_time - stored_bias
  K[1] ≈ 0.42 on first tick (was ~0 with old H=[[1,0]])

Kalman warm-start (implemented — session 7):
  On each new trading day (state row missing), load_or_initialize_filter()
  checks yesterday's system_state and initialises with:
    initial_bias  = yesterday.kalman_bias_estimate
    initial_cov   = yesterday.kalman_covariance * 1.2  (inflated 20%)
  Eliminates daily cold-start at bias=0.0 which kept the anchor offset
  active longer than needed and slowed convergence.

Kalman gap inflation (implemented — session 7):
  When restoring existing state, if (now - last_updated_utc) > 0.5h,
  inject accumulated process noise via predict(nwp_delta=0, dt=1.0)
  repeated up to 12 times (one per gap hour). Without this, Kalman
  gain K collapses to ~0.024 after many updates and a 9°F innovation
  moves T by only ~0.2°F/tick — nearly unresponsive.

- sigma/theta calibration meaningless for first ~30 days of operation.
  No UI warning for cold-start period.

### Monte Carlo — NWP Anchor Fix (Critical)
- Process: Ornstein-Uhlenbeck (NOT geometric Brownian motion)
- Core equation: dT = theta*(mu_t - T_t)*dt + sigma*sqrt(dt)*Z
- dt = 5/60 hours

NWP ANCHOR OFFSET (implemented — do not remove):
  peak_hour_idx = argmax of the WINDOW portion of nwp_curve
    (for stitched post-rollover curves, search starts at
    bridge_steps // 12 so bridge peaks don't force weight=1.0)
  anchor_weight = 1 - hours_to_peak / peak_hour_idx
    (fallback = 1.0 when peak_hour_idx == 0)
  nwp_anchor_offset = (T0 - nwp_curve[hour_offset]) * anchor_weight
  mu_t = nwp_curve[hour_idx] + nwp_anchor_offset + bias +
         drift_adj

anchor_weight ramps from 0 (far from peak, NWP curve dominates)
to 1 (at/past peak, attractor fully anchored to T0). This prevents
the original full-offset from permanently projecting an early-
morning T0 gap onto the afternoon peak. As Kalman bias warms up
over weeks, nwp_anchor_offset trends toward zero naturally.

BRIDGE STEPS (stitched post-rollover only — do not remove):
  bridge_steps = bridge_hours * 12  (12 five-minute steps/hour)
  During steps 0 .. bridge_steps-1: paths evolve normally but
  paths_max is NOT updated. Bridge temps (tonight's hours) are
  outside the NWS observation window for the next trading day and
  must not contaminate the daily max distribution. Without this
  guard, T0 (e.g., 46°F at 9 PM) immediately locks paths_max
  above the window's plausible range (~40°F), causing ~100%
  probability on all strikes.

OU SIGMA CAP (implemented — do not remove):
  sigma_used = min(sigma, ou_max_stationary_std * sqrt(2 * theta))
  ou_max_stationary_std = 2.0°F (default, overridable via env var
    OU_MAX_STATIONARY_STD; was 1.0 until March 29 — raised because
    with theta≈0.21 the old default suppressed calibrated sigma by
    47%, artificially thinning distribution tails)

  Without this cap, a calibrated sigma >> sqrt(2*theta) makes the
  per-step noise >> restoring force (e.g. sigma=1.385, theta=0.1559
  → noise/restoring ratio = 31×, half-life 4.4h). Paths spike 5–7°F
  above a declining NWP attractor before mean-reversion catches up,
  locking in wildly inflated paths_max values. The cap enforces that
  the OU stationary std ≤ ou_max_stationary_std ≈ NWP intraday RMSE.
  Capping is logged at DEBUG as mc.sigma_capped with noise/restoring
  ratio. mc.sigma_effective is logged on every run (cap_active flag).

  MCParams carries ou_max_stationary_std as a field. mc_params_builder
  reads state.ou_max_stationary_std_calibrated first; falls back to
  settings.ou_max_stationary_std when None (Phase 3 cold start).

  Phase 3 (implemented — March 29, 2026): calibrate_ou_max_stationary_std()
  in calibrator.py computes blended NWP RMSE from morning forecasts
  (first fetch in [10 AM, 1 PM) ET — no lookback bias) vs CLI-confirmed
  final_official_high, then sets calibrated_cap = RMSE × 1.5, clamped
  to [0.5, 5.0]°F. Requires ≥10 qualifying dates. Stored in
  system_state.ou_max_stationary_std_calibrated. Called from
  run_full_calibration() after calibrate_persistence_offset().
  Tab 5 Section 2F displays per-model RMSE chart, cap status, and
  manual calibration control.

sigma estimation (estimate_sigma_from_historical):
  Uses hourly-bucket diffs rather than 5-minute diffs (session 7).
  The ASOS 0.5°C persistence filter causes 5-minute readings to jump
  in 0.9°F increments, inflating mean(dT²/dt) by 3-4×. Bucketing to
  the nearest top-of-hour and computing consecutive hourly diffs
  averages through sensor steps, recovering true hourly volatility.
  Gap guard: readings > 40 minutes from top-of-hour are excluded.
  Sigma clamped to [0.1, 1.5] (was [0.1, 4.0]).
  Returns (pooled_sigma, sigma_by_block) tuple (Phase 1 — session 8).
  sigma_by_block: dict keyed by block label ("0-6", "6-10", etc.),
  falls back to pooled_sigma for blocks with < 10 samples.

TWO-REGIME THETA (Phase 2):
  theta_am: Optional[float] — OU mean-reversion speed for ET hours [6,13)
  theta_pm: Optional[float] — OU mean-reversion speed for ET hours [13,20)
  Overnight hours (0-5, 20-23) always use scalar theta fallback.
  Stored as separate Numeric(7,4) columns in system_state (matching
  morning/afternoon drift pattern — not JSONB).
  run_simulation() precomputes step_theta[n_steps] array (same pattern
  as step_noise): for each step, pick theta_am/theta_pm based on ET hour,
  or scalar theta for overnight.
  Falls back to (None, None) when either regime has < 20 weighted-equivalent
  pairs — prevents calibration on sparse early-operation data.

EXPONENTIALLY WEIGHTED CALIBRATION (Phase 2):
  calibration_lookback_days=30 (was 7 default), calibration_decay_tau_days=10
  Both in settings.py and respected by calibrate_sigma(), calibrate_theta(),
  calibrate_theta_by_regime(), and calibrate_intraday_drift().
  Weight per day: w = exp(-d / tau) where d = days before most recent date.
  estimate_sigma_from_historical() uses weighted accumulators:
    sigma = sqrt(Σ(w*dT²) / Σ(w)) instead of flat mean(dT²).
  calibrate_theta() / calibrate_theta_by_regime() use weighted OLS through
    origin: phi = Σ(w*x*y) / Σ(w*x²) (valid because OU departures have
    zero mean by construction).
  calibrate_intraday_drift() uses 14-day lookback (was 7) with exponential
    weighting — drift is more non-stationary than sigma.

MODEL WEIGHTS EQUAL-WEIGHT GUARD (Phase 2):
  calibrate_model_weights() counts settled dates with both final_official_high
  AND kalshi_strike. If < 14 qualifying dates, returns equal weights (1/3 each)
  immediately. Brier scores are pure noise with < 14 dates (insufficient
  signal to discriminate between HRRR/GFS/ECMWF).

TIME-VARYING SIGMA (Phase 1 — session 8):
  Five ET-hour blocks: 0-6, 6-10, 10-14, 14-18, 18-24
  Constants: SIGMA_BLOCKS, SIGMA_BLOCK_LABELS, _sigma_block_for_hour()
  MCParams.sigma_by_block: Optional[dict[str, float]] — None → scalar sigma
  run_simulation() precomputes step_noise[n_steps] array before hot loop:
    sigma_max applied per-block (cap still enforced per-block)
    step_noise[s] = min(block_sigma, sigma_max) * sqrt_dt
  Stored in system_state.sigma_by_block (JSONB, idempotent migration).
  Populated by calibrate_sigma() and passed through mc_params_builder.py.

NWP ANCHOR OFFSET DOUBLE-COUNT FIX (Phase 1 — session 8 Item 1.2):
  With H=[[1,1]], T0 = nwp + dT + B. Old formula: nwp_anchor_offset =
  (T0 - nwp_ref) * weight = (dT + B) * weight → B counted twice.
  Fix: gap_after_bias = (T0 - nwp_ref) - params.bias; offset uses gap_after_bias.

PERSISTENCE FILTER OFFSET (Phase 1 — session 8 Item 1.3):
  NWS daily max exceeds ASOS tabular max by ~0.2–0.4°F on average.
  effective_floor = hard_floor + persistence_filter_offset (default 0.3°F)
  paths_max initialized at effective_floor (not hard_floor).
  DB hard_floor is never modified — offset is MC-only.
  settings.persistence_filter_offset (ge=0.0, le=0.5).
  Calibrated by calibrate_persistence_offset(lookback_days=30):
    gap = final_official_high - max(ASOS readings) per settled date
    excludes gaps ≤ 0 (ASOS over-read artifact); requires ≥ 5 dates
    returns mean(positive_gaps) clamped to [0.0, 0.5]
  Stored in system_state.persistence_filter_offset (NUMERIC(4,2)).
  Both fields added via _migrate_system_state_phase1_columns().

- Hard floor: paths_max array initialized at hard_floor + persistence_filter_offset
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

### ASOS Fetch Architecture (as of session 7)

fetch_current_observation() tries four sources in order:
  1. NWS time-series (PRIMARY)
     _fetch_nws_since(since_utc) — GET /stations/KBOS/observations
     ?start=...&limit=N where N is gap-proportional (max 500).
     Returns all observations since last stored reading including
     max6h_f on each. No IEM indexing lag (~1–3 min faster on
     very recent data). Bug fixed: was hardcoded limit=10, which
     caused overnight gaps to only retrieve the 10 most-recent
     readings, silently skipping the historical window.

  2. IEM Mesonet bulk gap-fill (SECONDARY)
     _fetch_iem_since(last_stored_timestamp) — one CSV request
     returns ALL readings since last stored. Useful fallback when
     NWS API is down or rate-limited.

  3. Aviation Weather Center METAR JSON (TERTIARY)
     _fetch_aviationweather_metar() — hits aviationweather.gov
     /api/data/metar. KBOS issues SPECI (special) METARs on any
     significant condition change.

  4. NWS /observations/latest (LAST RESORT)
     _fetch_nws_latest() — single latest reading only; also
     provides max6h_f when the other sources fail entirely.

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
  bridge_steps = bridge_hours * 12
    paths_max is frozen during bridge steps; only updated once the
    NWS observation window for tomorrow opens at midnight EST.

This allows the OU simulation to correctly price overnight temperature
peaks (e.g., daily high at 1 AM when a warm front passes) that would
be missed if the simulation jumped straight to midnight tomorrow.

The anchor search is restricted to the window portion of the stitched
curve (nwp_curve[bridge_hours:]) so that the descending bridge temps
(tonight's still-warm readings) don't push peak_hour_idx to 0 and
force anchor_weight=1.0 throughout, which inflated every strike to
~100% before this fix.

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
   No UI warning for cold-start period.

10. [RESOLVED — session 7] Kalman bias cold-starts at 0.0 every new trading day
    Fixed: load_or_initialize_filter() now reads yesterday's system_state
    and warm-starts with prior day's converged bias and inflated covariance.
    Also added gap inflation (inject process noise proportional to downtime)
    to prevent Kalman gain from collapsing after app restarts.

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
    nws_cli_fetcher.py, db_manager.py, calibrate_ou_max_stationary_std().
    92 tests passing across test_kalman.py, test_monte_carlo.py,
    test_phase1.py, test_phase2.py, test_ingestion.py.

---

## What Has Been Built (as of March 29, 2026)

- [x] Phase 1: Config, schemas, db_manager
- [x] Phase 2: ASOS + NWP + Kalshi + NWS CLI fetchers
- [x] Phase 3: Kalman filter + Monte Carlo (NWP anchor fix applied)
- [x] Phase 4: Calibrator (7-day drift window) + snapshot manager
- [x] Phase 5: Execution engine + trader + position tracking
- [x] Phase 6: Streamlit — Trading Desk, Visualizer, Calibration,
              Model Transparency (Stages 1–6 complete)
- [x] Phase 7: Orchestrator + two-phase settlement + startup
              catch-up sequence
- [x] Session 7: NWS time-series as primary ASOS source (gap-
              proportional limit, limit=10 bug fixed); Kalman
              warm-start from yesterday's bias + gap inflation;
              NWP delta clamping in predict step; backfill_historical_
              asos() for cold-start seeding; calibration timestamp
              stamping fix; sigma estimation via hourly bucketing;
              OU sigma cap (ou_max_stationary_std) to prevent near-
              random-walk paths inflating P(daily_max)
- [x] Session 8: Kalman H=[[1,1]] restructure (bias observable, K[1]≈0.42
              on first tick); Phase 1 MC fixes: anchor offset no longer
              double-counts Kalman bias (1.2); persistence filter offset
              raises paths_max floor by ~0.3°F (1.3); time-varying sigma
              per ET-hour block with precomputed step_noise (1.1);
              calibrate_persistence_offset() + sigma_by_block JSONB in
              system_state; 81/81 tests passing
- [x] Phase 2: Two-regime theta (theta_am/theta_pm for ET hours 6-13/13-20);
              exponentially weighted calibration windows (lookback=30d,
              tau=10d) for sigma, theta, and drift; model weights equal-
              weight guard for < 14 qualifying dates;
              92/92 tests passing
- [x] March 29: ou_max_stationary_std raised 1.0 → 2.0 (cap was suppressing
              calibrated sigma by 47% with theta≈0.21, thinning tails);
              Phase 3 NWP RMSE calibration of ou_max_stationary_std
              (calibrate_ou_max_stationary_std(), system_state columns
              ou_max_stationary_std_calibrated + nwp_rmse_n_dates, MCParams
              field, mc_params_builder fallback, Tab 5 Section 2F RMSE chart);
              kalman sync_filter_to_db preserves all calibration fields
              including Phase 2 + Phase 3; orchestrator dual-syncs Kalman
              state to target_date after 6 PM rollover;
              92/92 tests passing
- [ ] End-to-end live cycle validation
- [ ] NWS CLI regex verification against real page
```
March 17 Updates
- Fixed hard floor bug where once trader roller over to tracking the market for the next day at 6 PM, the highest temp observed in the current day was being set as the hard floor for that ensuing day. The hard floor value is also now being rounded down to account for oddities in how the NWS rounds their max recorded temps.

March 23 - A senior quantitative trader and computer scientist has reviewed our system architecture and prepared a system improvement plan that is documented in System_Redesign.txt . We will be working to incorporate this feedback in the proposed phases. We will not implement multiple phases at once, so all plans must focus on the current phase one at a time.