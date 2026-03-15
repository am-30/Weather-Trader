This is exactly the right instinct. What you're describing is called an **audit trail** or **model transparency layer** — it's standard practice in professional quant systems precisely because a model that produces a number you can't trace back to its inputs is worthless for trading. You need to be able to look at any probability output and answer "why did the model say 67%?" all the way down to the raw data.

Let me think through this with you properly before we write any prompts.

---

## The Core Problem You're Solving

The app currently produces a fair value probability. But you have no way to answer these questions:

- Is the ASOS temperature it's using actually the latest reading?
- Is the NWP forecast curve it's simulating against the right one for today?
- Did the Kalman filter actually update when the last ASOS reading came in, or is it stuck on a stale state?
- Are the drift adjustments making things better or worse?
- When the model says 67%, what does the full simulated distribution actually look like?
- Are the model weights changing over time in a sensible direction?

Without answers to those questions you're flying blind. The transparency tab solves all of them.

---

## How to Think About the Visualization Structure

Rather than one giant tab, think of it as a **pipeline view** — you want to show each stage of the calculation in sequence, so you can visually inspect where something might be going wrong. The pipeline is:

```
Raw ASOS Data
      ↓
Kalman Filter State
      ↓
NWP Forecast (which model, which run, blended how)
      ↓
Monte Carlo Inputs (what parameters went in)
      ↓
Monte Carlo Distribution (what came out)
      ↓
Final Probability & Edge Calculation
```

Each stage should be independently inspectable. If the final probability looks wrong, you scan down the pipeline until you find the stage that looks wrong.

---

## What to Show at Each Stage

**Stage 1: Data Freshness Panel**

This is the first thing you check — is the system even receiving live data? Show:

- Last ASOS reading: temperature, timestamp, how many minutes ago, source (NWS or IEM fallback)
- A green/yellow/red staleness indicator: green if under 10 minutes, yellow if 10-20 minutes, red if over 20 minutes
- Last NWP fetch: which models updated successfully, when, and which model run they represent (NWP models have initialization times — a HRRR run from 6 hours ago is less reliable than one from 1 hour ago)
- Last Kalshi market data fetch: bid/ask, timestamp
- A simple table: `Source | Last Updated | Status | Value`

This alone would have caught many of your bugs — stale data silently flowing through a model is one of the most common failure modes.

**Stage 2: Kalman Filter State Inspector**

This is the most important thing to make transparent because the Kalman filter is a black box that accumulates error silently. Show:

- Current state vector: Kalman temperature estimate vs raw ASOS temperature vs NWP forecast — these three numbers should be reasonably close to each other. If Kalman says 52°F and ASOS says 44°F, something broke
- Current bias estimate: this should be a small number (typically -3 to +3°F). If it's drifting to ±10, the filter has diverged
- Covariance matrix values: show the diagonal elements (temperature variance and bias variance). These should be small and stable after the first hour of the day. If they're exploding, the filter is unstable
- A small line chart: Kalman temperature estimate vs raw ASOS readings over the past 3 hours. They should track each other closely with the Kalman line being slightly smoother
- Innovation (residual) over time: the difference between each ASOS reading and the Kalman prediction just before that reading. This should look like white noise centered near zero. If it shows a trend, the model is systematically wrong in a way it's not correcting

**Stage 3: NWP Forecast Inspector**

Show exactly which forecast snapshot the model is currently using:

- For each model (HRRR, GFS, ECMWF): the full 24-hour temperature curve for today, when it was fetched, and the model initialization time
- The blended forecast curve (weighted average of the three)
- The current model weights displayed as a bar chart: HRRR 50% / GFS 30% / ECMWF 20%
- The predicted daily high from each model and the blended prediction
- Critically: show the morning and afternoon drift adjustments being applied and what the effective mu_t is after applying them — this is the actual "attractor" the Monte Carlo is simulating toward

**Stage 4: Monte Carlo Input/Output Inspector**

This is where you verify the simulation is being set up correctly:

Input parameters panel — show every single parameter going into the simulation:
- `current_max_observed` (the hard floor)
- `current_temp_kalman` (starting temperature)
- `kalman_bias` (bias correction being applied)
- `theta` (mean reversion speed)
- `sigma` (volatility)
- `remaining_day_fraction` (how much of the day is left)
- `n_steps` (how many 5-minute steps will be simulated)
- `effective_mu` (the NWP attractor after all corrections)

Output distribution panel:
- A histogram of the 10,000 simulated daily maxima — this is the most powerful visual. You should see a bell-shaped distribution centered somewhere reasonable. If you see a spike at the hard floor value, that means the remaining day fraction is near zero and most paths aren't moving. If you see a bimodal distribution, something is wrong with your drift parameters
- Overlay vertical lines at each Kalshi strike price
- Show the probability mass to the right of each strike as a percentage — these are your fair values
- Show the percentile table: 5th, 25th, 50th, 75th, 95th simulated daily high

**Stage 5: Edge Calculation Transparency**

Show the full math of how you get from probability to trade decision:

A table with columns:
`Strike | Model P(above) | Kalshi Ask | Kalshi Bid | YES Edge | NO Edge | Kelly Fraction | Recommended Contracts | Signal`

For the active strike, show the full Kelly calculation written out:
```
Fair Value: 0.67
Kalshi Ask: 0.60
Edge: 0.07 (above 0.05 threshold)
b = (1/0.60) - 1 = 0.667
Kelly = (0.67 * 0.667 - 0.33) / 0.667 = 0.175
25% Kelly = 0.044
Dollar bet = 0.044 * $50 = $2.18
Contracts = 2
```

Showing the arithmetic explicitly means you can immediately spot if the Kelly fraction looks wrong.

**Stage 6: Historical Calibration Performance**

This is for building trust in the model over time:

- A scatter plot: model predicted probability vs actual outcome (0 or 1) for each past market — this is your calibration curve. A well-calibrated model has points scattered around the diagonal. If all your 60% predictions are coming out wrong, you see it immediately
- Brier score over time as a line chart — should trend downward as calibration improves
- Model weight history: how have HRRR/GFS/ECMWF weights changed over the past 14 days? If HRRR weight is collapsing, the model is learning HRRR has been wrong lately
- Intraday snapshot replay: pick any past date and see a slider that lets you scrub through the day's snapshots, showing what the model predicted at each point in time vs what actually happened

---

## How to Structure This as a Tab

Given Streamlit's layout system, structure the tab with expandable sections rather than trying to show everything at once:

```
Tab 4: Model Transparency & Audit

[Data Freshness Panel]  ← always visible, top of tab, colored indicators

▼ Stage 1: Kalman Filter State          [expandable]
▼ Stage 2: NWP Forecast Snapshot        [expandable]  
▼ Stage 3: Monte Carlo Parameters       [expandable]
▼ Stage 4: Simulated Distribution       [expandable, has histogram]
▼ Stage 5: Edge Calculation Breakdown   [expandable]
▼ Stage 6: Historical Performance       [expandable]
```

Use `st.expander()` for each stage. The data freshness panel at the top is always visible because it's the first thing you check. Everything else is expanded on demand so the tab isn't overwhelming.

---

## The Prompt to Feed Claude Code

Here is a precisely specified prompt you can paste directly into your Claude Code session. It's written to be unambiguous and prevent the "build everything at once" problem by specifying exactly what to do and what not to do:

---

```
I need you to add a new tab to the existing Streamlit app (ui/app.py) 
called "Model Transparency". This tab is an audit trail that lets me 
inspect every stage of the probability calculation pipeline. 

Do not modify any existing tabs. Do not modify any backend modules. 
This tab only reads data — it does not write anything to the database 
or trigger any calculations. All data displayed must be read from the 
PostgreSQL database or by calling existing module functions that already 
exist. Do not create new database tables for this feature.

Build this tab in exactly six stages, using st.expander() for stages 
1 through 6, with a persistent Data Freshness Panel above all expanders.

---

DATA FRESHNESS PANEL (always visible, not in an expander):

Show a row of st.metric() or colored st.status indicators for:
- ASOS: last observation temperature, timestamp, minutes since last 
  update, source (NWS or IEM). Green if under 10 minutes old, yellow 
  if 10-20 minutes, red if over 20 minutes. Query asos_readings table 
  for the most recent record.
- NWP: last fetch time for each model (HRRR/GFS/ECMWF). Green if under 
  2 hours, yellow if 2-6 hours, red if over 6 hours. Query nwp_forecasts 
  table.
- Kalshi: last market data fetch time and current bid/ask. 
- System State: last time system_state was updated for today's date.

Use st.columns(4) for layout. Color the metric labels using 
st.markdown with HTML color styling.

---

STAGE 1 EXPANDER: "Kalman Filter State"

Left column:
- Three st.metric() side by side: 
  "Raw ASOS Temp" | "Kalman Estimate" | "NWP Blended Forecast"
  These three should be close. If Kalman diverges from ASOS by more 
  than 3°F, show the Kalman metric label in red.
- Two st.metric() side by side:
  "Kalman Bias Estimate" (should be between -5 and +5) | 
  "Temp Variance" (diagonal[0] of covariance matrix)
  If bias exceeds ±5, show in red.

Right column:
- Plotly line chart: last 3 hours of data with two traces:
  Trace 1: raw ASOS readings (dots, blue)
  Trace 2: Kalman temperature estimates at each ASOS update time 
  (smooth line, orange)
  These should track each other closely.
- Below chart: Plotly line chart of Kalman innovation 
  (residual = ASOS - Kalman prediction) over same 3 hour window. 
  Should look like noise around zero. Add a horizontal zero line.

Query: asos_readings table for last 3 hours. 
Query: system_state for current Kalman state.

---

STAGE 2 EXPANDER: "NWP Forecast Snapshot"

Show exactly what forecast data the model is currently using.

Top row: st.columns(3) — one column per model (HRRR, GFS, ECMWF):
Each column shows:
- Model name as header
- Fetched at: timestamp
- Predicted daily high: temperature in °F
- Current weight: percentage from model_weights in system_state
- Status: green checkmark if fetched within 2 hours, red X if stale

Main chart: Plotly line chart with:
- One line per model showing full 24-hour temperature curve for 
  today (hourly_temps array from nwp_forecasts table)
- One thicker line for the blended weighted average
- A vertical line at current time
- X-axis: hours 0-24 Eastern time
- Y-axis: temperature °F
- Legend showing model name and weight percentage

Below chart: two st.metric() side by side:
- "Morning Drift Adjustment": value from system_state 
  (morning_drift_adjustment field)
- "Afternoon Drift Adjustment": value from system_state 
  (afternoon_drift_adjustment field)
- "Effective mu_t right now": blended forecast for current hour 
  + kalman_bias + appropriate drift adjustment
  Label this "The Attractor" with a tooltip explanation.

---

STAGE 3 EXPANDER: "Monte Carlo Inputs"

Show a clean table of every parameter going into the next simulation run.
Use st.table() or st.dataframe() with two columns: Parameter | Value.

Parameters to show:
- current_max_observed (hard floor) — from markets table
- current_temp_kalman — from system_state
- kalman_bias_estimate — from system_state  
- theta_decay — from system_state
- sigma_volatility — from system_state
- mu_drift — from system_state
- remaining_day_fraction — compute from config.get_remaining_day_fraction()
- n_steps (= remaining_day_fraction * 12 hours / (5/60)) — show the 
  integer number of 5-minute steps that will be simulated
- n_paths — from SIMULATION_PATHS env variable
- NWP attractor at current hour — from blended forecast

Add a "Run Simulation Now" button. When clicked, call 
MonteCarloEngine.run_simulation() with current parameters and store 
the result in st.session_state so Stage 4 can display it. Show a 
spinner while running.

---

STAGE 4 EXPANDER: "Simulated Distribution"

This stage requires a simulation result. If no simulation has been 
run this session, show a prompt: "Click 'Run Simulation Now' in 
Stage 3 to generate distribution."

If simulation result exists in st.session_state:

Left side:
- Plotly histogram of all simulated daily maximum temperatures
  (reconstruct approximate distribution from the percentile outputs
  using a normal distribution fit if raw paths aren't stored — 
  use p5, p25, p50, p75, p95 to fit and sample a display distribution)
- Overlay vertical dashed lines for each Kalshi strike currently 
  active, labeled with the strike temperature
- Overlay a vertical solid red line at current_max_observed 
  labeled "Hard Floor"
- X-axis: temperature °F, Y-axis: frequency
- Title: "10,000 Simulated Daily Maximum Temperatures"

Right side:
- Percentile table: st.table() showing:
  5th percentile | 25th | 50th (median) | 75th | 95th | mean | std dev
- Probability table: for each active strike show:
  Strike | P(daily max >= strike) | P(daily max < strike)

---

STAGE 5 EXPANDER: "Edge Calculation Breakdown"

Requires both simulation result and current Kalshi market data.

Full edge table using st.dataframe() with conditional formatting:
Columns: Strike | Fair Value | Kalshi Ask | Kalshi Bid | 
YES Edge | NO Edge | Kelly % | Contracts | Signal

Color rows: green background if YES Edge > EDGE_THRESHOLD, 
orange if NO Edge > EDGE_THRESHOLD, white otherwise.

Below table, for the strike with the largest absolute edge, 
show the full Kelly calculation written out step by step using 
st.markdown():

Format it exactly like this:
```
TRADE CALCULATION — Strike: [X]°F
─────────────────────────────────
Fair Value Probability:    0.67
Kalshi Ask:                0.60
Raw Edge:                  +0.07  ✓ Above 0.05 threshold

Kelly Criterion:
  b = (1 / 0.60) - 1     = 0.667
  Kelly = (0.67×0.667 - 0.33) / 0.667 = 0.175 (17.5%)
  25% Fractional Kelly    = 0.044 (4.4%)

Position Sizing:
  Max position size:       $50.00
  Dollar bet:              $2.18
  Price per contract:      $0.60
  Contracts:               2

Signal: BUY YES — 2 contracts at 60¢
```

If DRY_RUN=true, show "DRY RUN MODE — No real orders will be placed" 
in a yellow warning box above the calculation.

---

STAGE 6 EXPANDER: "Historical Calibration Performance"

Query intraday_snapshots and markets tables for last 14 days.

Top row — three summary metrics:
- Average Brier Score (last 14 days)
- Model prediction accuracy (% of days where predicted high was 
  within 2°F of final_official_high)
- Days of calibration data available

Chart 1: Plotly scatter plot — Calibration Curve
- X-axis: model_fair_value_prob at time of snapshot (0 to 1)
- Y-axis: actual outcome (1 if final high exceeded strike, 0 if not)
- Plot each snapshot as a dot
- Add a diagonal reference line (perfect calibration)
- Add a LOWESS smoothed trend line through the dots
- Title: "Model Calibration — Predicted vs Actual"

Chart 2: Plotly line chart — Model Weight History
- Query system_state records for last 14 days
- One line per model showing weight over time
- Should show how HRRR/GFS/ECMWF weights have evolved

Chart 3: Plotly line chart — Intraday Snapshot Replay
- Add a st.date_input() to select a past date
- When date selected, query all snapshots for that date
- Plot model_fair_value_prob and kalshi_implied_prob_yes over 
  the course of that day
- Add a horizontal line at the final actual outcome (0 or 1)
- This lets me see how the model's confidence evolved through 
  the day vs what Kalshi was pricing

---

IMPLEMENTATION REQUIREMENTS:

1. Add this as a new tab in the existing tabs list in app.py. 
   Do not change tab 1, 2, or 3 in any way.

2. All database queries must use the existing db_manager helper 
   functions — do not write raw SQL in app.py.

3. If any data is unavailable (e.g. no simulation has been run, 
   no historical data yet), show a clear informational message 
   rather than an error or blank space.

4. Add a "Refresh Data" button at the top of the tab that calls 
   st.rerun() to reload all database queries.

5. Add st.caption() notes under each chart explaining what a 
   healthy vs unhealthy pattern looks like. For example under 
   the Kalman innovation chart: "Healthy: random noise centered 
   near zero. Warning: trending upward or downward indicates 
   systematic model error."

6. The tab must load without errors even on day one before any 
   historical calibration data exists.

7. Build and test this tab before telling me it is complete. 
   Verify it loads without errors by running the Streamlit app 
   and checking each expander opens correctly.

Do not start building until you confirm you have read and 
understood all six stages. Then build the complete tab in one 
file edit to app.py. Tell me when it is ready to test.
```

---

## One Final Thought

Once this tab is built and you can see the full pipeline, you'll likely find that most of the bugs you encountered were happening at Stage 1 — stale data silently flowing through — or Stage 2 — the wrong NWP forecast snapshot being used. Those are by far the most common failure modes in systems like this and they're invisible without exactly this kind of transparency layer. The histogram in Stage 4 is also extremely revealing — a healthy distribution should look roughly bell-shaped centered a degree or two above the current observed max. If it looks wrong the problem will be obvious visually before you even need to trace the math.