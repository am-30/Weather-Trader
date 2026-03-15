# Workspace

## Overview

pnpm workspace monorepo (TypeScript/Node.js) + Python trading engine.
Primary workload: automated weather-trading system targeting Kalshi daily maximum temperature markets for KBOS (Boston Logan Airport).

## Stack

### Python Trading Engine
- **Runtime**: Python 3.11
- **Config**: pydantic-settings v2 (all settings type-validated from `.env`)
- **Logging**: structlog (JSON in production, colored console in TTY)
- **HTTP**: httpx with tenacity exponential-backoff retry (max 3 attempts)
- **Scheduler**: APScheduler 3.x (BackgroundScheduler, UTC timezone)
- **Database**: psycopg2-binary + raw SQL (same PostgreSQL instance as Node.js)
- **Data science**: numpy, scipy (Gaussian temperature distribution math)

### Node.js / TypeScript (monorepo infra)
- **Monorepo tool**: pnpm workspaces
- **Node.js version**: 24
- **Package manager**: pnpm
- **TypeScript version**: 5.9
- **API framework**: Express 5
- **Database**: PostgreSQL + Drizzle ORM
- **Validation**: Zod (`zod/v4`), `drizzle-zod`
- **API codegen**: Orval (from OpenAPI spec)
- **Build**: esbuild (CJS bundle)

## Structure

```text
artifacts-monorepo/
├── trading-engine/             # Python trading system (PRIMARY)
│   ├── run.py                  # Workflow entry point
│   ├── requirements.txt        # Python dependencies
│   ├── .env                    # Secrets (gitignored) — copy from .env.example
│   ├── .env.example            # Template with all required variables documented
│   └── src/
│       ├── main.py             # Startup, signal handling, keep-alive loop
│       ├── config.py           # pydantic-settings Settings singleton
│       ├── logging_config.py   # structlog configuration
│       ├── scheduler.py        # APScheduler job registration and lifecycle
│       ├── models/
│       │   ├── weather.py      # WeatherObservation, DailyMaxObservation, NWSGridForecastPeriod
│       │   ├── market.py       # KalshiMarket, OrderBook, OrderRequest, OrderResponse
│       │   ├── position.py     # TradeRecord, Position, SystemState
│       │   └── forecast.py     # ForecastDistribution, TemperatureForecast
│       ├── db/
│       │   ├── connection.py   # ThreadedConnectionPool + get_connection() context manager
│       │   └── schema.py       # DDL (init_schema) + log_system_event()
│       ├── data_feeds/
│       │   └── nws.py          # NWS API client, run_weather_cycle()
│       ├── kalshi/
│       │   └── client.py       # KalshiClient (auth, markets, orders, positions)
│       ├── forecasting/
│       │   └── temperature.py  # generate_forecast(), get_latest_forecast()
│       └── trading/
│           ├── strategy.py     # scan_markets(), evaluate_market(), TradeSignal
│           └── engine.py       # execute_signal(), run_trade_evaluation()
├── artifacts/              # TypeScript deployable applications
│   └── api-server/         # Express API server
├── lib/                    # Shared libraries
│   ├── api-spec/           # OpenAPI spec + Orval codegen config
│   ├── api-client-react/   # Generated React Query hooks
│   ├── api-zod/            # Generated Zod schemas from OpenAPI
│   └── db/                 # Drizzle ORM schema + DB connection
├── scripts/                # Utility scripts (single workspace package)
│   └── src/                # Individual .ts scripts, run via `pnpm --filter @workspace/scripts run <script>`
├── pnpm-workspace.yaml     # pnpm workspace (artifacts/*, lib/*, lib/integrations/*, scripts)
├── tsconfig.base.json      # Shared TS options (composite, bundler resolution, es2022)
├── tsconfig.json           # Root TS project references
└── package.json            # Root package with hoisted devDeps
```

## TypeScript & Composite Projects

Every package extends `tsconfig.base.json` which sets `composite: true`. The root `tsconfig.json` lists all packages as project references. This means:

- **Always typecheck from the root** — run `pnpm run typecheck` (which runs `tsc --build --emitDeclarationOnly`). This builds the full dependency graph so that cross-package imports resolve correctly. Running `tsc` inside a single package will fail if its dependencies haven't been built yet.
- **`emitDeclarationOnly`** — we only emit `.d.ts` files during typecheck; actual JS bundling is handled by esbuild/tsx/vite...etc, not `tsc`.
- **Project references** — when package A depends on package B, A's `tsconfig.json` must list B in its `references` array. `tsc --build` uses this to determine build order and skip up-to-date packages.

## Root Scripts

- `pnpm run build` — runs `typecheck` first, then recursively runs `build` in all packages that define it
- `pnpm run typecheck` — runs `tsc --build --emitDeclarationOnly` using project references

## Packages

### `artifacts/api-server` (`@workspace/api-server`)

Express 5 API server. Routes live in `src/routes/` and use `@workspace/api-zod` for request and response validation and `@workspace/db` for persistence.

- Entry: `src/index.ts` — reads `PORT`, starts Express
- App setup: `src/app.ts` — mounts CORS, JSON/urlencoded parsing, routes at `/api`
- Routes: `src/routes/index.ts` mounts sub-routers; `src/routes/health.ts` exposes `GET /health` (full path: `/api/health`)
- Depends on: `@workspace/db`, `@workspace/api-zod`
- `pnpm --filter @workspace/api-server run dev` — run the dev server
- `pnpm --filter @workspace/api-server run build` — production esbuild bundle (`dist/index.cjs`)
- Build bundles an allowlist of deps (express, cors, pg, drizzle-orm, zod, etc.) and externalizes the rest

### `lib/db` (`@workspace/db`)

Database layer using Drizzle ORM with PostgreSQL. Exports a Drizzle client instance and schema models.

- `src/index.ts` — creates a `Pool` + Drizzle instance, exports schema
- `src/schema/index.ts` — barrel re-export of all models
- `src/schema/<modelname>.ts` — table definitions with `drizzle-zod` insert schemas (no models definitions exist right now)
- `drizzle.config.ts` — Drizzle Kit config (requires `DATABASE_URL`, automatically provided by Replit)
- Exports: `.` (pool, db, schema), `./schema` (schema only)

Production migrations are handled by Replit when publishing. In development, we just use `pnpm --filter @workspace/db run push`, and we fallback to `pnpm --filter @workspace/db run push-force`.

### `lib/api-spec` (`@workspace/api-spec`)

Owns the OpenAPI 3.1 spec (`openapi.yaml`) and the Orval config (`orval.config.ts`). Running codegen produces output into two sibling packages:

1. `lib/api-client-react/src/generated/` — React Query hooks + fetch client
2. `lib/api-zod/src/generated/` — Zod schemas

Run codegen: `pnpm --filter @workspace/api-spec run codegen`

### `lib/api-zod` (`@workspace/api-zod`)

Generated Zod schemas from the OpenAPI spec (e.g. `HealthCheckResponse`). Used by `api-server` for response validation.

### `lib/api-client-react` (`@workspace/api-client-react`)

Generated React Query hooks and fetch client from the OpenAPI spec (e.g. `useHealthCheck`, `healthCheck`).

### `scripts` (`@workspace/scripts`)

Utility scripts package. Each script is a `.ts` file in `src/` with a corresponding npm script in `package.json`. Run scripts via `pnpm --filter @workspace/scripts run <script>`. Scripts can import any workspace package (e.g., `@workspace/db`) by adding it as a dependency in `scripts/package.json`.

## Trading Engine — Operations

### First-time setup
1. Open `trading-engine/.env`
2. Replace `REPLACE_WITH_YOUR_KALSHI_API_KEY` with your real Kalshi API key
3. Set `KALSHI_ENV=demo` (paper trading) or `KALSHI_ENV=prod` (live)
4. Start the **Trading Engine** workflow in Replit

### Workflow
- **Trading Engine** — runs `python trading-engine/run.py` as a console process
- The engine initialises the DB schema on startup (idempotent, safe to restart)
- Scheduler fires four periodic jobs: weather fetch, market snapshot, forecast, trade eval

### Database tables
| Table                    | Purpose                                       |
|--------------------------|-----------------------------------------------|
| `weather_observations`   | Hourly KBOS readings from NWS                |
| `daily_max_observations` | Computed daily maximum temperatures           |
| `temperature_forecasts`  | Probabilistic Gaussian forecasts              |
| `market_snapshots`       | Kalshi market quote history                  |
| `trades`                 | Every submitted order and its status          |
| `system_events`          | Audit log for system-level events             |

### Key design rules (non-negotiable)
- All API calls use tenacity retry: `stop_after_attempt(3)`, `wait_exponential`
- All DB writes are wrapped in `try/except` and logged via structlog
- All timestamps stored as UTC `TIMESTAMPTZ`, converted to ET only at display time
- All temperatures stored as `NUMERIC(5,1)` Fahrenheit
- No secrets are hardcoded — all come from `.env` via pydantic-settings
- `print()` is never used; every log line goes through `structlog.get_logger()`

### Adjusting trading aggression
Edit `trading-engine/.env`:
- Raise `MIN_EDGE_CENTS` to trade only on high-conviction edges
- Lower `MAX_TRADE_SIZE_USD` to reduce position sizing
- Set `MAX_CONTRACTS_PER_MARKET=1` to limit exposure per market
