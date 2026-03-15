"""
Database schema management: DDL creation and migration helpers.

All tables are created with ``IF NOT EXISTS`` so this module is safe to
call on every startup. Schema changes in future versions should be applied
as ``ALTER TABLE`` statements appended to ``MIGRATIONS`` rather than
modifying the CREATE statements.
"""

from __future__ import annotations

import structlog

from src.db.connection import get_connection

logger = structlog.get_logger(__name__)

_CREATE_WEATHER_OBSERVATIONS = """
CREATE TABLE IF NOT EXISTS weather_observations (
    id              BIGSERIAL PRIMARY KEY,
    station_id      VARCHAR(10)  NOT NULL,
    observed_at     TIMESTAMPTZ  NOT NULL,
    temp_f          NUMERIC(5,1) NOT NULL,
    dew_point_f     NUMERIC(5,1),
    wind_speed_mph  NUMERIC(6,1),
    wind_dir_deg    NUMERIC(5,1),
    precip_in       NUMERIC(6,2) NOT NULL DEFAULT 0,
    sky_cover       NUMERIC(5,1),
    raw_text        TEXT,
    inserted_at     TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    UNIQUE (station_id, observed_at)
);
CREATE INDEX IF NOT EXISTS idx_obs_station_time
    ON weather_observations (station_id, observed_at DESC);
"""

_CREATE_DAILY_MAX = """
CREATE TABLE IF NOT EXISTS daily_max_observations (
    id                  BIGSERIAL PRIMARY KEY,
    station_id          VARCHAR(10)  NOT NULL,
    date_utc            DATE         NOT NULL,
    max_temp_f          NUMERIC(5,1) NOT NULL,
    observation_count   INT          NOT NULL,
    computed_at         TIMESTAMPTZ  NOT NULL,
    inserted_at         TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    UNIQUE (station_id, date_utc)
);
"""

_CREATE_FORECASTS = """
CREATE TABLE IF NOT EXISTS temperature_forecasts (
    id                      BIGSERIAL PRIMARY KEY,
    station_id              VARCHAR(10)  NOT NULL,
    target_date_utc         DATE         NOT NULL,
    mean_f                  NUMERIC(5,1) NOT NULL,
    std_f                   NUMERIC(5,2) NOT NULL,
    nws_point_forecast_f    NUMERIC(5,1),
    historical_bias_f       NUMERIC(5,2) NOT NULL DEFAULT 0,
    model_version           VARCHAR(20)  NOT NULL DEFAULT 'v1.0',
    observation_count       INT          NOT NULL DEFAULT 0,
    generated_at            TIMESTAMPTZ  NOT NULL,
    inserted_at             TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_fc_station_date
    ON temperature_forecasts (station_id, target_date_utc DESC);
"""

_CREATE_MARKET_SNAPSHOTS = """
CREATE TABLE IF NOT EXISTS market_snapshots (
    id              BIGSERIAL PRIMARY KEY,
    ticker          VARCHAR(100) NOT NULL,
    event_ticker    VARCHAR(100) NOT NULL,
    title           TEXT,
    status          VARCHAR(20)  NOT NULL,
    yes_bid         SMALLINT     NOT NULL DEFAULT 0,
    yes_ask         SMALLINT     NOT NULL DEFAULT 0,
    no_bid          SMALLINT     NOT NULL DEFAULT 0,
    no_ask          SMALLINT     NOT NULL DEFAULT 0,
    last_price      SMALLINT     NOT NULL DEFAULT 0,
    volume          INT          NOT NULL DEFAULT 0,
    open_interest   INT          NOT NULL DEFAULT 0,
    close_time      TIMESTAMPTZ,
    expiration_time TIMESTAMPTZ,
    fetched_at      TIMESTAMPTZ  NOT NULL,
    inserted_at     TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_snap_ticker_time
    ON market_snapshots (ticker, fetched_at DESC);
"""

_CREATE_TRADES = """
CREATE TABLE IF NOT EXISTS trades (
    id                  UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    kalshi_order_id     VARCHAR(100) NOT NULL UNIQUE,
    ticker              VARCHAR(100) NOT NULL,
    side                VARCHAR(5)   NOT NULL,
    action              VARCHAR(5)   NOT NULL,
    contracts           INT          NOT NULL,
    limit_price_cents   SMALLINT     NOT NULL,
    fill_price_cents    SMALLINT,
    status              VARCHAR(20)  NOT NULL DEFAULT 'pending',
    strategy_signal     VARCHAR(100) NOT NULL DEFAULT '',
    forecast_temp_f     NUMERIC(5,1),
    created_at          TIMESTAMPTZ  NOT NULL,
    filled_at           TIMESTAMPTZ,
    inserted_at         TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_trades_ticker ON trades (ticker);
CREATE INDEX IF NOT EXISTS idx_trades_created ON trades (created_at DESC);
"""

_CREATE_SYSTEM_EVENTS = """
CREATE TABLE IF NOT EXISTS system_events (
    id          BIGSERIAL    PRIMARY KEY,
    event_type  VARCHAR(50)  NOT NULL,
    level       VARCHAR(10)  NOT NULL DEFAULT 'info',
    message     TEXT         NOT NULL,
    details     JSONB,
    occurred_at TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_events_type_time
    ON system_events (event_type, occurred_at DESC);
"""

_ALL_DDL = [
    ("weather_observations", _CREATE_WEATHER_OBSERVATIONS),
    ("daily_max_observations", _CREATE_DAILY_MAX),
    ("temperature_forecasts", _CREATE_FORECASTS),
    ("market_snapshots", _CREATE_MARKET_SNAPSHOTS),
    ("trades", _CREATE_TRADES),
    ("system_events", _CREATE_SYSTEM_EVENTS),
]


def init_schema() -> None:
    """Apply all DDL statements to create tables and indexes if they don't exist.

    This is safe to run on every application startup. It will not overwrite
    existing data. Log an error and raise on failure.

    Returns:
        None

    Raises:
        psycopg2.Error: If any DDL statement fails.
    """
    logger.info("db.schema.init.start", table_count=len(_ALL_DDL))
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                for table_name, ddl in _ALL_DDL:
                    logger.debug("db.schema.applying", table=table_name)
                    cur.execute(ddl)
        logger.info("db.schema.init.complete")
    except Exception as exc:
        logger.error("db.schema.init.failed", error=str(exc), exc_info=True)
        raise


def log_system_event(
    event_type: str,
    message: str,
    level: str = "info",
    details: dict | None = None,
) -> None:
    """Insert a structured event into the system_events audit table.

    Args:
        event_type: Short category string (e.g. ``"trade.placed"``).
        message:    Human-readable description.
        level:      Log level string: ``"info"``, ``"warning"``, ``"error"``.
        details:    Optional dict of extra data, stored as JSONB.

    Returns:
        None

    Raises:
        Nothing — errors are logged but not re-raised to avoid disrupting callers.
    """
    import json

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO system_events (event_type, level, message, details)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (event_type, level, message, json.dumps(details) if details else None),
                )
        logger.debug("db.system_event.logged", event_type=event_type)
    except Exception as exc:
        logger.error(
            "db.system_event.log_failed",
            event_type=event_type,
            error=str(exc),
            exc_info=True,
        )
