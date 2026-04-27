"""
SQLAlchemy 2.0 database layer for the Kalshi weather trading system.

Provides:
- ORM table definitions (DeclarativeBase)
- Thread-safe scoped_session factory
- ``init_schema()`` — creates all tables on startup
- CRUD helpers for every table
- ``update_hard_floor()`` using PostgreSQL GREATEST() for atomic updates

Never import from ``ingestion/``, ``quant/``, or ``execution/`` here.
Only Layer 1 (schemas) is imported.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import pytz
import structlog
from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Integer,
    JSON,
    Numeric,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
    update,
)
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import DeclarativeBase, scoped_session, sessionmaker

from kalshi_weather_trader.config.settings import settings
from kalshi_weather_trader.db.schemas import (
    ASOSReadingDocument,
    IntradaySnapshotDocument,
    MarketDocument,
    NWPForecastDocument,
    PaperTradeDocument,
    SystemStateDocument,
    TradeLogDocument,
)

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Engine + session factory
# ---------------------------------------------------------------------------

_engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,          # detect stale connections
    pool_size=5,
    max_overflow=10,
    echo=False,
)

_session_factory = sessionmaker(bind=_engine, expire_on_commit=False)
Session = scoped_session(_session_factory)  # thread-local sessions for APScheduler


# ---------------------------------------------------------------------------
# ORM model definitions
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""
    pass


class MarketORM(Base):
    """ORM mapping for the ``markets`` table."""

    __tablename__ = "markets"

    target_date = Column(Date, primary_key=True)
    current_max_observed = Column(Numeric(5, 1), nullable=True, default=None)
    market_status = Column(String(20), nullable=False, default="open")
    auto_trade_enabled = Column(Boolean, nullable=False, default=True)
    final_official_high = Column(Numeric(5, 1), nullable=True)
    cli_settlement_confirmed = Column(Boolean, nullable=False, default=False)
    last_updated_utc = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


class ASOSReadingORM(Base):
    """ORM mapping for the ``asos_readings`` table."""

    __tablename__ = "asos_readings"
    __table_args__ = (
        UniqueConstraint("station_id", "observation_time_utc", name="uq_asos_station_time"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    station_id = Column(String(10), nullable=False, default="KBOS")
    observation_time_utc = Column(DateTime(timezone=True), nullable=False)
    temperature_f = Column(Numeric(5, 1), nullable=False)
    dew_point_f = Column(Numeric(5, 1), nullable=True)
    wind_speed_mph = Column(Numeric(6, 1), nullable=True)
    raw_metar = Column(Text, nullable=True)
    inserted_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


class NWPForecastORM(Base):
    """ORM mapping for the ``nwp_forecasts`` table."""

    __tablename__ = "nwp_forecasts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    target_date = Column(Date, nullable=False)
    model_name = Column(String(10), nullable=False)
    fetched_at_utc = Column(DateTime(timezone=True), nullable=False)
    hourly_temps = Column(JSON, nullable=False)
    predicted_daily_high = Column(Numeric(5, 1), nullable=False)
    mean_cloudcover_10_16 = Column(Numeric(5, 2), nullable=True)
    ensemble_highs = Column(JSON, nullable=True)
    ensemble_spread = Column(Numeric(5, 2), nullable=True)


class SystemStateORM(Base):
    """ORM mapping for the ``system_state`` table."""

    __tablename__ = "system_state"

    target_date = Column(Date, primary_key=True)
    kalman_temp_estimate = Column(Numeric(6, 2), nullable=False)
    kalman_bias_estimate = Column(Numeric(6, 2), nullable=False, default=0.0)
    kalman_covariance = Column(JSON, nullable=False)
    model_weights = Column(JSON, nullable=False)
    mu_drift = Column(Numeric(6, 3), nullable=False, default=0.0)
    theta_decay = Column(Numeric(7, 4), nullable=False, default=0.1)
    sigma_volatility = Column(Numeric(6, 3), nullable=False, default=2.0)
    morning_drift_adjustment = Column(Numeric(6, 3), nullable=False, default=0.0)
    afternoon_drift_adjustment = Column(Numeric(6, 3), nullable=False, default=0.0)
    persistence_filter_offset = Column(Numeric(4, 2), nullable=True)
    sigma_by_block = Column(JSON, nullable=True)
    theta_am = Column(Numeric(7, 4), nullable=True)
    theta_pm = Column(Numeric(7, 4), nullable=True)
    ou_max_stationary_std_calibrated = Column(Numeric(5, 3), nullable=True)
    nwp_rmse_n_dates = Column(Integer, nullable=True)
    kalman_bias_decay_calibrated = Column(Numeric(5, 4), nullable=True)
    nwp_daily_max_bias = Column(Numeric(6, 3), nullable=False, server_default="0.0")
    last_calibrated_utc = Column(DateTime(timezone=True), nullable=True)
    last_updated_utc = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


class IntradaySnapshotORM(Base):
    """ORM mapping for the ``intraday_snapshots`` table."""

    __tablename__ = "intraday_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    target_date = Column(Date, nullable=False)
    snapshot_time_utc = Column(DateTime(timezone=True), nullable=False)
    snapshot_time_eastern = Column(String(5), nullable=False)
    current_asos_temp_f = Column(Numeric(5, 1), nullable=False)
    current_max_observed_f = Column(Numeric(5, 1), nullable=False)
    hrrr_predicted_high = Column(Numeric(5, 1), nullable=True)
    gfs_predicted_high = Column(Numeric(5, 1), nullable=True)
    ecmwf_predicted_high = Column(Numeric(5, 1), nullable=True)
    blended_predicted_high = Column(Numeric(5, 1), nullable=False)
    kalman_temp_estimate = Column(Numeric(6, 2), nullable=False)
    kalman_bias_estimate = Column(Numeric(6, 2), nullable=False)
    kalshi_implied_prob_yes = Column(Numeric(6, 4), nullable=True)
    kalshi_bid = Column(Numeric(6, 4), nullable=True)
    kalshi_ask = Column(Numeric(6, 4), nullable=True)
    kalshi_strike = Column(Numeric(5, 1), nullable=True)
    model_fair_value_prob = Column(Numeric(6, 4), nullable=True)
    model_edge = Column(Numeric(6, 4), nullable=True)
    is_forced = Column(Boolean, nullable=False, default=False)
    inserted_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


class HistoricalDailyHighORM(Base):
    """ORM mapping for the ``historical_daily_highs`` table (climatology baseline)."""

    __tablename__ = "historical_daily_highs"

    station_id = Column(String(10), primary_key=True, nullable=False)
    obs_date = Column(Date, primary_key=True, nullable=False)
    high_f = Column(Numeric(5, 1), nullable=False)
    source = Column(String(20), nullable=False, default="IEM")


class TradeLogORM(Base):
    """ORM mapping for the ``trade_logs`` table."""

    __tablename__ = "trade_logs"

    trade_id = Column(String(36), primary_key=True)
    target_date = Column(Date, nullable=False)
    executed_at_utc = Column(DateTime(timezone=True), nullable=False)
    market_ticker = Column(String(100), nullable=False)
    action = Column(String(10), nullable=False)
    kalshi_strike = Column(Numeric(5, 1), nullable=False)
    contracts = Column(SmallInteger, nullable=False)
    price_cents = Column(SmallInteger, nullable=False)
    fair_value_prob = Column(Numeric(6, 4), nullable=False)
    kalshi_implied_prob = Column(Numeric(6, 4), nullable=False)
    edge_at_execution = Column(Numeric(6, 4), nullable=False)
    kelly_fraction = Column(Numeric(8, 6), nullable=True)
    dry_run = Column(Boolean, nullable=False, default=True)
    order_id = Column(String(100), nullable=True)
    status = Column(String(20), nullable=False, default="pending")
    notes = Column(Text, nullable=True)
    inserted_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


class PaperTradeORM(Base):
    """ORM mapping for the ``paper_trade_positions`` table."""

    __tablename__ = "paper_trade_positions"

    position_id = Column(String(36), primary_key=True)
    target_date = Column(Date, nullable=False, index=True)
    market_ticker = Column(String(100), nullable=False)
    action = Column(String(10), nullable=False)
    kalshi_strike = Column(Numeric(5, 1), nullable=False)
    entry_at_utc = Column(DateTime(timezone=True), nullable=False)
    entry_price_cents = Column(SmallInteger, nullable=False)
    contracts = Column(SmallInteger, nullable=False)
    cost_usd = Column(Numeric(8, 2), nullable=False)
    fair_value_prob = Column(Numeric(6, 4), nullable=False)
    edge_at_entry = Column(Numeric(6, 4), nullable=False)
    kelly_fraction = Column(Numeric(8, 6), nullable=True)
    budget_mode = Column(String(10), nullable=False)
    bankroll_at_entry = Column(Numeric(10, 2), nullable=True)
    status = Column(String(25), nullable=False, default="open")
    exit_at_utc = Column(DateTime(timezone=True), nullable=True)
    exit_price_cents = Column(SmallInteger, nullable=True)
    pnl_cents = Column(Numeric(8, 2), nullable=True)
    pnl_usd = Column(Numeric(8, 4), nullable=True)
    official_high_f = Column(Numeric(5, 1), nullable=True)
    settlement_win = Column(Boolean, nullable=True)
    inserted_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------


def _migrate_paper_trade_positions() -> None:
    """Create paper_trade_positions table if absent (idempotent).

    Uses CREATE TABLE IF NOT EXISTS so safe to run on every startup.

    Args:
        None

    Returns:
        None

    Raises:
        Nothing — errors are logged.
    """
    log = structlog.get_logger()
    try:
        with _engine.begin() as conn:
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS paper_trade_positions ("
                "  position_id VARCHAR(36) PRIMARY KEY,"
                "  target_date DATE NOT NULL,"
                "  market_ticker VARCHAR(100) NOT NULL,"
                "  action VARCHAR(10) NOT NULL,"
                "  kalshi_strike NUMERIC(5,1) NOT NULL,"
                "  entry_at_utc TIMESTAMPTZ NOT NULL,"
                "  entry_price_cents SMALLINT NOT NULL,"
                "  contracts SMALLINT NOT NULL,"
                "  cost_usd NUMERIC(8,2) NOT NULL,"
                "  fair_value_prob NUMERIC(6,4) NOT NULL,"
                "  edge_at_entry NUMERIC(6,4) NOT NULL,"
                "  kelly_fraction NUMERIC(8,6),"
                "  budget_mode VARCHAR(10) NOT NULL,"
                "  bankroll_at_entry NUMERIC(10,2),"
                "  status VARCHAR(25) NOT NULL DEFAULT 'open',"
                "  exit_at_utc TIMESTAMPTZ,"
                "  exit_price_cents SMALLINT,"
                "  pnl_cents NUMERIC(8,2),"
                "  pnl_usd NUMERIC(8,4),"
                "  official_high_f NUMERIC(5,1),"
                "  settlement_win BOOLEAN,"
                "  inserted_at TIMESTAMPTZ NOT NULL DEFAULT NOW()"
                ")"
            ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_paper_trade_date "
                "ON paper_trade_positions(target_date, entry_at_utc DESC)"
            ))
        log.info("db.migration.paper_trade_positions.done")
    except Exception as e:
        log.warning("db.migration.paper_trade_positions.failed", error=str(e))


def _migrate_kalshi_strike_columns() -> None:
    """Migrate kalshi_strike columns from SmallInteger to NUMERIC(5,1).

    Safe to run on every startup — ALTER TYPE is idempotent via USING cast.
    If the table does not yet exist this is a no-op.

    Args:
        None

    Returns:
        None

    Raises:
        Nothing — all errors are caught and logged.
    """
    log = structlog.get_logger()
    stmts = [
        "ALTER TABLE intraday_snapshots ALTER COLUMN kalshi_strike TYPE NUMERIC(5,1) USING kalshi_strike::NUMERIC(5,1)",
        "ALTER TABLE trade_logs ALTER COLUMN kalshi_strike TYPE NUMERIC(5,1) USING kalshi_strike::NUMERIC(5,1)",
    ]
    try:
        with _engine.begin() as conn:
            for stmt in stmts:
                try:
                    conn.execute(text(stmt))
                    log.info("db.migration.kalshi_strike.applied", stmt=stmt)
                except Exception as col_err:
                    # Column may not exist yet (first run) or already correct type
                    log.debug("db.migration.kalshi_strike.skipped", stmt=stmt, reason=str(col_err))
    except Exception as e:
        log.warning("db.migration.kalshi_strike.failed", error=str(e))


def _migrate_add_cli_confirmed() -> None:
    """Add cli_settlement_confirmed column to markets table if absent.

    Idempotent — uses IF NOT EXISTS so safe to run on every startup.

    Args:
        None

    Returns:
        None

    Raises:
        Nothing — all errors are caught and logged.
    """
    try:
        with _engine.begin() as conn:
            try:
                conn.execute(text(
                    "ALTER TABLE markets ADD COLUMN IF NOT EXISTS "
                    "cli_settlement_confirmed BOOLEAN NOT NULL DEFAULT FALSE"
                ))
                logger.info("db.migration.cli_confirmed.applied")
            except Exception as col_err:
                logger.debug("db.migration.cli_confirmed.skipped", reason=str(col_err))
    except Exception as e:
        logger.warning("db.migration.cli_confirmed.failed", error=str(e))


def _migrate_system_state_phase1_columns() -> None:
    """Add persistence_filter_offset and sigma_by_block columns to system_state if absent.

    Idempotent — uses IF NOT EXISTS so safe to run on every startup.

    Args:
        None

    Returns:
        None

    Raises:
        Nothing — all errors are caught and logged.
    """
    try:
        with _engine.begin() as conn:
            for stmt in [
                "ALTER TABLE system_state ADD COLUMN IF NOT EXISTS "
                "persistence_filter_offset NUMERIC(4,2) DEFAULT 0.3",
                "ALTER TABLE system_state ADD COLUMN IF NOT EXISTS "
                "sigma_by_block JSONB",
            ]:
                try:
                    conn.execute(text(stmt))
                    logger.info("db.migration.phase1_columns.applied", stmt=stmt[:60])
                except Exception as col_err:
                    logger.debug("db.migration.phase1_columns.skipped", reason=str(col_err))
    except Exception as e:
        logger.warning("db.migration.phase1_columns.failed", error=str(e))


def _migrate_null_hard_floor() -> None:
    """Replace -999 sentinel hard-floor values with NULL.

    Idempotent — safe to run on every startup.

    Actions:
      1. Make markets.current_max_observed nullable (was NOT NULL with default -999).
      2. NULL-out any existing -999 rows in markets.
      3. Delete intraday_snapshot rows where current_max_observed_f = -999
         (these are pre-first-ASOS snapshots with no valid floor; all other
         columns in those rows are also meaningless at that point).

    Args:
        None

    Returns:
        None

    Raises:
        Nothing — all errors are caught and logged.
    """
    try:
        with _engine.begin() as conn:
            for stmt in [
                "ALTER TABLE markets ALTER COLUMN current_max_observed DROP NOT NULL",
                "UPDATE markets SET current_max_observed = NULL "
                "WHERE current_max_observed = -999",
                "DELETE FROM intraday_snapshots WHERE current_max_observed_f = -999",
            ]:
                try:
                    result = conn.execute(text(stmt))
                    rows = getattr(result, "rowcount", None)
                    logger.info(
                        "db.migration.null_hard_floor.applied",
                        stmt=stmt[:70],
                        rows_affected=rows,
                    )
                except Exception as col_err:
                    # ALTER TABLE will raise if column is already nullable — that's fine.
                    logger.debug(
                        "db.migration.null_hard_floor.skipped", reason=str(col_err)
                    )
    except Exception as e:
        logger.warning("db.migration.null_hard_floor.failed", error=str(e))


def _migrate_system_state_phase2_columns() -> None:
    """Add theta_am and theta_pm columns to system_state if absent.

    Idempotent — uses IF NOT EXISTS so safe to run on every startup.

    Args:
        None

    Returns:
        None

    Raises:
        Nothing — all errors are caught and logged.
    """
    try:
        with _engine.begin() as conn:
            for stmt in [
                "ALTER TABLE system_state ADD COLUMN IF NOT EXISTS theta_am NUMERIC(7,4)",
                "ALTER TABLE system_state ADD COLUMN IF NOT EXISTS theta_pm NUMERIC(7,4)",
            ]:
                try:
                    conn.execute(text(stmt))
                    logger.info("db.migration.phase2_columns.applied", stmt=stmt[:60])
                except Exception as col_err:
                    logger.debug("db.migration.phase2_columns.skipped", reason=str(col_err))
    except Exception as e:
        logger.warning("db.migration.phase2_columns.failed", error=str(e))


def _migrate_system_state_phase3_columns() -> None:
    """Add Phase 3 NWP-RMSE sigma-cap calibration columns to system_state if absent.

    Idempotent — uses IF NOT EXISTS so safe to run on every startup.

    Args:
        None

    Returns:
        None

    Raises:
        Nothing — all errors are caught and logged.
    """
    try:
        with _engine.begin() as conn:
            for stmt in [
                "ALTER TABLE system_state ADD COLUMN IF NOT EXISTS "
                "ou_max_stationary_std_calibrated NUMERIC(5,3)",
                "ALTER TABLE system_state ADD COLUMN IF NOT EXISTS "
                "nwp_rmse_n_dates INTEGER",
            ]:
                try:
                    conn.execute(text(stmt))
                    logger.info("db.migration.phase3_columns.applied", stmt=stmt[:70])
                except Exception as col_err:
                    logger.debug("db.migration.phase3_columns.skipped", reason=str(col_err))
    except Exception as e:
        logger.warning("db.migration.phase3_columns.failed", error=str(e))


def _migrate_system_state_phase_c_columns() -> None:
    """Add kalman_bias_decay_calibrated column to system_state if absent.

    Idempotent — uses IF NOT EXISTS so safe to run on every startup.
    """
    try:
        with _engine.begin() as conn:
            try:
                conn.execute(text(
                    "ALTER TABLE system_state ADD COLUMN IF NOT EXISTS "
                    "kalman_bias_decay_calibrated NUMERIC(5,4)"
                ))
                logger.info("db.migration.phase_c_columns.applied")
            except Exception as col_err:
                logger.debug("db.migration.phase_c_columns.skipped", reason=str(col_err))
    except Exception as e:
        logger.warning("db.migration.phase_c_columns.failed", error=str(e))


def _migrate_system_state_phase_d_columns() -> None:
    """Add nwp_daily_max_bias column to system_state if absent (D3 fix).

    Idempotent — uses IF NOT EXISTS so safe to run on every startup.
    """
    try:
        with _engine.begin() as conn:
            try:
                conn.execute(text(
                    "ALTER TABLE system_state ADD COLUMN IF NOT EXISTS "
                    "nwp_daily_max_bias NUMERIC(6,3) NOT NULL DEFAULT 0.0"
                ))
                logger.info("db.migration.phase_d_columns.applied")
            except Exception as col_err:
                logger.debug("db.migration.phase_d_columns.skipped", reason=str(col_err))
    except Exception as e:
        logger.warning("db.migration.phase_d_columns.failed", error=str(e))


def _migrate_nwp_forecasts_phase3_columns() -> None:
    """Add cloud cover and ensemble columns to nwp_forecasts; create historical_daily_highs table.

    Idempotent — uses IF NOT EXISTS so safe to run on every startup.

    Args:
        None

    Returns:
        None

    Raises:
        Nothing — all errors are caught and logged.
    """
    stmts = [
        "ALTER TABLE nwp_forecasts ADD COLUMN IF NOT EXISTS mean_cloudcover_10_16 NUMERIC(5,2)",
        "ALTER TABLE nwp_forecasts ADD COLUMN IF NOT EXISTS ensemble_highs JSONB",
        "ALTER TABLE nwp_forecasts ADD COLUMN IF NOT EXISTS ensemble_spread NUMERIC(5,2)",
        (
            "CREATE TABLE IF NOT EXISTS historical_daily_highs ("
            "station_id VARCHAR(10) NOT NULL, "
            "obs_date DATE NOT NULL, "
            "high_f NUMERIC(5,1) NOT NULL, "
            "source VARCHAR(20) NOT NULL DEFAULT 'IEM', "
            "PRIMARY KEY (station_id, obs_date))"
        ),
    ]
    try:
        with _engine.begin() as conn:
            for stmt in stmts:
                try:
                    conn.execute(text(stmt))
                    logger.info("db.migration.nwp_phase3.applied", stmt=stmt[:70])
                except Exception as col_err:
                    logger.debug("db.migration.nwp_phase3.skipped", reason=str(col_err))
    except Exception as e:
        logger.warning("db.migration.nwp_phase3.failed", error=str(e))


def _ensure_indexes() -> None:
    """Create performance indexes on high-frequency query columns if absent.

    Idempotent — uses ``CREATE INDEX IF NOT EXISTS``.  Called from
    ``init_schema()`` on every startup so indexes are present even if the DB
    was created before this function was added.

    Args:
        None

    Returns:
        None

    Raises:
        Nothing — errors are logged.
    """
    stmts = [
        # ASOS: every 5-min fetch queries by station_id + time range
        "CREATE INDEX IF NOT EXISTS idx_asos_station_time "
        "ON asos_readings(station_id, observation_time_utc DESC)",
        # NWP: calibration loops 14 days × 3 models, filtered by date+model+fetch time
        "CREATE INDEX IF NOT EXISTS idx_nwp_target_model_fetch "
        "ON nwp_forecasts(target_date, model_name, fetched_at_utc DESC)",
        # Snapshots: every 2-hr insert + 7-day calibration scan
        "CREATE INDEX IF NOT EXISTS idx_snapshot_date_time "
        "ON intraday_snapshots(target_date, snapshot_time_utc DESC)",
        # Trade logs: UI dashboard history query
        "CREATE INDEX IF NOT EXISTS idx_tradelog_date_time "
        "ON trade_logs(target_date, executed_at_utc DESC)",
    ]
    try:
        with _engine.begin() as conn:
            for stmt in stmts:
                conn.execute(text(stmt))
        logger.info("db.indexes.ensured")
    except Exception as exc:
        logger.warning("db.indexes.failed", error=str(exc))


def init_schema() -> None:
    """Create all tables if they do not already exist.

    Safe to call multiple times — uses ``checkfirst=True``.

    Args:
        None

    Returns:
        None

    Raises:
        sqlalchemy.exc.SQLAlchemyError: If table creation fails.
    """
    _migrate_kalshi_strike_columns()
    _migrate_add_cli_confirmed()
    _migrate_system_state_phase1_columns()
    _migrate_system_state_phase2_columns()
    _migrate_system_state_phase3_columns()
    _migrate_system_state_phase_c_columns()
    _migrate_system_state_phase_d_columns()
    _migrate_nwp_forecasts_phase3_columns()
    _migrate_null_hard_floor()
    _migrate_paper_trade_positions()
    _ensure_indexes()
    try:
        Base.metadata.create_all(_engine, checkfirst=True)
        logger.info("db.init_schema.done", tables=list(Base.metadata.tables.keys()))
    except Exception as exc:
        logger.error("db.init_schema.failed", error=str(exc), exc_info=True)
        raise


# ---------------------------------------------------------------------------
# Markets CRUD
# ---------------------------------------------------------------------------


def get_market(target_date: date) -> Optional[MarketDocument]:
    """Fetch a market row for the given target date.

    Args:
        target_date: The calendar date to look up.

    Returns:
        ``MarketDocument`` if the row exists, else None.

    Raises:
        sqlalchemy.exc.SQLAlchemyError: On database error.
    """
    session = Session()
    try:
        row = session.get(MarketORM, target_date)
        if row is None:
            return None
        return MarketDocument(
            target_date=row.target_date,
            current_max_observed=(float(row.current_max_observed) if row.current_max_observed is not None else None),
            market_status=row.market_status,
            auto_trade_enabled=row.auto_trade_enabled,
            final_official_high=(
                float(row.final_official_high) if row.final_official_high is not None else None
            ),
            cli_settlement_confirmed=bool(row.cli_settlement_confirmed),
            last_updated_utc=row.last_updated_utc,
        )
    except Exception as exc:
        logger.error("db.get_market.failed", target_date=str(target_date), error=str(exc))
        raise
    finally:
        Session.remove()


def upsert_market(doc: MarketDocument) -> None:
    """Insert or update a market row.

    Args:
        doc: ``MarketDocument`` to persist.

    Returns:
        None

    Raises:
        sqlalchemy.exc.SQLAlchemyError: On database error.
    """
    session = Session()
    try:
        stmt = pg_insert(MarketORM).values(
            target_date=doc.target_date,
            current_max_observed=doc.current_max_observed,
            market_status=doc.market_status,
            auto_trade_enabled=doc.auto_trade_enabled,
            final_official_high=doc.final_official_high,
            cli_settlement_confirmed=doc.cli_settlement_confirmed,
            last_updated_utc=doc.last_updated_utc,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["target_date"],
            set_={
                "current_max_observed": stmt.excluded.current_max_observed,
                "market_status": stmt.excluded.market_status,
                "auto_trade_enabled": stmt.excluded.auto_trade_enabled,
                "final_official_high": stmt.excluded.final_official_high,
                "cli_settlement_confirmed": stmt.excluded.cli_settlement_confirmed,
                "last_updated_utc": stmt.excluded.last_updated_utc,
            },
        )
        session.execute(stmt)
        session.commit()
        logger.info("db.upsert_market.done", target_date=str(doc.target_date))
    except Exception as exc:
        session.rollback()
        logger.error("db.upsert_market.failed", target_date=str(doc.target_date), error=str(exc))
        raise
    finally:
        Session.remove()


def update_hard_floor(target_date: date, new_temp: float) -> float:
    """Atomically update current_max_observed using PostgreSQL GREATEST().

    Only increases the stored value — never decreases it.

    Args:
        target_date: Trading date to update.
        new_temp:    Latest ASOS temperature reading.

    Returns:
        The new stored current_max_observed value after the update.

    Raises:
        sqlalchemy.exc.SQLAlchemyError: On database error.
    """
    session = Session()
    try:
        # Ensure the market row exists first
        existing = session.get(MarketORM, target_date)
        if existing is None:
            # Bootstrap a new market row
            session.add(
                MarketORM(
                    target_date=target_date,
                    current_max_observed=new_temp,
                    last_updated_utc=datetime.now(timezone.utc),
                )
            )
            session.commit()
            logger.info(
                "db.update_hard_floor.bootstrapped",
                target_date=str(target_date),
                value=new_temp,
            )
            return new_temp

        stmt = (
            update(MarketORM)
            .where(MarketORM.target_date == target_date)
            .values(
                current_max_observed=func.greatest(
                    MarketORM.current_max_observed, new_temp
                ),
                last_updated_utc=datetime.now(timezone.utc),
            )
            .returning(MarketORM.current_max_observed)
        )
        result = session.execute(stmt)
        session.commit()
        row = result.fetchone()
        new_value = float(row[0]) if row else new_temp
        logger.debug(
            "db.update_hard_floor.done",
            target_date=str(target_date),
            new_temp=new_temp,
            stored=new_value,
        )
        return new_value
    except Exception as exc:
        session.rollback()
        logger.error(
            "db.update_hard_floor.failed",
            target_date=str(target_date),
            new_temp=new_temp,
            error=str(exc),
        )
        raise
    finally:
        Session.remove()


def set_kill_switch(target_date: date, enabled: bool) -> None:
    """Set the auto_trade_enabled kill switch for a market.

    Args:
        target_date: Trading date to update.
        enabled:     True to enable trading, False to halt.

    Returns:
        None

    Raises:
        sqlalchemy.exc.SQLAlchemyError: On database error.
    """
    session = Session()
    try:
        stmt = (
            update(MarketORM)
            .where(MarketORM.target_date == target_date)
            .values(
                auto_trade_enabled=enabled,
                last_updated_utc=datetime.now(timezone.utc),
            )
        )
        session.execute(stmt)
        session.commit()
        logger.info(
            "db.set_kill_switch",
            target_date=str(target_date),
            auto_trade_enabled=enabled,
        )
    except Exception as exc:
        session.rollback()
        logger.error(
            "db.set_kill_switch.failed",
            target_date=str(target_date),
            error=str(exc),
        )
        raise
    finally:
        Session.remove()


# ---------------------------------------------------------------------------
# ASOS readings CRUD
# ---------------------------------------------------------------------------


def upsert_asos_reading(doc: ASOSReadingDocument) -> None:
    """Insert an ASOS reading, ignoring duplicates (same station + time).

    Args:
        doc: ``ASOSReadingDocument`` to persist.

    Returns:
        None

    Raises:
        sqlalchemy.exc.SQLAlchemyError: On database error.
    """
    session = Session()
    try:
        stmt = pg_insert(ASOSReadingORM).values(
            station_id=doc.station_id,
            observation_time_utc=doc.observation_time_utc,
            temperature_f=doc.temperature_f,
            dew_point_f=doc.dew_point_f,
            wind_speed_mph=doc.wind_speed_mph,
            raw_metar=doc.raw_metar,
            inserted_at=doc.inserted_at,
        )
        stmt = stmt.on_conflict_do_nothing(
            constraint="uq_asos_station_time"
        )
        session.execute(stmt)
        session.commit()
        logger.debug(
            "db.upsert_asos_reading.done",
            station=doc.station_id,
            time=str(doc.observation_time_utc),
            temp_f=doc.temperature_f,
        )
    except Exception as exc:
        session.rollback()
        logger.error(
            "db.upsert_asos_reading.failed",
            station=doc.station_id,
            time=str(doc.observation_time_utc),
            error=str(exc),
        )
        raise
    finally:
        Session.remove()


def get_latest_asos_reading(station_id: str = "KBOS") -> Optional[ASOSReadingDocument]:
    """Fetch the most recent ASOS reading for the given station.

    Args:
        station_id: ICAO station code. Defaults to 'KBOS'.

    Returns:
        Most recent ``ASOSReadingDocument``, or None if no readings exist.

    Raises:
        sqlalchemy.exc.SQLAlchemyError: On database error.
    """
    session = Session()
    try:
        row = (
            session.query(ASOSReadingORM)
            .filter(ASOSReadingORM.station_id == station_id)
            .order_by(ASOSReadingORM.observation_time_utc.desc())
            .first()
        )
        if row is None:
            return None
        return ASOSReadingDocument(
            station_id=row.station_id,
            observation_time_utc=row.observation_time_utc,
            temperature_f=float(row.temperature_f),
            dew_point_f=float(row.dew_point_f) if row.dew_point_f is not None else None,
            wind_speed_mph=float(row.wind_speed_mph) if row.wind_speed_mph is not None else None,
            raw_metar=row.raw_metar,
        )
    except Exception as exc:
        logger.error("db.get_latest_asos_reading.failed", error=str(exc))
        raise
    finally:
        Session.remove()


def get_earliest_asos_reading(station_id: str = "KBOS") -> Optional[ASOSReadingDocument]:
    """Fetch the oldest ASOS reading for the given station.

    Args:
        station_id: ICAO station code. Defaults to 'KBOS'.

    Returns:
        Oldest ``ASOSReadingDocument``, or None if no readings exist.

    Raises:
        sqlalchemy.exc.SQLAlchemyError: On database error.
    """
    session = Session()
    try:
        row = (
            session.query(ASOSReadingORM)
            .filter(ASOSReadingORM.station_id == station_id)
            .order_by(ASOSReadingORM.observation_time_utc.asc())
            .first()
        )
        if row is None:
            return None
        return ASOSReadingDocument(
            station_id=row.station_id,
            observation_time_utc=row.observation_time_utc,
            temperature_f=float(row.temperature_f),
            dew_point_f=float(row.dew_point_f) if row.dew_point_f is not None else None,
            wind_speed_mph=float(row.wind_speed_mph) if row.wind_speed_mph is not None else None,
            raw_metar=row.raw_metar,
        )
    except Exception as exc:
        logger.error("db.get_earliest_asos_reading.failed", error=str(exc))
        raise
    finally:
        Session.remove()


def get_asos_readings_since(
    since_utc: datetime, station_id: str = "KBOS"
) -> list[ASOSReadingDocument]:
    """Fetch all ASOS readings for a station since a given UTC timestamp.

    Args:
        since_utc:  Fetch readings with observation_time_utc >= this value.
        station_id: ICAO station code. Defaults to 'KBOS'.

    Returns:
        List of ``ASOSReadingDocument`` sorted oldest-first.

    Raises:
        sqlalchemy.exc.SQLAlchemyError: On database error.
    """
    session = Session()
    try:
        rows = (
            session.query(ASOSReadingORM)
            .filter(
                ASOSReadingORM.station_id == station_id,
                ASOSReadingORM.observation_time_utc >= since_utc,
            )
            .order_by(ASOSReadingORM.observation_time_utc.asc())
            .all()
        )
        return [
            ASOSReadingDocument(
                station_id=r.station_id,
                observation_time_utc=r.observation_time_utc,
                temperature_f=float(r.temperature_f),
                dew_point_f=float(r.dew_point_f) if r.dew_point_f is not None else None,
                wind_speed_mph=float(r.wind_speed_mph) if r.wind_speed_mph is not None else None,
                raw_metar=r.raw_metar,
            )
            for r in rows
        ]
    except Exception as exc:
        logger.error("db.get_asos_readings_since.failed", error=str(exc))
        raise
    finally:
        Session.remove()


def get_asos_readings_for_date(
    past_date: date,
    station_id: str = "KBOS",
) -> list[ASOSReadingDocument]:
    """Return all ASOS readings for a past ET calendar date (midnight to midnight ET).

    Used by Brier scoring to reconstruct the observed state at ~10 AM on each
    historical trading day.

    Args:
        past_date:  The ET calendar date to fetch readings for.
        station_id: ICAO station code. Defaults to 'KBOS'.

    Returns:
        List of ASOSReadingDocument ordered by observation_time_utc ascending.
        Empty list if no readings found or on error.

    Raises:
        Nothing — errors are logged and an empty list is returned.
    """
    _eastern = pytz.timezone("America/New_York")
    day_start_et = _eastern.localize(
        datetime(past_date.year, past_date.month, past_date.day, 0, 0)
    )
    day_end_et = day_start_et + timedelta(days=1)
    day_start_utc = day_start_et.astimezone(timezone.utc)
    day_end_utc = day_end_et.astimezone(timezone.utc)
    session = Session()
    try:
        rows = (
            session.query(ASOSReadingORM)
            .filter(
                ASOSReadingORM.station_id == station_id,
                ASOSReadingORM.observation_time_utc >= day_start_utc,
                ASOSReadingORM.observation_time_utc < day_end_utc,
            )
            .order_by(ASOSReadingORM.observation_time_utc.asc())
            .all()
        )
        return [
            ASOSReadingDocument(
                station_id=r.station_id,
                observation_time_utc=r.observation_time_utc,
                temperature_f=float(r.temperature_f),
                dew_point_f=float(r.dew_point_f) if r.dew_point_f is not None else None,
                wind_speed_mph=float(r.wind_speed_mph) if r.wind_speed_mph is not None else None,
                raw_metar=r.raw_metar,
            )
            for r in rows
        ]
    except Exception as exc:
        logger.warning("db.get_asos_for_date.failed", date=str(past_date), error=str(exc))
        return []
    finally:
        Session.remove()


# ---------------------------------------------------------------------------
# NWP forecasts CRUD
# ---------------------------------------------------------------------------


def upsert_nwp_forecast(doc: NWPForecastDocument) -> None:
    """Insert a new NWP forecast row (always inserts; no unique key on model+date).

    Args:
        doc: ``NWPForecastDocument`` to persist.

    Returns:
        None

    Raises:
        sqlalchemy.exc.SQLAlchemyError: On database error.
    """
    session = Session()
    try:
        orm = NWPForecastORM(
            target_date=doc.target_date,
            model_name=doc.model_name,
            fetched_at_utc=doc.fetched_at_utc,
            hourly_temps=doc.hourly_temps,
            predicted_daily_high=doc.predicted_daily_high,
            mean_cloudcover_10_16=doc.mean_cloudcover_10_16,
            ensemble_highs=doc.ensemble_highs,
            ensemble_spread=doc.ensemble_spread,
        )
        session.add(orm)
        session.commit()
        logger.info(
            "db.upsert_nwp_forecast.done",
            model=doc.model_name,
            target_date=str(doc.target_date),
            high=doc.predicted_daily_high,
        )
    except Exception as exc:
        session.rollback()
        logger.error(
            "db.upsert_nwp_forecast.failed",
            model=doc.model_name,
            error=str(exc),
        )
        raise
    finally:
        Session.remove()


def get_morning_nwp_forecasts(target_date: date) -> dict[str, NWPForecastDocument]:
    """Fetch the first NWP forecast between 10 AM and 1 PM ET for each model on the given date.

    Used by Brier score calibration so that model accuracy is judged on the
    morning-of prediction rather than a late-day revision that has seen much of
    the day's temperature evolution.  The window [10 AM, 1 PM) ET accommodates
    late app starts (e.g. 10:30 AM) while excluding afternoon revisions that
    introduce lookback bias.  Days where no fetch falls in this window are
    excluded from calibration.

    Args:
        target_date: The trading date to look up.

    Returns:
        Dict mapping model_name → ``NWPForecastDocument`` (earliest fetch in window).
        Models with no fetch in the [10 AM, 1 PM) ET window are excluded.

    Raises:
        sqlalchemy.exc.SQLAlchemyError: On database error.
    """
    import pytz

    _ET = pytz.timezone("America/New_York")
    # Accept the first fetch between 10 AM and 1 PM ET.
    # Lower bound: markets are active by 10 AM; earlier fetches reflect pre-open model runs.
    # Upper bound: fetches after 1 PM have seen too much of the day's temperature evolution.
    # This 3-hour window accommodates late app starts (e.g. 10:30 AM) while excluding
    # afternoon revisions that introduce lookback bias.
    window_start_utc = _ET.localize(
        datetime(target_date.year, target_date.month, target_date.day, 10, 0, 0)
    ).astimezone(timezone.utc)
    window_end_utc = _ET.localize(
        datetime(target_date.year, target_date.month, target_date.day, 13, 0, 0)
    ).astimezone(timezone.utc)

    session = Session()
    try:
        rows = (
            session.query(NWPForecastORM)
            .filter(
                NWPForecastORM.target_date == target_date,
                NWPForecastORM.fetched_at_utc >= window_start_utc,
                NWPForecastORM.fetched_at_utc < window_end_utc,
                NWPForecastORM.model_name.in_(["HRRR", "GFS", "ECMWF"]),
            )
            .order_by(
                NWPForecastORM.model_name,
                NWPForecastORM.fetched_at_utc.asc(),  # earliest qualifying fetch first
            )
            .all()
        )

        result: dict[str, NWPForecastDocument] = {}
        for row in rows:
            if row.model_name in result:
                continue  # keep first (earliest at-or-after 10 AM)
            result[row.model_name] = NWPForecastDocument(
                target_date=row.target_date,
                model_name=row.model_name,
                fetched_at_utc=row.fetched_at_utc,
                hourly_temps=row.hourly_temps,
                predicted_daily_high=float(row.predicted_daily_high),
                mean_cloudcover_10_16=(
                    float(row.mean_cloudcover_10_16)
                    if row.mean_cloudcover_10_16 is not None else None
                ),
                ensemble_spread=(
                    float(row.ensemble_spread)
                    if row.ensemble_spread is not None else None
                ),
            )
        return result
    except Exception as exc:
        logger.error("db.get_morning_nwp_forecasts.failed", error=str(exc))
        raise
    finally:
        Session.remove()


def get_nwp_rmse_data(
    lookback_days: int = 30,
    target_date: Optional[date] = None,
) -> list[dict]:
    """Return per-date, per-model NWP RMSE data for calibration and UI display.

    For each CLI-confirmed settled date in the lookback window, fetches the
    first morning NWP forecast in [10 AM, 1 PM) ET for each model and computes
    the signed error against the NWS confirmed final_official_high.

    Uses the same morning-window logic as get_morning_nwp_forecasts() to avoid
    lookback bias from intraday model revisions.

    Args:
        lookback_days: Number of past days to include. Defaults to 30.
        target_date:   Reference date (exclusive upper bound). Defaults to today.

    Returns:
        List of dicts, each with keys:
            target_date (date), model_name (str),
            predicted_daily_high (float), final_official_high (float),
            error_f (float)  — predicted minus actual (positive = overforecast).
        Sorted by target_date ascending, then model_name.

    Raises:
        sqlalchemy.exc.SQLAlchemyError: On database error.
    """
    if target_date is None:
        from kalshi_weather_trader.config.settings import get_target_date
        target_date = get_target_date()

    import pytz as _pytz
    _ET = _pytz.timezone("America/New_York")

    records: list[dict] = []
    session = Session()
    try:
        for d in range(1, lookback_days + 1):
            past_date = target_date - timedelta(days=d)
            # Only include CLI-confirmed settlements to avoid ASOS preliminary values.
            market_row = session.get(MarketORM, past_date)
            if (
                market_row is None
                or market_row.final_official_high is None
                or not market_row.cli_settlement_confirmed
            ):
                continue
            official_high = float(market_row.final_official_high)

            window_start = _ET.localize(
                datetime(past_date.year, past_date.month, past_date.day, 10, 0, 0)
            ).astimezone(timezone.utc)
            window_end = _ET.localize(
                datetime(past_date.year, past_date.month, past_date.day, 13, 0, 0)
            ).astimezone(timezone.utc)

            rows = (
                session.query(NWPForecastORM)
                .filter(
                    NWPForecastORM.target_date == past_date,
                    NWPForecastORM.fetched_at_utc >= window_start,
                    NWPForecastORM.fetched_at_utc < window_end,
                )
                .order_by(NWPForecastORM.model_name, NWPForecastORM.fetched_at_utc.asc())
                .all()
            )
            seen: set[str] = set()
            for row in rows:
                if row.model_name in seen:
                    continue
                seen.add(row.model_name)
                predicted = float(row.predicted_daily_high)
                records.append({
                    "target_date": past_date,
                    "model_name": row.model_name,
                    "predicted_daily_high": predicted,
                    "final_official_high": official_high,
                    "error_f": predicted - official_high,
                })

        records.sort(key=lambda r: (r["target_date"], r["model_name"]))
        return records
    except Exception as exc:
        logger.error("db.get_nwp_rmse_data.failed", error=str(exc))
        raise
    finally:
        Session.remove()


def get_latest_nwp_forecasts(target_date: date) -> dict[str, NWPForecastDocument]:
    """Fetch the most recent forecast for each model for the given date.

    Args:
        target_date: The trading date to look up.

    Returns:
        Dict mapping model_name → ``NWPForecastDocument`` (latest per model).

    Raises:
        sqlalchemy.exc.SQLAlchemyError: On database error.
    """
    session = Session()
    try:
        # Subquery: for each model, get the max fetched_at_utc
        from sqlalchemy import func as sa_func

        rows = (
            session.query(NWPForecastORM)
            .filter(
                NWPForecastORM.target_date == target_date,
                NWPForecastORM.model_name.in_(["HRRR", "GFS", "ECMWF"]),
            )
            .order_by(
                NWPForecastORM.model_name,
                NWPForecastORM.fetched_at_utc.desc(),
            )
            .all()
        )

        result: dict[str, NWPForecastDocument] = {}
        for row in rows:
            if row.model_name in result:
                continue  # keep first (most recent, due to ordering)
            result[row.model_name] = NWPForecastDocument(
                target_date=row.target_date,
                model_name=row.model_name,
                fetched_at_utc=row.fetched_at_utc,
                hourly_temps=row.hourly_temps,
                predicted_daily_high=float(row.predicted_daily_high),
                mean_cloudcover_10_16=float(row.mean_cloudcover_10_16) if row.mean_cloudcover_10_16 is not None else None,
            )
        return result
    except Exception as exc:
        logger.error("db.get_latest_nwp_forecasts.failed", error=str(exc))
        raise
    finally:
        Session.remove()


def get_nwp_forecasts_before_utc(
    target_date: date,
    before_utc: datetime,
) -> dict[str, NWPForecastDocument]:
    """Fetch the latest forecast for each model fetched strictly before before_utc.

    Used by the replay engine to prevent future leakage: at a historical eval_hour,
    only NWP fetches that existed at that moment in time are visible.

    Args:
        target_date: The trading date for which forecasts were made.
        before_utc:  Cutoff UTC timestamp (exclusive upper bound on fetched_at_utc).

    Returns:
        Dict mapping model_name → ``NWPForecastDocument`` (latest fetch before
        cutoff per model). Models with no fetch before ``before_utc`` are excluded.

    Raises:
        sqlalchemy.exc.SQLAlchemyError: On database error.
    """
    session = Session()
    try:
        rows = (
            session.query(NWPForecastORM)
            .filter(
                NWPForecastORM.target_date == target_date,
                NWPForecastORM.fetched_at_utc < before_utc,
                NWPForecastORM.model_name.in_(["HRRR", "GFS", "ECMWF"]),
            )
            .order_by(
                NWPForecastORM.model_name,
                NWPForecastORM.fetched_at_utc.desc(),
            )
            .all()
        )

        result: dict[str, NWPForecastDocument] = {}
        for row in rows:
            if row.model_name in result:
                continue  # keep first (most recent before cutoff, due to ordering)
            result[row.model_name] = NWPForecastDocument(
                target_date=row.target_date,
                model_name=row.model_name,
                fetched_at_utc=row.fetched_at_utc,
                hourly_temps=row.hourly_temps,
                predicted_daily_high=float(row.predicted_daily_high),
                mean_cloudcover_10_16=(
                    float(row.mean_cloudcover_10_16)
                    if row.mean_cloudcover_10_16 is not None else None
                ),
                ensemble_spread=(
                    float(row.ensemble_spread)
                    if row.ensemble_spread is not None else None
                ),
            )
        return result
    except Exception as exc:
        logger.error("db.get_nwp_forecasts_before_utc.failed", error=str(exc))
        raise
    finally:
        Session.remove()


# ---------------------------------------------------------------------------
# System state CRUD
# ---------------------------------------------------------------------------


def get_system_state(target_date: date) -> Optional[SystemStateDocument]:
    """Fetch the system state row for the given trading date.

    Args:
        target_date: The trading date to look up.

    Returns:
        ``SystemStateDocument`` if the row exists, else None.

    Raises:
        sqlalchemy.exc.SQLAlchemyError: On database error.
    """
    session = Session()
    try:
        row = session.get(SystemStateORM, target_date)
        if row is None:
            return None
        return SystemStateDocument(
            target_date=row.target_date,
            kalman_temp_estimate=float(row.kalman_temp_estimate),
            kalman_bias_estimate=float(row.kalman_bias_estimate),
            kalman_covariance=row.kalman_covariance,
            model_weights=row.model_weights,
            mu_drift=float(row.mu_drift),
            theta_decay=float(row.theta_decay),
            sigma_volatility=float(row.sigma_volatility),
            morning_drift_adjustment=float(row.morning_drift_adjustment),
            afternoon_drift_adjustment=float(row.afternoon_drift_adjustment),
            persistence_filter_offset=float(row.persistence_filter_offset) if row.persistence_filter_offset is not None else 0.3,
            sigma_by_block=row.sigma_by_block,
            theta_am=float(row.theta_am) if row.theta_am is not None else None,
            theta_pm=float(row.theta_pm) if row.theta_pm is not None else None,
            ou_max_stationary_std_calibrated=(
                float(row.ou_max_stationary_std_calibrated)
                if row.ou_max_stationary_std_calibrated is not None else None
            ),
            nwp_rmse_n_dates=(
                int(row.nwp_rmse_n_dates) if row.nwp_rmse_n_dates is not None else None
            ),
            kalman_bias_decay_calibrated=(
                float(row.kalman_bias_decay_calibrated)
                if row.kalman_bias_decay_calibrated is not None else None
            ),
            nwp_daily_max_bias=float(row.nwp_daily_max_bias) if row.nwp_daily_max_bias is not None else 0.0,
            last_calibrated_utc=row.last_calibrated_utc,
            last_updated_utc=row.last_updated_utc,
        )
    except Exception as exc:
        logger.error("db.get_system_state.failed", error=str(exc))
        raise
    finally:
        Session.remove()


def upsert_system_state(doc: SystemStateDocument) -> None:
    """Insert or update the system state row.

    Args:
        doc: ``SystemStateDocument`` to persist.

    Returns:
        None

    Raises:
        sqlalchemy.exc.SQLAlchemyError: On database error.
    """
    session = Session()
    try:
        stmt = pg_insert(SystemStateORM).values(
            target_date=doc.target_date,
            kalman_temp_estimate=doc.kalman_temp_estimate,
            kalman_bias_estimate=doc.kalman_bias_estimate,
            kalman_covariance=doc.kalman_covariance,
            model_weights=doc.model_weights,
            mu_drift=doc.mu_drift,
            theta_decay=doc.theta_decay,
            sigma_volatility=doc.sigma_volatility,
            morning_drift_adjustment=doc.morning_drift_adjustment,
            afternoon_drift_adjustment=doc.afternoon_drift_adjustment,
            persistence_filter_offset=doc.persistence_filter_offset,
            sigma_by_block=doc.sigma_by_block,
            theta_am=doc.theta_am,
            theta_pm=doc.theta_pm,
            ou_max_stationary_std_calibrated=doc.ou_max_stationary_std_calibrated,
            nwp_rmse_n_dates=doc.nwp_rmse_n_dates,
            kalman_bias_decay_calibrated=doc.kalman_bias_decay_calibrated,
            nwp_daily_max_bias=doc.nwp_daily_max_bias,
            last_calibrated_utc=doc.last_calibrated_utc,
            last_updated_utc=doc.last_updated_utc,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["target_date"],
            set_={
                "kalman_temp_estimate": stmt.excluded.kalman_temp_estimate,
                "kalman_bias_estimate": stmt.excluded.kalman_bias_estimate,
                "kalman_covariance": stmt.excluded.kalman_covariance,
                "model_weights": stmt.excluded.model_weights,
                "mu_drift": stmt.excluded.mu_drift,
                "theta_decay": stmt.excluded.theta_decay,
                "sigma_volatility": stmt.excluded.sigma_volatility,
                "morning_drift_adjustment": stmt.excluded.morning_drift_adjustment,
                "afternoon_drift_adjustment": stmt.excluded.afternoon_drift_adjustment,
                "persistence_filter_offset": stmt.excluded.persistence_filter_offset,
                "sigma_by_block": stmt.excluded.sigma_by_block,
                "theta_am": stmt.excluded.theta_am,
                "theta_pm": stmt.excluded.theta_pm,
                "ou_max_stationary_std_calibrated": stmt.excluded.ou_max_stationary_std_calibrated,
                "nwp_rmse_n_dates": stmt.excluded.nwp_rmse_n_dates,
                "kalman_bias_decay_calibrated": stmt.excluded.kalman_bias_decay_calibrated,
                "nwp_daily_max_bias": stmt.excluded.nwp_daily_max_bias,
                "last_calibrated_utc": stmt.excluded.last_calibrated_utc,
                "last_updated_utc": stmt.excluded.last_updated_utc,
            },
        )
        session.execute(stmt)
        session.commit()
        logger.debug(
            "db.upsert_system_state.done",
            target_date=str(doc.target_date),
            kalman_T=doc.kalman_temp_estimate,
        )
    except Exception as exc:
        session.rollback()
        logger.error(
            "db.upsert_system_state.failed",
            target_date=str(doc.target_date),
            error=str(exc),
        )
        raise
    finally:
        Session.remove()


# ---------------------------------------------------------------------------
# Intraday snapshots CRUD
# ---------------------------------------------------------------------------


def insert_snapshot(doc: IntradaySnapshotDocument) -> None:
    """Insert a new intraday snapshot row.

    Args:
        doc: ``IntradaySnapshotDocument`` to persist.

    Returns:
        None

    Raises:
        sqlalchemy.exc.SQLAlchemyError: On database error.
    """
    session = Session()
    try:
        orm = IntradaySnapshotORM(
            target_date=doc.target_date,
            snapshot_time_utc=doc.snapshot_time_utc,
            snapshot_time_eastern=doc.snapshot_time_eastern,
            current_asos_temp_f=doc.current_asos_temp_f,
            current_max_observed_f=doc.current_max_observed_f,
            hrrr_predicted_high=doc.hrrr_predicted_high,
            gfs_predicted_high=doc.gfs_predicted_high,
            ecmwf_predicted_high=doc.ecmwf_predicted_high,
            blended_predicted_high=doc.blended_predicted_high,
            kalman_temp_estimate=doc.kalman_temp_estimate,
            kalman_bias_estimate=doc.kalman_bias_estimate,
            kalshi_implied_prob_yes=doc.kalshi_implied_prob_yes,
            kalshi_bid=doc.kalshi_bid,
            kalshi_ask=doc.kalshi_ask,
            kalshi_strike=doc.kalshi_strike,
            model_fair_value_prob=doc.model_fair_value_prob,
            model_edge=doc.model_edge,
            is_forced=doc.is_forced,
        )
        session.add(orm)
        session.commit()
        logger.info(
            "db.insert_snapshot.done",
            target_date=str(doc.target_date),
            time_et=doc.snapshot_time_eastern,
        )
    except Exception as exc:
        session.rollback()
        logger.error("db.insert_snapshot.failed", error=str(exc))
        raise
    finally:
        Session.remove()


def get_snapshots_for_date(target_date: date) -> list[IntradaySnapshotDocument]:
    """Fetch all intraday snapshots for a given trading date, oldest first.

    Args:
        target_date: The trading date to query.

    Returns:
        List of ``IntradaySnapshotDocument`` sorted by snapshot_time_utc ascending.

    Raises:
        sqlalchemy.exc.SQLAlchemyError: On database error.
    """
    session = Session()
    try:
        rows = (
            session.query(IntradaySnapshotORM)
            .filter(IntradaySnapshotORM.target_date == target_date)
            .order_by(IntradaySnapshotORM.snapshot_time_utc.asc())
            .all()
        )
        return [
            IntradaySnapshotDocument(
                target_date=r.target_date,
                snapshot_time_utc=r.snapshot_time_utc,
                snapshot_time_eastern=r.snapshot_time_eastern,
                current_asos_temp_f=float(r.current_asos_temp_f),
                current_max_observed_f=float(r.current_max_observed_f),
                hrrr_predicted_high=float(r.hrrr_predicted_high) if r.hrrr_predicted_high else None,
                gfs_predicted_high=float(r.gfs_predicted_high) if r.gfs_predicted_high else None,
                ecmwf_predicted_high=float(r.ecmwf_predicted_high) if r.ecmwf_predicted_high else None,
                blended_predicted_high=float(r.blended_predicted_high),
                kalman_temp_estimate=float(r.kalman_temp_estimate),
                kalman_bias_estimate=float(r.kalman_bias_estimate),
                kalshi_implied_prob_yes=float(r.kalshi_implied_prob_yes) if r.kalshi_implied_prob_yes else None,
                kalshi_bid=float(r.kalshi_bid) if r.kalshi_bid else None,
                kalshi_ask=float(r.kalshi_ask) if r.kalshi_ask else None,
                kalshi_strike=r.kalshi_strike,
                model_fair_value_prob=float(r.model_fair_value_prob) if r.model_fair_value_prob else None,
                model_edge=float(r.model_edge) if r.model_edge else None,
                is_forced=r.is_forced,
            )
            for r in rows
        ]
    except Exception as exc:
        logger.error("db.get_snapshots_for_date.failed", error=str(exc))
        raise
    finally:
        Session.remove()


# ---------------------------------------------------------------------------
# Trade logs CRUD
# ---------------------------------------------------------------------------


def insert_trade_log(doc: TradeLogDocument) -> None:
    """Insert a trade log row.

    Args:
        doc: ``TradeLogDocument`` to persist.

    Returns:
        None

    Raises:
        sqlalchemy.exc.SQLAlchemyError: On database error.
    """
    session = Session()
    try:
        orm = TradeLogORM(
            trade_id=doc.trade_id,
            target_date=doc.target_date,
            executed_at_utc=doc.executed_at_utc,
            market_ticker=doc.market_ticker,
            action=doc.action,
            kalshi_strike=doc.kalshi_strike,
            contracts=doc.contracts,
            price_cents=doc.price_cents,
            fair_value_prob=doc.fair_value_prob,
            kalshi_implied_prob=doc.kalshi_implied_prob,
            edge_at_execution=doc.edge_at_execution,
            kelly_fraction=doc.kelly_fraction,
            dry_run=doc.dry_run,
            order_id=doc.order_id,
            status=doc.status,
            notes=doc.notes,
            inserted_at=doc.inserted_at,
        )
        session.add(orm)
        session.commit()
        logger.info(
            "db.insert_trade_log.done",
            trade_id=doc.trade_id,
            action=doc.action,
            strike=doc.kalshi_strike,
            contracts=doc.contracts,
            dry_run=doc.dry_run,
        )
    except Exception as exc:
        session.rollback()
        logger.error("db.insert_trade_log.failed", trade_id=doc.trade_id, error=str(exc))
        raise
    finally:
        Session.remove()


def get_recent_asos_readings_by_hours(
    hours: int = 3, station_id: str = "KBOS"
) -> list[ASOSReadingDocument]:
    """Fetch ASOS readings for a station within the last N hours.

    Args:
        hours:      How many hours back to query. Defaults to 3.
        station_id: ICAO station code. Defaults to 'KBOS'.

    Returns:
        List of ``ASOSReadingDocument`` sorted by observation_time_utc ascending.
        Returns empty list on error (logged).

    Raises:
        Nothing — errors are caught, logged, and an empty list is returned.
    """
    from datetime import timedelta

    session = Session()
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        rows = (
            session.query(ASOSReadingORM)
            .filter(
                ASOSReadingORM.station_id == station_id,
                ASOSReadingORM.observation_time_utc >= cutoff,
            )
            .order_by(ASOSReadingORM.observation_time_utc.asc())
            .all()
        )
        return [
            ASOSReadingDocument(
                station_id=r.station_id,
                observation_time_utc=r.observation_time_utc,
                temperature_f=float(r.temperature_f),
                dew_point_f=float(r.dew_point_f) if r.dew_point_f is not None else None,
                wind_speed_mph=float(r.wind_speed_mph) if r.wind_speed_mph is not None else None,
                raw_metar=r.raw_metar,
            )
            for r in rows
        ]
    except Exception as exc:
        logger.error("db.get_recent_asos_readings_by_hours.failed", hours=hours, error=str(exc))
        return []
    finally:
        Session.remove()


def get_recent_snapshots_by_hours(
    target_date: date, hours: int = 3
) -> list[IntradaySnapshotDocument]:
    """Fetch intraday snapshots for a trading date within the last N hours.

    Args:
        target_date: The trading date to query.
        hours:       How many hours back to query. Defaults to 3.

    Returns:
        List of ``IntradaySnapshotDocument`` sorted by snapshot_time_utc ascending.
        Returns empty list on error (logged).

    Raises:
        Nothing — errors are caught, logged, and an empty list is returned.
    """
    from datetime import timedelta

    session = Session()
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        rows = (
            session.query(IntradaySnapshotORM)
            .filter(
                IntradaySnapshotORM.target_date == target_date,
                IntradaySnapshotORM.snapshot_time_utc >= cutoff,
            )
            .order_by(IntradaySnapshotORM.snapshot_time_utc.asc())
            .all()
        )
        return [
            IntradaySnapshotDocument(
                target_date=r.target_date,
                snapshot_time_utc=r.snapshot_time_utc,
                snapshot_time_eastern=r.snapshot_time_eastern,
                current_asos_temp_f=float(r.current_asos_temp_f),
                current_max_observed_f=float(r.current_max_observed_f),
                hrrr_predicted_high=float(r.hrrr_predicted_high) if r.hrrr_predicted_high else None,
                gfs_predicted_high=float(r.gfs_predicted_high) if r.gfs_predicted_high else None,
                ecmwf_predicted_high=float(r.ecmwf_predicted_high) if r.ecmwf_predicted_high else None,
                blended_predicted_high=float(r.blended_predicted_high),
                kalman_temp_estimate=float(r.kalman_temp_estimate),
                kalman_bias_estimate=float(r.kalman_bias_estimate),
                kalshi_implied_prob_yes=float(r.kalshi_implied_prob_yes) if r.kalshi_implied_prob_yes else None,
                kalshi_bid=float(r.kalshi_bid) if r.kalshi_bid else None,
                kalshi_ask=float(r.kalshi_ask) if r.kalshi_ask else None,
                kalshi_strike=r.kalshi_strike,
                model_fair_value_prob=float(r.model_fair_value_prob) if r.model_fair_value_prob else None,
                model_edge=float(r.model_edge) if r.model_edge else None,
                is_forced=r.is_forced,
            )
            for r in rows
        ]
    except Exception as exc:
        logger.error(
            "db.get_recent_snapshots_by_hours.failed",
            target_date=str(target_date),
            hours=hours,
            error=str(exc),
        )
        return []
    finally:
        Session.remove()


def get_recent_trades(target_date: date, limit: int = 10) -> list[TradeLogDocument]:
    """Fetch the most recent trade log entries for a date.

    Args:
        target_date: The trading date to query.
        limit:       Maximum rows to return. Defaults to 10.

    Returns:
        List of ``TradeLogDocument`` sorted by executed_at_utc descending.

    Raises:
        sqlalchemy.exc.SQLAlchemyError: On database error.
    """
    session = Session()
    try:
        rows = (
            session.query(TradeLogORM)
            .filter(TradeLogORM.target_date == target_date)
            .order_by(TradeLogORM.executed_at_utc.desc())
            .limit(limit)
            .all()
        )
        return [
            TradeLogDocument(
                trade_id=r.trade_id,
                target_date=r.target_date,
                executed_at_utc=r.executed_at_utc,
                market_ticker=r.market_ticker,
                action=r.action,
                kalshi_strike=r.kalshi_strike,
                contracts=r.contracts,
                price_cents=r.price_cents,
                fair_value_prob=float(r.fair_value_prob),
                kalshi_implied_prob=float(r.kalshi_implied_prob),
                edge_at_execution=float(r.edge_at_execution),
                kelly_fraction=float(r.kelly_fraction) if r.kelly_fraction else None,
                dry_run=r.dry_run,
                order_id=r.order_id,
                status=r.status,
                notes=r.notes,
                inserted_at=r.inserted_at,
            )
            for r in rows
        ]
    except Exception as exc:
        logger.error("db.get_recent_trades.failed", error=str(exc))
        raise
    finally:
        Session.remove()


# ---------------------------------------------------------------------------
# Phase 3 ensemble and climatology helpers
# ---------------------------------------------------------------------------


def get_latest_ensemble_spread(target_date: date) -> Optional[float]:
    """Return the highest ensemble spread (°F) across GFS_ENS and ECMWF_ENS for the date.

    Uses the maximum spread across models (conservative — takes the most uncertain model's
    view). Returns None if no ensemble data is available.

    Args:
        target_date: Trading date.

    Returns:
        Maximum ensemble spread in °F, or None.

    Raises:
        Nothing — errors are logged.
    """
    session = Session()
    try:
        rows = (
            session.query(NWPForecastORM)
            .filter(
                NWPForecastORM.target_date == target_date,
                NWPForecastORM.model_name.in_(["GFS_ENS", "ECMWF_ENS"]),
                NWPForecastORM.ensemble_spread.isnot(None),
            )
            .order_by(NWPForecastORM.fetched_at_utc.desc())
            .all()
        )
        if not rows:
            return None
        # Use latest fetch per model, then take the max spread
        seen: set[str] = set()
        spreads: list[float] = []
        for row in rows:
            if row.model_name not in seen:
                seen.add(row.model_name)
                spreads.append(float(row.ensemble_spread))
        return max(spreads) if spreads else None
    except Exception as exc:
        logger.error("db.get_latest_ensemble_spread.failed", error=str(exc))
        return None
    finally:
        Session.remove()


def get_blended_cloudcover(target_date: date) -> Optional[float]:
    """Return the blended mean cloudcover (10-16 ET) from the latest NWP forecasts.

    Uses the same model weights as the temperature blend. Returns None if no
    cloudcover data is available for any model.

    Args:
        target_date: Trading date.

    Returns:
        Blended mean cloudcover in [0, 100], or None.

    Raises:
        Nothing — errors are logged.
    """
    try:
        forecasts = get_latest_nwp_forecasts(target_date)
        if not forecasts:
            return None
        # Filter to models that have cloudcover data
        cc_by_model = {
            m: f.mean_cloudcover_10_16
            for m, f in forecasts.items()
            if f.mean_cloudcover_10_16 is not None
        }
        if not cc_by_model:
            return None
        # Equal-weight blend for now (model weights are temperature-calibrated, not CC-specific)
        return round(sum(cc_by_model.values()) / len(cc_by_model), 1)
    except Exception as exc:
        logger.error("db.get_blended_cloudcover.failed", error=str(exc))
        return None


def upsert_historical_daily_high(station_id: str, obs_date: date, high_f: float, source: str = "IEM") -> None:
    """Upsert a single historical daily high temperature record.

    Args:
        station_id: ICAO station code (e.g. 'KBOS').
        obs_date:   Calendar date of observation.
        high_f:     Daily maximum temperature in °F.
        source:     Data source label (default 'IEM').

    Returns:
        None

    Raises:
        sqlalchemy.exc.SQLAlchemyError: On database error.
    """
    session = Session()
    try:
        stmt = pg_insert(HistoricalDailyHighORM).values(
            station_id=station_id,
            obs_date=obs_date,
            high_f=round(float(high_f), 1),
            source=source,
        ).on_conflict_do_update(
            index_elements=["station_id", "obs_date"],
            set_={"high_f": round(float(high_f), 1), "source": source},
        )
        session.execute(stmt)
        session.commit()
    except Exception as exc:
        session.rollback()
        logger.error("db.upsert_historical_daily_high.failed", obs_date=str(obs_date), error=str(exc))
        raise
    finally:
        Session.remove()


def get_historical_daily_highs(station_id: str, start_date: date, end_date: date) -> list[tuple[date, float]]:
    """Return historical daily high temperatures for a date range.

    Args:
        station_id: ICAO station code.
        start_date: First date (inclusive).
        end_date:   Last date (inclusive).

    Returns:
        List of (obs_date, high_f) tuples sorted by date.

    Raises:
        Nothing — errors are logged; returns empty list on failure.
    """
    session = Session()
    try:
        rows = (
            session.query(HistoricalDailyHighORM)
            .filter(
                HistoricalDailyHighORM.station_id == station_id,
                HistoricalDailyHighORM.obs_date >= start_date,
                HistoricalDailyHighORM.obs_date <= end_date,
            )
            .order_by(HistoricalDailyHighORM.obs_date.asc())
            .all()
        )
        return [(row.obs_date, float(row.high_f)) for row in rows]
    except Exception as exc:
        logger.error("db.get_historical_daily_highs.failed", error=str(exc))
        return []
    finally:
        Session.remove()


# ---------------------------------------------------------------------------
# Paper trading CRUD
# ---------------------------------------------------------------------------


def _paper_orm_to_doc(r: PaperTradeORM) -> PaperTradeDocument:
    return PaperTradeDocument(
        position_id=r.position_id,
        target_date=r.target_date,
        market_ticker=r.market_ticker,
        action=r.action,
        kalshi_strike=float(r.kalshi_strike),
        entry_at_utc=r.entry_at_utc,
        entry_price_cents=r.entry_price_cents,
        contracts=r.contracts,
        cost_usd=float(r.cost_usd),
        fair_value_prob=float(r.fair_value_prob),
        edge_at_entry=float(r.edge_at_entry),
        kelly_fraction=float(r.kelly_fraction) if r.kelly_fraction is not None else None,
        budget_mode=r.budget_mode,
        bankroll_at_entry=float(r.bankroll_at_entry) if r.bankroll_at_entry is not None else None,
        status=r.status,
        exit_at_utc=r.exit_at_utc,
        exit_price_cents=r.exit_price_cents,
        pnl_cents=float(r.pnl_cents) if r.pnl_cents is not None else None,
        pnl_usd=float(r.pnl_usd) if r.pnl_usd is not None else None,
        official_high_f=float(r.official_high_f) if r.official_high_f is not None else None,
        settlement_win=r.settlement_win,
        inserted_at=r.inserted_at,
    )


def insert_paper_trade(doc: PaperTradeDocument) -> None:
    """Insert a new paper trade position.

    Args:
        doc: PaperTradeDocument to persist.

    Returns:
        None

    Raises:
        sqlalchemy.exc.SQLAlchemyError: On database error.
    """
    session = Session()
    try:
        row = PaperTradeORM(
            position_id=doc.position_id,
            target_date=doc.target_date,
            market_ticker=doc.market_ticker,
            action=doc.action,
            kalshi_strike=doc.kalshi_strike,
            entry_at_utc=doc.entry_at_utc,
            entry_price_cents=doc.entry_price_cents,
            contracts=doc.contracts,
            cost_usd=round(doc.cost_usd, 2),
            fair_value_prob=round(doc.fair_value_prob, 4),
            edge_at_entry=round(doc.edge_at_entry, 4),
            kelly_fraction=round(doc.kelly_fraction, 6) if doc.kelly_fraction is not None else None,
            budget_mode=doc.budget_mode,
            bankroll_at_entry=round(doc.bankroll_at_entry, 2) if doc.bankroll_at_entry is not None else None,
            status=doc.status,
            inserted_at=doc.inserted_at,
        )
        session.add(row)
        session.commit()
        logger.info("db.insert_paper_trade.ok", position_id=doc.position_id, ticker=doc.market_ticker)
    except Exception as exc:
        session.rollback()
        logger.error("db.insert_paper_trade.failed", error=str(exc))
        raise
    finally:
        Session.remove()


def update_paper_trade_exit(
    position_id: str,
    status: str,
    exit_at_utc: datetime,
    exit_price_cents: int,
    pnl_cents: float,
    pnl_usd: float,
    official_high_f: Optional[float] = None,
    settlement_win: Optional[bool] = None,
) -> None:
    """Close a paper trade position with exit details.

    Args:
        position_id:      UUID of the position to update.
        status:           New status string.
        exit_at_utc:      UTC time of exit.
        exit_price_cents: Exit price in cents (75 for limit sell; 0 or 100 at settlement).
        pnl_cents:        Net profit/loss in total cents.
        pnl_usd:          Net profit/loss in dollars.
        official_high_f:  NWS official daily high (settlement closes only).
        settlement_win:   True if settled as a win (settlement closes only).

    Returns:
        None

    Raises:
        sqlalchemy.exc.SQLAlchemyError: On database error.
    """
    session = Session()
    try:
        vals: dict = {
            "status": status,
            "exit_at_utc": exit_at_utc,
            "exit_price_cents": exit_price_cents,
            "pnl_cents": round(pnl_cents, 2),
            "pnl_usd": round(pnl_usd, 4),
        }
        if official_high_f is not None:
            vals["official_high_f"] = round(official_high_f, 1)
        if settlement_win is not None:
            vals["settlement_win"] = settlement_win
        session.execute(
            update(PaperTradeORM)
            .where(PaperTradeORM.position_id == position_id)
            .values(**vals)
        )
        session.commit()
        logger.info("db.update_paper_trade_exit.ok", position_id=position_id, status=status)
    except Exception as exc:
        session.rollback()
        logger.error("db.update_paper_trade_exit.failed", position_id=position_id, error=str(exc))
        raise
    finally:
        Session.remove()


def get_open_paper_trades(target_date: Optional[date] = None) -> list[PaperTradeDocument]:
    """Return all open (un-exited) paper trade positions.

    Args:
        target_date: If provided, filter to a specific trading date.

    Returns:
        List of open PaperTradeDocument objects, newest-first.

    Raises:
        sqlalchemy.exc.SQLAlchemyError: On database error.
    """
    session = Session()
    try:
        q = session.query(PaperTradeORM).filter(PaperTradeORM.status == "open")
        if target_date is not None:
            q = q.filter(PaperTradeORM.target_date == target_date)
        rows = q.order_by(PaperTradeORM.entry_at_utc.desc()).all()
        return [_paper_orm_to_doc(r) for r in rows]
    except Exception as exc:
        logger.error("db.get_open_paper_trades.failed", error=str(exc))
        raise
    finally:
        Session.remove()


def get_paper_trades_for_date(target_date: date) -> list[PaperTradeDocument]:
    """Return all paper trade positions for a specific date (all statuses), newest-first.

    Args:
        target_date: The trading date to query.

    Returns:
        List of PaperTradeDocument objects.

    Raises:
        sqlalchemy.exc.SQLAlchemyError: On database error.
    """
    session = Session()
    try:
        rows = (
            session.query(PaperTradeORM)
            .filter(PaperTradeORM.target_date == target_date)
            .order_by(PaperTradeORM.entry_at_utc.desc())
            .all()
        )
        return [_paper_orm_to_doc(r) for r in rows]
    except Exception as exc:
        logger.error("db.get_paper_trades_for_date.failed", error=str(exc))
        raise
    finally:
        Session.remove()


def get_paper_trade_summary(
    since_date: Optional[date] = None,
    until_date: Optional[date] = None,
) -> dict:
    """Return aggregate P&L statistics across closed paper trades.

    Args:
        since_date: Include only trades on or after this date (inclusive). None = all history.
        until_date: Include only trades on or before this date (inclusive). None = today.

    Returns:
        Dict with keys: total_trades, closed_trades, wins, losses,
        win_rate, total_pnl_usd, open_positions.

    Raises:
        Nothing — returns zeroed dict on error.
    """
    session = Session()
    try:
        q = session.query(PaperTradeORM)
        if since_date is not None:
            q = q.filter(PaperTradeORM.target_date >= since_date)
        if until_date is not None:
            q = q.filter(PaperTradeORM.target_date <= until_date)
        all_rows = q.all()
        total = len(all_rows)
        open_count = sum(1 for r in all_rows if r.status == "open")
        closed = [r for r in all_rows if r.status != "open"]
        wins = sum(1 for r in closed if r.status in ("settled_win", "limit_sell_closed"))
        losses = sum(1 for r in closed if r.status == "settled_loss")
        total_pnl = sum(float(r.pnl_usd) for r in closed if r.pnl_usd is not None)
        closed_count = len(closed)
        win_rate = wins / closed_count if closed_count > 0 else 0.0
        return {
            "total_trades": total,
            "closed_trades": closed_count,
            "open_positions": open_count,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "total_pnl_usd": round(total_pnl, 2),
        }
    except Exception as exc:
        logger.error("db.get_paper_trade_summary.failed", error=str(exc))
        return {
            "total_trades": 0,
            "closed_trades": 0,
            "open_positions": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_pnl_usd": 0.0,
        }
    finally:
        Session.remove()


def get_paper_trade_daily_series() -> list[dict]:
    """Return a day-by-day P&L series for all closed paper trades.

    Each entry covers one trading date: the number of trades, wins, daily
    P&L, and a running cumulative balance starting from $0.  Only dates
    that have at least one closed trade are included.  Open positions are
    excluded so the balance only reflects resolved outcomes.

    Returns:
        List of dicts ordered by date ascending, each with keys:
          date, trades, wins, losses, daily_pnl_usd, cumulative_balance_usd.

    Raises:
        Nothing — returns empty list on error.
    """
    session = Session()
    try:
        rows = (
            session.query(PaperTradeORM)
            .filter(PaperTradeORM.status != "open")
            .order_by(PaperTradeORM.target_date.asc(), PaperTradeORM.entry_at_utc.asc())
            .all()
        )

        # Group by date
        from collections import defaultdict
        by_date: dict = defaultdict(list)
        for r in rows:
            by_date[r.target_date].append(r)

        series = []
        cumulative = 0.0
        for d in sorted(by_date.keys()):
            day_rows = by_date[d]
            wins = sum(1 for r in day_rows if r.status in ("settled_win", "limit_sell_closed"))
            losses = sum(1 for r in day_rows if r.status == "settled_loss")
            daily_pnl = sum(float(r.pnl_usd) for r in day_rows if r.pnl_usd is not None)
            cumulative += daily_pnl
            series.append({
                "date": d,
                "trades": len(day_rows),
                "wins": wins,
                "losses": losses,
                "daily_pnl_usd": round(daily_pnl, 2),
                "cumulative_balance_usd": round(cumulative, 2),
            })

        return series
    except Exception as exc:
        logger.error("db.get_paper_trade_daily_series.failed", error=str(exc))
        return []
    finally:
        Session.remove()
