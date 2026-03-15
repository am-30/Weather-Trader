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
from datetime import date, datetime, timezone
from typing import Optional

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
    current_max_observed = Column(Numeric(5, 1), nullable=False, default=-999.0)
    market_status = Column(String(20), nullable=False, default="open")
    auto_trade_enabled = Column(Boolean, nullable=False, default=True)
    final_official_high = Column(Numeric(5, 1), nullable=True)
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
    kalshi_strike = Column(SmallInteger, nullable=True)
    model_fair_value_prob = Column(Numeric(6, 4), nullable=True)
    model_edge = Column(Numeric(6, 4), nullable=True)
    is_forced = Column(Boolean, nullable=False, default=False)
    inserted_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


class TradeLogORM(Base):
    """ORM mapping for the ``trade_logs`` table."""

    __tablename__ = "trade_logs"

    trade_id = Column(String(36), primary_key=True)
    target_date = Column(Date, nullable=False)
    executed_at_utc = Column(DateTime(timezone=True), nullable=False)
    market_ticker = Column(String(100), nullable=False)
    action = Column(String(10), nullable=False)
    kalshi_strike = Column(SmallInteger, nullable=False)
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


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------


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
            current_max_observed=float(row.current_max_observed),
            market_status=row.market_status,
            auto_trade_enabled=row.auto_trade_enabled,
            final_official_high=(
                float(row.final_official_high) if row.final_official_high is not None else None
            ),
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
            last_updated_utc=doc.last_updated_utc,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["target_date"],
            set_={
                "current_max_observed": stmt.excluded.current_max_observed,
                "market_status": stmt.excluded.market_status,
                "auto_trade_enabled": stmt.excluded.auto_trade_enabled,
                "final_official_high": stmt.excluded.final_official_high,
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
            .filter(NWPForecastORM.target_date == target_date)
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
            )
        return result
    except Exception as exc:
        logger.error("db.get_latest_nwp_forecasts.failed", error=str(exc))
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
