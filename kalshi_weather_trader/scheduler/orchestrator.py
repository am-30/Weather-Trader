"""
APScheduler orchestrator — wires all system components together.

Six scheduled jobs:
  1. fetch_asos_and_update    every 5 min   — ASOS → Kalman update → sync DB
  2. fetch_nwp_and_predict    every 60 min  — NWP → Kalman predict step
  3. evaluate_trade           every 5 min   — trader.evaluate_and_trade()
  4. take_snapshot            every 2 hr    — calibrator.record_snapshot()
  5. midnight_calibration     00:05 ET daily — full calibration cycle
  6. rollover_check           every 30 min  — detect 18:00 ET rollover

Startup sequence:
  1. load_dotenv()
  2. configure_logging()
  3. init_schema()
  4. Immediate ASOS + NWP fetch
  5. load_or_initialize_filter()
  6. scheduler.start()
  7. Signal handlers + event loop

Each scheduler job creates and removes its own SQLAlchemy session
(scoped_session is thread-local — safe for APScheduler thread pool).
"""

from __future__ import annotations

import signal
import sys
import time
from datetime import date, datetime, timezone
from typing import Optional

import pytz
import structlog
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from dotenv import load_dotenv

logger = structlog.get_logger(__name__)

_EASTERN = pytz.timezone("America/New_York")


# ---------------------------------------------------------------------------
# Job 1: ASOS fetch + Kalman update
# ---------------------------------------------------------------------------


def job_fetch_asos_and_update() -> None:
    """Fetch the latest ASOS reading and run a Kalman filter update step.

    1. Fetch ASOS (NWS primary, IEM fallback) → persist + update hard floor.
    2. Load Kalman filter state from DB for today's target date.
    3. Run Kalman update with the new ASOS temperature.
    4. Sync updated filter state back to DB.

    Args:
        None

    Returns:
        None

    Raises:
        Nothing — all errors are logged.
    """
    from kalshi_weather_trader.config.settings import get_target_date
    from kalshi_weather_trader.ingestion.asos_fetcher import fetch_current_observation
    from kalshi_weather_trader.quant.kalman_filter import (
        load_or_initialize_filter,
        sync_filter_to_db,
    )

    try:
        target_date = get_target_date()
        reading = fetch_current_observation()
        if reading is None:
            logger.warning("orchestrator.asos_job.no_reading")
            return

        kf = load_or_initialize_filter(target_date, reading.temperature_f)
        kf.update(reading.temperature_f)
        sync_filter_to_db(kf, target_date)

        logger.info(
            "orchestrator.asos_job.done",
            temp_f=reading.temperature_f,
            kalman_T=round(kf.temperature, 2),
            kalman_B=round(kf.bias, 2),
        )
    except Exception as exc:
        logger.error("orchestrator.asos_job.failed", error=str(exc), exc_info=True)


# ---------------------------------------------------------------------------
# Job 2: NWP fetch + Kalman predict
# ---------------------------------------------------------------------------


def job_fetch_nwp_and_predict() -> None:
    """Fetch NWP forecasts and run a Kalman filter predict step.

    1. Fetch all three NWP models → persist.
    2. Compute blended hourly NWP delta.
    3. Load Kalman filter state and run predict step.
    4. Sync to DB.

    Args:
        None

    Returns:
        None

    Raises:
        Nothing — all errors are logged.
    """
    from kalshi_weather_trader.config.settings import get_target_date
    from kalshi_weather_trader.ingestion.nwp_fetcher import fetch_all_models, get_nwp_curve
    from kalshi_weather_trader.quant.kalman_filter import (
        load_or_initialize_filter,
        sync_filter_to_db,
    )
    from kalshi_weather_trader.db import db_manager

    try:
        target_date = get_target_date()
        fetch_all_models(target_date)

        nwp_curve = get_nwp_curve(target_date)
        if not nwp_curve or len(nwp_curve) < 2:
            logger.warning("orchestrator.nwp_job.no_curve")
            return

        # Compute NWP delta for current hour
        now_et = datetime.now(timezone.utc).astimezone(_EASTERN)
        hour_idx = min(now_et.hour, len(nwp_curve) - 2)
        nwp_delta = nwp_curve[hour_idx + 1] - nwp_curve[hour_idx]

        # Load current ASOS reading as initialisation temp fallback
        latest_asos = db_manager.get_latest_asos_reading()
        init_temp = latest_asos.temperature_f if latest_asos else 60.0

        kf = load_or_initialize_filter(target_date, init_temp)
        kf.predict(nwp_delta=nwp_delta, dt=1.0)
        sync_filter_to_db(kf, target_date)

        logger.info(
            "orchestrator.nwp_job.done",
            nwp_delta=round(nwp_delta, 2),
            kalman_T=round(kf.temperature, 2),
        )
    except Exception as exc:
        logger.error("orchestrator.nwp_job.failed", error=str(exc), exc_info=True)


# ---------------------------------------------------------------------------
# Job 3: Trade evaluation
# ---------------------------------------------------------------------------


def job_evaluate_trade() -> None:
    """Run the full trade evaluation and (potentially) submit an order.

    Args:
        None

    Returns:
        None

    Raises:
        Nothing — errors are logged.
    """
    try:
        from kalshi_weather_trader.execution.trader import evaluate_and_trade

        evaluate_and_trade()
    except Exception as exc:
        logger.error("orchestrator.trade_job.failed", error=str(exc), exc_info=True)


# ---------------------------------------------------------------------------
# Job 4: Intraday snapshot
# ---------------------------------------------------------------------------


def job_take_snapshot() -> None:
    """Record an intraday snapshot with current state and MC pricing.

    Args:
        None

    Returns:
        None

    Raises:
        Nothing — errors are logged.
    """
    try:
        from kalshi_weather_trader.calibration.calibrator import record_snapshot

        record_snapshot(is_forced=False)
    except Exception as exc:
        logger.error("orchestrator.snapshot_job.failed", error=str(exc), exc_info=True)


# ---------------------------------------------------------------------------
# Job 5: Midnight calibration
# ---------------------------------------------------------------------------


def job_midnight_calibration() -> None:
    """Run all four calibration routines at 00:05 Eastern.

    Args:
        None

    Returns:
        None

    Raises:
        Nothing — errors are logged.
    """
    try:
        from kalshi_weather_trader.calibration.calibrator import run_full_calibration

        run_full_calibration()
        logger.info("orchestrator.midnight_calibration.done")
    except Exception as exc:
        logger.error("orchestrator.midnight_calibration.failed", error=str(exc), exc_info=True)


# ---------------------------------------------------------------------------
# Job 6: Rollover check
# ---------------------------------------------------------------------------


def job_rollover_check() -> None:
    """Detect the 6 PM Eastern rollover and initialise tomorrow's market row.

    After 18:00 Eastern, get_target_date() returns tomorrow.  This job ensures
    that a ``markets`` row exists for the new target date so that subsequent
    jobs can write to it immediately.

    Args:
        None

    Returns:
        None

    Raises:
        Nothing — errors are logged.
    """
    from kalshi_weather_trader.config.settings import get_target_date
    from kalshi_weather_trader.db import db_manager
    from kalshi_weather_trader.db.schemas import MarketDocument

    try:
        target_date = get_target_date()
        existing = db_manager.get_market(target_date)
        if existing is None:
            doc = MarketDocument(
                target_date=target_date,
                current_max_observed=-999.0,
                market_status="open",
                auto_trade_enabled=True,
            )
            db_manager.upsert_market(doc)
            logger.info(
                "orchestrator.rollover.new_market_row",
                target_date=str(target_date),
            )
        else:
            logger.debug("orchestrator.rollover.row_exists", target_date=str(target_date))
    except Exception as exc:
        logger.error("orchestrator.rollover.failed", error=str(exc), exc_info=True)


# ---------------------------------------------------------------------------
# Scheduler factory
# ---------------------------------------------------------------------------


def build_scheduler() -> BackgroundScheduler:
    """Construct and configure the APScheduler BackgroundScheduler.

    Args:
        None

    Returns:
        Configured (but not yet started) ``BackgroundScheduler``.

    Raises:
        Nothing.
    """
    from kalshi_weather_trader.config.settings import settings

    scheduler = BackgroundScheduler(
        job_defaults={
            "coalesce": True,
            "max_instances": 1,
            "misfire_grace_time": 60,
        }
    )

    # Job 1: ASOS + Kalman update every 5 minutes
    scheduler.add_job(
        job_fetch_asos_and_update,
        trigger=IntervalTrigger(minutes=settings.asos_fetch_interval_minutes),
        id="fetch_asos",
        name="ASOS Fetch + Kalman Update",
    )

    # Job 2: NWP + Kalman predict every 60 minutes
    scheduler.add_job(
        job_fetch_nwp_and_predict,
        trigger=IntervalTrigger(minutes=settings.nwp_fetch_interval_minutes),
        id="fetch_nwp",
        name="NWP Fetch + Kalman Predict",
    )

    # Job 3: Trade evaluation every 5 minutes
    scheduler.add_job(
        job_evaluate_trade,
        trigger=IntervalTrigger(minutes=settings.trade_eval_interval_minutes),
        id="evaluate_trade",
        name="Trade Evaluation",
    )

    # Job 4: Snapshot every 2 hours
    scheduler.add_job(
        job_take_snapshot,
        trigger=IntervalTrigger(hours=settings.snapshot_interval_hours),
        id="take_snapshot",
        name="Intraday Snapshot",
    )

    # Job 5: Midnight calibration at 00:05 Eastern (APScheduler handles DST)
    scheduler.add_job(
        job_midnight_calibration,
        trigger=CronTrigger(hour=0, minute=5, timezone="America/New_York"),
        id="midnight_calibration",
        name="Midnight Calibration",
    )

    # Job 6: Rollover check every 30 minutes
    scheduler.add_job(
        job_rollover_check,
        trigger=IntervalTrigger(minutes=settings.rollover_check_interval_minutes),
        id="rollover_check",
        name="Rollover Check",
    )

    logger.info("orchestrator.scheduler.built", jobs=[j.id for j in scheduler.get_jobs()])
    return scheduler


# ---------------------------------------------------------------------------
# Startup sequence
# ---------------------------------------------------------------------------


def startup_sequence() -> None:
    """Run immediate data fetches before starting the scheduler.

    Performs:
    1. Initial ASOS fetch to establish current temperature
    2. Initial NWP fetch to populate forecast curves
    3. Ensure today's market row exists in DB
    4. Bootstrap Kalman filter

    Args:
        None

    Returns:
        None

    Raises:
        Nothing — errors are logged; startup continues.
    """
    from kalshi_weather_trader.config.settings import get_target_date
    from kalshi_weather_trader.db import db_manager
    from kalshi_weather_trader.db.schemas import MarketDocument
    from kalshi_weather_trader.ingestion.asos_fetcher import fetch_current_observation
    from kalshi_weather_trader.ingestion.nwp_fetcher import fetch_all_models
    from kalshi_weather_trader.quant.kalman_filter import (
        load_or_initialize_filter,
        sync_filter_to_db,
    )

    target_date = get_target_date()
    logger.info("orchestrator.startup.begin", target_date=str(target_date))

    # Ensure market row exists
    try:
        if db_manager.get_market(target_date) is None:
            db_manager.upsert_market(
                MarketDocument(
                    target_date=target_date,
                    current_max_observed=-999.0,
                    market_status="open",
                    auto_trade_enabled=True,
                )
            )
            logger.info("orchestrator.startup.market_row_created", date=str(target_date))
    except Exception as exc:
        logger.error("orchestrator.startup.market_row_failed", error=str(exc))

    # Initial ASOS fetch
    try:
        reading = fetch_current_observation()
        if reading:
            logger.info("orchestrator.startup.asos_ok", temp_f=reading.temperature_f)
        else:
            logger.warning("orchestrator.startup.asos_failed")
            reading = None
    except Exception as exc:
        logger.error("orchestrator.startup.asos_error", error=str(exc))
        reading = None

    # Initial NWP fetch
    try:
        forecasts = fetch_all_models(target_date)
        logger.info("orchestrator.startup.nwp_ok", models=list(forecasts.keys()))
    except Exception as exc:
        logger.error("orchestrator.startup.nwp_error", error=str(exc))

    # Bootstrap Kalman filter
    try:
        init_temp = reading.temperature_f if reading else 60.0
        kf = load_or_initialize_filter(target_date, init_temp)
        if reading:
            kf.update(reading.temperature_f)
        sync_filter_to_db(kf, target_date)
        logger.info(
            "orchestrator.startup.kalman_ok",
            T=round(kf.temperature, 2),
            B=round(kf.bias, 2),
        )
    except Exception as exc:
        logger.error("orchestrator.startup.kalman_error", error=str(exc))

    logger.info("orchestrator.startup.done")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Start the trading system orchestrator.

    Loads config, initialises the database schema, runs the startup sequence,
    starts the scheduler, and blocks in a heartbeat loop until SIGINT/SIGTERM.

    Args:
        None

    Returns:
        None

    Raises:
        SystemExit: On SIGINT or SIGTERM.
    """
    # 1. Load .env
    load_dotenv()

    # 2. Configure logging
    from kalshi_weather_trader.config.logging_config import configure_logging
    from kalshi_weather_trader.config.settings import settings as cfg

    configure_logging(cfg.log_level)

    logger.info("orchestrator.main.starting", dry_run=cfg.dry_run, env=cfg.kalshi_env)

    # 3. Initialise database schema
    from kalshi_weather_trader.db.db_manager import init_schema

    try:
        init_schema()
    except Exception as exc:
        logger.critical("orchestrator.main.schema_init_failed", error=str(exc))
        sys.exit(1)

    # 4. Run startup sequence
    startup_sequence()

    # 5. Build and start scheduler
    scheduler = build_scheduler()

    def _shutdown(signum, frame):
        logger.info("orchestrator.main.shutdown_signal", signum=signum)
        scheduler.shutdown(wait=False)
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    scheduler.start()
    logger.info("orchestrator.main.scheduler_started")

    # 6. Heartbeat loop
    while True:
        time.sleep(60)
        logger.debug("orchestrator.main.heartbeat", jobs=len(scheduler.get_jobs()))


if __name__ == "__main__":
    main()
