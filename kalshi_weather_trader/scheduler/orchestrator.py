"""
APScheduler orchestrator — wires all system components together.

Seven scheduled jobs:
  1. fetch_asos_and_update    every 2 min   — ASOS → Kalman update → sync DB
  2. fetch_nwp_and_predict    every 60 min  — NWP → Kalman predict step
  3. evaluate_trade           every 5 min   — trader.evaluate_and_trade()
  4. take_snapshot            every 2 hr    — calibrator.record_snapshot()
  5. midnight_calibration     00:05 ET daily — full calibration cycle
  6. rollover_check           every 30 min  — detect 18:00 ET rollover
  7. confirm_settlement       10:05 ET daily — NWS CLI official high → update DB

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
from datetime import date, datetime, timedelta, timezone
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

    1. Fetch ASOS (IEM primary, AVWX secondary, NWS last resort) → persist all
       new readings + update hard floor.
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

        # Look up NWP blended forecast for the current ET hour (cheap DB read).
        # Required so the Kalman update computes the departure z = asos - nwp
        # rather than treating the raw ASOS reading as the observation.
        from kalshi_weather_trader.ingestion.nwp_fetcher import get_nwp_curve
        nwp_curve_asos = get_nwp_curve(target_date)
        now_et_asos = datetime.now(timezone.utc).astimezone(_EASTERN)
        nwp_current_hour: Optional[float] = (
            nwp_curve_asos[now_et_asos.hour]
            if nwp_curve_asos and len(nwp_curve_asos) > now_et_asos.hour
            else None
        )
        if nwp_current_hour is None:
            logger.warning("orchestrator.asos_job.nwp_unavailable_for_update")

        # Snapshot the stored state timestamp before loading so we can inject
        # proportional process noise below (Fix B: keeps K nonzero between the
        # hourly NWP predict steps during continuous normal operation).
        from kalshi_weather_trader.db import db_manager as _db
        state_before = _db.get_system_state(target_date)

        kf = load_or_initialize_filter(
            target_date, reading.temperature_f, nwp_at_load_time=nwp_current_hour
        )

        # Inject Q scaled by the time elapsed since the last DB sync (dt < 0.5h
        # means normal tick cadence; larger gaps are already inflated inside
        # load_or_initialize_filter, so we skip here to avoid double-counting).
        if state_before is not None and state_before.last_updated_utc is not None:
            now_utc = datetime.now(timezone.utc)
            dt_hours = (now_utc - state_before.last_updated_utc).total_seconds() / 3600
            if 0 < dt_hours < 0.5:
                kf.predict(dt=dt_hours)  # no NWP arg — just inflate P

        kf.update(reading.temperature_f, nwp_current_hour=nwp_current_hour)
        sync_filter_to_db(kf, target_date)

        logger.info(
            "orchestrator.asos_job.done",
            temp_f=reading.temperature_f,
            nwp_current=nwp_current_hour,
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
    2. Guard: skip predict step during the post-rollover pre-market gap
       (6 PM–midnight ET) when target_date is tomorrow and the NWP curve
       is indexed from midnight tomorrow — applying tonight's hour index
       to tomorrow's NWP curve produces a physically wrong delta.
    3. Compute blended hourly NWP delta (normal trading hours only).
    4. Load Kalman filter state and run predict step.
    5. Sync to DB.

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
        now_et = datetime.now(timezone.utc).astimezone(_EASTERN)

        # Always fetch and persist NWP — needed for MC and rollover pre-fetch.
        fetch_all_models(target_date)

        # PRE-MARKET GAP GUARD: between the 6 PM rollover and midnight ET,
        # target_date is tomorrow but wall-clock time is still tonight.
        # nwp_curve[hour_et] at e.g. 8 PM maps to tomorrow's 8 PM slope
        # (potentially a post-frontal drop of −3 °F/hr), not tonight's
        # actual temperature evolution.  ASOS updates every 5 min are
        # sufficient to track T0 during this window.
        if target_date > now_et.date():
            logger.info(
                "orchestrator.nwp_job.pre_market_gap_skip",
                target_date=str(target_date),
                wall_clock_hour_et=now_et.hour,
            )
            return

        nwp_curve = get_nwp_curve(target_date)
        if not nwp_curve or len(nwp_curve) < 2:
            logger.warning("orchestrator.nwp_job.no_curve")
            return

        # Current hour's absolute NWP forecast (used to update filter's NWP reference).
        # The state equation no longer uses a delta — departure dT is stable across
        # NWP hours; only the NWP reference and P are updated.
        hour_idx = min(now_et.hour, len(nwp_curve) - 1)
        nwp_current = nwp_curve[hour_idx]

        # Load current ASOS reading as initialisation temp fallback
        latest_asos = db_manager.get_latest_asos_reading()
        init_temp = latest_asos.temperature_f if latest_asos else 60.0

        kf = load_or_initialize_filter(target_date, init_temp, nwp_at_load_time=nwp_current)
        kf.predict(nwp_at_current_hour=nwp_current, dt=1.0)
        sync_filter_to_db(kf, target_date)

        logger.info(
            "orchestrator.nwp_job.done",
            nwp_current=round(nwp_current, 2),
            kalman_T=round(kf.temperature, 2),
            kalman_B=round(kf.bias, 2),
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
    jobs can write to it immediately.  On first detection of the new target
    date it also initialises tomorrow's Kalman state with the current ASOS
    temperature and today's converged bias (warm-start), preventing the NWP
    predict job from racing to initialise with a stale fallback or resetting
    bias to 0.0 (CLAUDE.md open issue #10).

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
        now_et = datetime.now(timezone.utc).astimezone(_EASTERN)
        target_date = get_target_date()
        existing = db_manager.get_market(target_date)
        if existing is None:
            doc = MarketDocument(
                target_date=target_date,
                current_max_observed=-999.0,
                market_status="open",
                auto_trade_enabled=False,
            )
            db_manager.upsert_market(doc)
            logger.info(
                "orchestrator.rollover.new_market_row",
                target_date=str(target_date),
            )
            # Pre-fetch tomorrow's NWP immediately so the post-rollover MC
            # simulation has forecast data and doesn't fall back to a flat curve.
            try:
                from kalshi_weather_trader.ingestion.nwp_fetcher import fetch_all_models
                nwp_results = fetch_all_models(target_date)
                logger.info(
                    "orchestrator.rollover.nwp_prefetched",
                    date=str(target_date),
                    models=list(nwp_results.keys()),
                )
            except Exception as nwp_exc:
                logger.warning(
                    "orchestrator.rollover.nwp_prefetch_failed",
                    error=str(nwp_exc),
                )

            # Re-fetch today's NWP with the latest model run so the overnight
            # bridge portion of the stitched MC curve reflects current forecasts.
            try:
                from kalshi_weather_trader.ingestion.nwp_fetcher import fetch_all_models
                today_nwp_results = fetch_all_models(now_et.date())
                logger.info(
                    "orchestrator.rollover.today_nwp_refreshed",
                    date=str(now_et.date()),
                    models=list(today_nwp_results.keys()),
                )
            except Exception as today_nwp_exc:
                logger.warning(
                    "orchestrator.rollover.today_nwp_refresh_failed",
                    error=str(today_nwp_exc),
                )

            # Initialise tomorrow's Kalman state at rollover time.
            # T0  = current ASOS temperature (actual current conditions).
            # Bias = today's converged Kalman bias (warm-start; avoids cold
            #        start at 0.0 per CLAUDE.md open issue #10).
            # Guard: only write if system_state for tomorrow doesn't already
            # exist, making this block idempotent on repeated firings.
            try:
                from kalshi_weather_trader.quant.kalman_filter import KalmanFilter, sync_filter_to_db
                from kalshi_weather_trader.ingestion.asos_fetcher import fetch_current_observation

                if db_manager.get_system_state(target_date) is None:
                    today = now_et.date()
                    asos_reading = fetch_current_observation()
                    init_temp = asos_reading.temperature_f if asos_reading else 60.0

                    today_state = db_manager.get_system_state(today)
                    init_bias = today_state.kalman_bias_estimate if today_state else 0.0

                    kf = KalmanFilter(
                        initial_dt=0.0,          # fresh departure for tomorrow's trading day
                        initial_bias=init_bias,  # carry today's converged NWP bias
                        nwp_current_hour=None,   # NWP for tomorrow fires within 60 min
                    )
                    sync_filter_to_db(kf, target_date)
                    logger.info(
                        "orchestrator.rollover.kalman_initialized",
                        target_date=str(target_date),
                        T0=init_temp,
                        bias=init_bias,
                        bias_source="warm" if today_state else "cold",
                    )
            except Exception as kf_exc:
                logger.warning("orchestrator.rollover.kalman_init_failed", error=str(kf_exc))
        else:
            logger.debug("orchestrator.rollover.row_exists", target_date=str(target_date))
    except Exception as exc:
        logger.error("orchestrator.rollover.failed", error=str(exc), exc_info=True)


# ---------------------------------------------------------------------------
# Job 7: Settlement check
# ---------------------------------------------------------------------------


def job_check_settlement() -> None:
    """Check if today's market has settled and record the official high.

    Runs after 7 PM ET. Uses the actual calendar date (not target_date,
    which is already tomorrow after 6 PM rollover). Computes the daily
    high from stored ASOS readings and marks the market as settled.

    Args:
        None

    Returns:
        None

    Raises:
        Nothing — all errors caught and logged.
    """
    log = structlog.get_logger()
    try:
        eastern = pytz.timezone("US/Eastern")
        now_et = datetime.now(eastern)
        if now_et.hour < 19:
            return

        # Use actual calendar date, not get_target_date() which rolls over at 6 PM
        calendar_date = now_et.date()

        # Check if already settled
        from kalshi_weather_trader.db import db_manager
        from kalshi_weather_trader.db.schemas import MarketDocument

        market = db_manager.get_market(calendar_date)
        if market is not None and market.final_official_high is not None:
            return  # Already settled

        # Compute day's max from stored ASOS readings.
        # Use NWS EST-fixed midnight (UTC-5, no DST) to match the official
        # observation window that Kalshi settles on.
        from kalshi_weather_trader.config.settings import get_nws_day_bounds
        day_start, day_end = get_nws_day_bounds(calendar_date)
        asos_readings = db_manager.get_asos_readings_since(day_start)
        # Filter to only today's readings
        today_readings = [r for r in asos_readings if r.observation_time_utc < day_end]

        if not today_readings:
            log.warning("settlement.no_asos_readings", date=str(calendar_date))
            return

        official_high = max(r.temperature_f for r in today_readings)
        log.info(
            "settlement.recording",
            date=str(calendar_date),
            official_high=official_high,
            source="asos_preliminary",
        )

        # Upsert market with settled status
        if market is None:
            market_doc = MarketDocument(
                target_date=calendar_date,
                final_official_high=official_high,
                market_status="settled",
                auto_trade_enabled=False,
            )
        else:
            market_doc = MarketDocument(
                target_date=market.target_date,
                final_official_high=official_high,
                market_status="settled",
                auto_trade_enabled=market.auto_trade_enabled,
                current_max_observed=market.current_max_observed,
            )
        db_manager.upsert_market(market_doc)
        log.info("settlement.complete", date=str(calendar_date), official_high=official_high)
    except Exception as e:
        log.error("settlement.failed", error=str(e))


# ---------------------------------------------------------------------------
# Job 8: NWS CLI confirmation settlement
# ---------------------------------------------------------------------------


def job_confirm_settlement() -> None:
    """Update yesterday's market with the NWS official daily maximum temperature.

    Runs once daily at 10:05 AM Eastern, after the NWS CLI product is typically
    posted (~9:30 AM ET).  Overwrites the preliminary ASOS-based
    ``final_official_high`` written by ``job_check_settlement()`` with the
    authoritative NWS value, then triggers a full calibration so Brier scores
    and drift adjustments are computed against the correct settlement figure.

    If the CLI product has not yet been posted (returns None), no DB change is
    made and no calibration is triggered — the ASOS preliminary value remains
    as the calibration fallback.

    Args:
        None

    Returns:
        None

    Raises:
        Nothing — all errors are caught and logged.
    """
    log = structlog.get_logger(__name__)
    try:
        eastern = pytz.timezone("America/New_York")
        yesterday = (datetime.now(eastern) - timedelta(days=1)).date()

        from kalshi_weather_trader.ingestion.nws_cli_fetcher import fetch_official_daily_high

        cli_high = fetch_official_daily_high(yesterday)

        if cli_high is None:
            log.warning(
                "settlement.cli_not_available",
                date=str(yesterday),
                reason="NWS CLI not posted yet or MAXIMUM field missing",
            )
            return

        from kalshi_weather_trader.db import db_manager
        from kalshi_weather_trader.db.schemas import MarketDocument

        market = db_manager.get_market(yesterday)
        old_high = market.final_official_high if market else None

        log.info(
            "settlement.cli_confirmed",
            date=str(yesterday),
            old_high=old_high,
            cli_high=cli_high,
        )

        if market is None:
            market_doc = MarketDocument(
                target_date=yesterday,
                final_official_high=cli_high,
                market_status="settled",
                auto_trade_enabled=False,
                cli_settlement_confirmed=True,
            )
        else:
            market_doc = MarketDocument(
                target_date=market.target_date,
                final_official_high=cli_high,
                market_status="settled",
                auto_trade_enabled=market.auto_trade_enabled,
                current_max_observed=market.current_max_observed,
                cli_settlement_confirmed=True,
            )

        db_manager.upsert_market(market_doc)
        log.info("settlement.cli_written", date=str(yesterday), official_high=cli_high)

        # Re-run calibration so Brier scores reflect the authoritative value.
        # Must pass `yesterday` explicitly — get_target_date() has already rolled
        # to today's date at 10:05 AM, and calibrating today instead of yesterday
        # would compute Brier scores against today's (incomplete) NWP forecasts.
        try:
            from kalshi_weather_trader.calibration.calibrator import run_full_calibration

            run_full_calibration(yesterday)
            log.info("settlement.calibration_triggered", date=str(yesterday))
        except Exception as cal_exc:
            log.error(
                "settlement.calibration_failed",
                date=str(yesterday),
                error=str(cal_exc),
            )

    except Exception as exc:
        structlog.get_logger(__name__).error(
            "settlement.confirm_job.failed", error=str(exc), exc_info=True
        )


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

    # Job 1: ASOS + Kalman update every 2 minutes (default).
    # Actual API calls are rate-limited inside fetch_current_observation() to
    # asos_min_fetch_interval_minutes (default 4 min), so the shorter scheduler
    # interval reduces reaction latency without increasing server load.
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

    # Job 7: Settlement check every 30 minutes (active after 7 PM ET)
    scheduler.add_job(
        job_check_settlement,
        trigger=IntervalTrigger(minutes=30),
        id="settlement_check",
        name="Settlement Check",
        replace_existing=True,
    )

    # Job 8: NWS CLI confirmation at 10:05 AM ET daily
    scheduler.add_job(
        job_confirm_settlement,
        trigger=CronTrigger(hour=10, minute=5, timezone="America/New_York"),
        id="confirm_settlement",
        name="NWS CLI Settlement Confirmation",
        replace_existing=True,
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
    from kalshi_weather_trader.config.settings import get_target_date, get_trading_day_bounds  # noqa: F401
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

    # Run schema migrations before any DB access — ensures columns added in
    # later sessions exist regardless of which entry point started the app.
    try:
        from kalshi_weather_trader.db.db_manager import init_schema
        init_schema()
    except Exception as exc:
        logger.error("orchestrator.startup.schema_init_failed", error=str(exc))

    # Ensure market row exists
    try:
        if db_manager.get_market(target_date) is None:
            db_manager.upsert_market(
                MarketDocument(
                    target_date=target_date,
                    current_max_observed=-999.0,
                    market_status="open",
                    auto_trade_enabled=False,
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

    # Hard-floor catch-up: scan today's ASOS readings for the actual peak
    try:
        day_start, _ = get_trading_day_bounds()
        asos_today = db_manager.get_asos_readings_since(day_start)
        if asos_today:
            import math
            day_max = math.floor(max(r.temperature_f for r in asos_today))
            db_manager.update_hard_floor(target_date, float(day_max))
            logger.info("startup.hard_floor.catchup", max_temp=day_max, readings=len(asos_today))
    except Exception as e:
        logger.warning("startup.hard_floor.catchup.failed", error=str(e))

    # Initial NWP fetch
    try:
        forecasts = fetch_all_models(target_date)
        logger.info("orchestrator.startup.nwp_ok", models=list(forecasts.keys()))
    except Exception as exc:
        logger.error("orchestrator.startup.nwp_error", error=str(exc))

    # Bootstrap Kalman filter
    try:
        init_temp = reading.temperature_f if reading else 60.0

        # NWP fetch completed just above — get current-hour value for filter init.
        from kalshi_weather_trader.ingestion.nwp_fetcher import get_nwp_curve as _get_nwp_startup
        _startup_nwp_curve = _get_nwp_startup(target_date)
        _now_et_startup = datetime.now(timezone.utc).astimezone(_EASTERN)
        _nwp_startup: Optional[float] = (
            _startup_nwp_curve[_now_et_startup.hour]
            if _startup_nwp_curve and len(_startup_nwp_curve) > _now_et_startup.hour
            else None
        )

        kf = load_or_initialize_filter(target_date, init_temp, nwp_at_load_time=_nwp_startup)
        if reading:
            kf.update(reading.temperature_f, nwp_current_hour=_nwp_startup)
        sync_filter_to_db(kf, target_date)
        logger.info(
            "orchestrator.startup.kalman_ok",
            T=round(kf.temperature, 2),
            B=round(kf.bias, 2),
            nwp_startup=_nwp_startup,
        )
    except Exception as exc:
        logger.error("orchestrator.startup.kalman_error", error=str(exc))

    # Missed calibration catch-up
    try:
        from kalshi_weather_trader.calibration.calibrator import run_full_calibration

        state = db_manager.get_system_state(target_date)
        eastern = pytz.timezone("US/Eastern")
        today_et = datetime.now(eastern).date()
        needs_calibration = (
            state is None
            or state.last_calibrated_utc is None
            or state.last_calibrated_utc.astimezone(eastern).date() < today_et
        )
        if needs_calibration:
            logger.info("startup.calibration.catchup", reason="missed midnight calibration")
            run_full_calibration(target_date)
    except Exception as e:
        logger.warning("startup.calibration.catchup.failed", error=str(e))

    # NWS CLI catch-up: if yesterday's final_official_high is missing or still
    # equals current_max_observed (ASOS preliminary), try to fetch the official value.
    try:
        eastern = pytz.timezone("US/Eastern")
        yesterday = (datetime.now(eastern) - timedelta(days=1)).date()
        yesterday_market = db_manager.get_market(yesterday)
        needs_cli = (
            yesterday_market is not None
            and not yesterday_market.cli_settlement_confirmed
        )
        if needs_cli:
            logger.info(
                "startup.cli_catchup.attempting",
                date=str(yesterday),
                current_value=yesterday_market.final_official_high,
            )
            from kalshi_weather_trader.ingestion.nws_cli_fetcher import fetch_official_daily_high
            cli_high = fetch_official_daily_high(yesterday)
            if cli_high is not None:
                from kalshi_weather_trader.db.schemas import MarketDocument
                market_doc = MarketDocument(
                    target_date=yesterday_market.target_date,
                    final_official_high=cli_high,
                    market_status="settled",
                    auto_trade_enabled=yesterday_market.auto_trade_enabled,
                    current_max_observed=yesterday_market.current_max_observed,
                    cli_settlement_confirmed=True,
                )
                db_manager.upsert_market(market_doc)
                logger.info(
                    "startup.cli_catchup.success",
                    date=str(yesterday),
                    official_high=cli_high,
                )
            else:
                logger.info("startup.cli_catchup.not_available", date=str(yesterday))
    except Exception as e:
        logger.warning("startup.cli_catchup.failed", error=str(e))

    # Startup snapshot: if in trading hours and no snapshot in the last 90 min,
    # take one immediately so each app restart contributes a drift calibration
    # data point even when the scheduler never fires during a short session.
    try:
        from kalshi_weather_trader.calibration.calibrator import record_snapshot

        now_et = datetime.now(timezone.utc).astimezone(_EASTERN)
        if 8 <= now_et.hour < 19:
            existing_snaps = db_manager.get_snapshots_for_date(target_date)
            cutoff = datetime.now(timezone.utc) - timedelta(minutes=90)
            recent = any(s.snapshot_time_utc >= cutoff for s in existing_snaps)
            if not recent:
                logger.info("startup.snapshot.taking", reason="no recent snapshot")
                record_snapshot(target_date, is_forced=False)
    except Exception as e:
        logger.warning("startup.snapshot.failed", error=str(e))

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
