"""
APScheduler-based task scheduler for the trading system.

Configures and starts four periodic jobs:
    1. weather_cycle      — Fetch NWS observations and compute daily max.
    2. market_snapshot    — Fetch and persist Kalshi market quotes.
    3. forecast_cycle     — Re-generate the temperature forecast.
    4. trade_eval_cycle   — Evaluate markets and place orders.

All intervals are controlled by ``settings``. The scheduler runs in the
background thread pool; call ``start()`` from the main process.
"""

from __future__ import annotations

from datetime import datetime, timezone

import structlog
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED, JobExecutionEvent

from src.config import settings
from src.db.schema import log_system_event

logger = structlog.get_logger(__name__)

_scheduler: BackgroundScheduler | None = None


def _job_listener(event: JobExecutionEvent) -> None:
    """APScheduler event listener for completed and errored jobs.

    Args:
        event: APScheduler ``JobExecutionEvent``.

    Returns:
        None
    """
    if event.exception:
        logger.error(
            "scheduler.job.error",
            job_id=event.job_id,
            error=str(event.exception),
        )
        log_system_event(
            "scheduler.job.error",
            f"Job {event.job_id} raised: {event.exception}",
            level="error",
        )
    else:
        logger.debug("scheduler.job.success", job_id=event.job_id)


def _weather_job() -> None:
    """Scheduled job: fetch NWS observations and compute daily max.

    Returns:
        None
    """
    from src.data_feeds.nws import run_weather_cycle

    logger.info("scheduler.weather_job.start")
    result = run_weather_cycle()
    logger.info("scheduler.weather_job.done", **result)


def _market_snapshot_job() -> None:
    """Scheduled job: fetch Kalshi market quotes and persist snapshots.

    Returns:
        None
    """
    from src.kalshi.client import kalshi_client
    from src.db.connection import get_connection

    logger.info("scheduler.market_snapshot_job.start")
    try:
        markets = kalshi_client.get_markets(event_ticker="KXMAXTEMP", status="open")
        kbos = [m for m in markets if settings.nws_station in m.ticker.upper()]

        with get_connection() as conn:
            with conn.cursor() as cur:
                for market in kbos:
                    cur.execute(
                        """
                        INSERT INTO market_snapshots
                            (ticker, event_ticker, title, status,
                             yes_bid, yes_ask, no_bid, no_ask,
                             last_price, volume, open_interest,
                             close_time, expiration_time, fetched_at)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        """,
                        (
                            market.ticker,
                            market.event_ticker,
                            market.title,
                            market.status.value,
                            market.yes_bid,
                            market.yes_ask,
                            market.no_bid,
                            market.no_ask,
                            market.last_price,
                            market.volume,
                            market.open_interest,
                            market.close_time,
                            market.expiration_time,
                            market.fetched_at,
                        ),
                    )
        logger.info("scheduler.market_snapshot_job.done", markets_saved=len(kbos))
    except Exception as exc:
        logger.error("scheduler.market_snapshot_job.failed", error=str(exc), exc_info=True)


def _forecast_job() -> None:
    """Scheduled job: re-generate the temperature forecast.

    Returns:
        None
    """
    from src.forecasting.temperature import generate_forecast

    logger.info("scheduler.forecast_job.start")
    try:
        forecast = generate_forecast()
        logger.info(
            "scheduler.forecast_job.done",
            mean_f=forecast.mean_f,
            std_f=forecast.std_f,
        )
    except Exception as exc:
        logger.error("scheduler.forecast_job.failed", error=str(exc), exc_info=True)


def _trade_eval_job() -> None:
    """Scheduled job: evaluate markets and place orders.

    Returns:
        None
    """
    from src.trading.engine import run_trade_evaluation

    logger.info("scheduler.trade_eval_job.start")
    result = run_trade_evaluation(dry_run=False)
    logger.info("scheduler.trade_eval_job.done", **result)


def start() -> BackgroundScheduler:
    """Create, configure, and start the background scheduler.

    Registers all four periodic jobs with intervals from ``settings`` and
    attaches the job listener. Idempotent — returns the existing scheduler
    if already running.

    Returns:
        The running ``BackgroundScheduler`` instance.

    Raises:
        RuntimeError: If the scheduler fails to start.
    """
    global _scheduler

    if _scheduler is not None and _scheduler.running:
        logger.warning("scheduler.already_running")
        return _scheduler

    _scheduler = BackgroundScheduler(timezone="UTC")
    _scheduler.add_listener(_job_listener, EVENT_JOB_ERROR | EVENT_JOB_EXECUTED)

    _scheduler.add_job(
        _weather_job,
        trigger="interval",
        minutes=settings.weather_fetch_interval_minutes,
        id="weather_cycle",
        next_run_time=datetime.now(timezone.utc),
        max_instances=1,
    )

    _scheduler.add_job(
        _market_snapshot_job,
        trigger="interval",
        minutes=settings.market_fetch_interval_minutes,
        id="market_snapshot",
        next_run_time=datetime.now(timezone.utc),
        max_instances=1,
    )

    _scheduler.add_job(
        _forecast_job,
        trigger="interval",
        minutes=settings.forecast_interval_minutes,
        id="forecast_cycle",
        next_run_time=datetime.now(timezone.utc),
        max_instances=1,
    )

    _scheduler.add_job(
        _trade_eval_job,
        trigger="interval",
        minutes=settings.trade_eval_interval_minutes,
        id="trade_eval",
        max_instances=1,
    )

    try:
        _scheduler.start()
        logger.info(
            "scheduler.started",
            weather_interval_min=settings.weather_fetch_interval_minutes,
            market_interval_min=settings.market_fetch_interval_minutes,
            forecast_interval_min=settings.forecast_interval_minutes,
            trade_eval_interval_min=settings.trade_eval_interval_minutes,
        )
    except Exception as exc:
        logger.error("scheduler.start.failed", error=str(exc))
        raise RuntimeError(f"Scheduler failed to start: {exc}") from exc

    return _scheduler


def stop() -> None:
    """Gracefully shut down the scheduler.

    Returns:
        None
    """
    global _scheduler
    if _scheduler and _scheduler.running:
        logger.info("scheduler.stopping")
        _scheduler.shutdown(wait=True)
        _scheduler = None
