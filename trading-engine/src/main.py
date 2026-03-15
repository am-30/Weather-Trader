"""
Application entry point for the weather-trading system.

Startup sequence
----------------
1. Configure structlog.
2. Load and validate settings (fails fast on bad config).
3. Initialise the database schema.
4. Start the APScheduler background scheduler.
5. Block on a signal-aware keep-alive loop.

Shutdown
--------
On SIGINT or SIGTERM, the scheduler is gracefully stopped and the DB
connection pool is closed before the process exits.
"""

from __future__ import annotations

import signal
import sys
import time
from types import FrameType
from typing import Optional

import structlog

from src.logging_config import configure_logging
from src.config import settings
from src.db.connection import close_pool
from src.db.schema import init_schema, log_system_event
import src.scheduler as scheduler_module

logger = structlog.get_logger(__name__)

_running = True


def _handle_shutdown(signum: int, frame: Optional[FrameType]) -> None:
    """Signal handler for SIGINT / SIGTERM.

    Args:
        signum: Signal number.
        frame:  Current stack frame (unused).

    Returns:
        None
    """
    global _running
    logger.info("main.shutdown_signal", signal=signum)
    _running = False


def main() -> None:
    """Run the trading system: configure, initialise, schedule, and block.

    Returns:
        None

    Raises:
        SystemExit: On fatal startup error.
    """
    configure_logging(settings.log_level)
    logger.info(
        "main.startup",
        station=settings.nws_station,
        kalshi_env=settings.kalshi_env,
        max_trade_usd=settings.max_trade_size_usd,
        min_edge_cents=settings.min_edge_cents,
    )

    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)

    try:
        init_schema()
    except Exception as exc:
        logger.critical("main.schema_init_failed", error=str(exc))
        sys.exit(1)

    log_system_event("system.startup", "Trading system started", details={
        "station": settings.nws_station,
        "kalshi_env": settings.kalshi_env,
    })

    try:
        scheduler_module.start()
    except RuntimeError as exc:
        logger.critical("main.scheduler_failed", error=str(exc))
        sys.exit(1)

    logger.info("main.running", message="Scheduler active — press Ctrl+C to stop")

    try:
        while _running:
            time.sleep(1)
    finally:
        logger.info("main.shutdown.begin")
        scheduler_module.stop()
        log_system_event("system.shutdown", "Trading system stopped cleanly")
        close_pool()
        logger.info("main.shutdown.complete")


if __name__ == "__main__":
    main()
