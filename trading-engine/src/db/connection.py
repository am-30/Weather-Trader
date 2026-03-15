"""
PostgreSQL connection pool using psycopg2.

A module-level ``ThreadedConnectionPool`` is created once on import.
Callers use the ``get_connection()`` context manager which handles
borrow / return from the pool, and logs and re-raises on failure.

Usage::

    from src.db.connection import get_connection

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
"""

from __future__ import annotations

import contextlib
from contextlib import contextmanager
from typing import Generator

import psycopg2
import psycopg2.pool
import structlog

from src.config import settings

logger = structlog.get_logger(__name__)

_pool: psycopg2.pool.ThreadedConnectionPool | None = None

_MIN_CONN = 2
_MAX_CONN = 10


def _create_pool() -> psycopg2.pool.ThreadedConnectionPool:
    """Create and return a new ThreadedConnectionPool.

    Returns:
        An initialised psycopg2 ThreadedConnectionPool.

    Raises:
        psycopg2.OperationalError: If the database is unreachable.
    """
    logger.info("db.pool.creating", dsn_prefix=settings.database_url[:30] + "…")
    return psycopg2.pool.ThreadedConnectionPool(
        minconn=_MIN_CONN,
        maxconn=_MAX_CONN,
        dsn=settings.database_url,
        connect_timeout=10,
    )


def get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    """Return the shared connection pool, creating it on first call.

    Returns:
        The module-level ThreadedConnectionPool instance.

    Raises:
        psycopg2.OperationalError: If the database is unreachable on first use.
    """
    global _pool
    if _pool is None:
        _pool = _create_pool()
    return _pool


@contextmanager
def get_connection() -> Generator[psycopg2.extensions.connection, None, None]:
    """Yield a psycopg2 connection from the pool, returning it when done.

    The connection is automatically committed on successful exit and rolled
    back on exception before being returned to the pool.

    Args:
        None

    Yields:
        An active psycopg2 connection.

    Raises:
        psycopg2.Error: On database-level errors (re-raised after rollback).
        Exception: Any other exception (re-raised after rollback).
    """
    pool = get_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except psycopg2.Error as exc:
        logger.error("db.connection.error", error=str(exc), exc_info=True)
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        pool.putconn(conn)


def close_pool() -> None:
    """Close all connections in the pool on application shutdown.

    Returns:
        None
    """
    global _pool
    if _pool is not None:
        logger.info("db.pool.closing")
        _pool.closeall()
        _pool = None
