"""
Database layer: connection pool and schema management.

Exposes ``get_connection()`` for obtaining a psycopg2 connection from the
shared pool, and ``init_schema()`` for applying DDL on startup.
"""
