# db/connection.py

"""
Database connection factory.

Design decision: we use SQLAlchemy Core (not the ORM) throughout this project.
Raw SQL is written explicitly in the modules that own each query (e.g.
core/data_split.py, core/model_store.py) rather than hidden behind ORM
query-building — this is a deliberate SQL-fluency signal (per the spec,
Section 3) and keeps the analytical queries (Qini-curve input construction,
model comparison) visible and reviewable as SQL, not abstracted away.

DATABASE_URL environment variable controls the backend:
    - unset -> local SQLite file at ./causal_uplift.db (zero-setup dev default)
    - set to a Postgres DSN -> one-line swap to Postgres for production parity

This module intentionally exposes only two things: get_engine() and
init_schema(). Everything else (queries) lives with the code that issues it, so
this file's job stays narrow: "how do we connect," not "what do we ask."
"""

import os
from pathlib import Path

from sqlalchemy import Engine, create_engine, event, text

_DEFAULT_SQLITE_PATH = Path(__file__).resolve().parent.parent / "causal_uplift.db"
_SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def get_database_url() -> str:
    """Resolve the DB connection string, defaulting to a local SQLite file."""
    return os.environ.get("DATABASE_URL", f"sqlite:///{_DEFAULT_SQLITE_PATH}")


def get_engine() -> Engine:
    """
    Create a SQLAlchemy engine.

    For SQLite, `check_same_thread=False` is required because Streamlit and
    pytest can both touch the connection from different threads within a
    single process; this is safe here because we are not sharing a single
    connection object across threads concurrently — each call opens its own
    connection via the engine's pool.

    SQLite also ships with foreign-key enforcement OFF per connection, so
    every REFERENCES clause in schema.sql would otherwise be decorative --
    orphan rows that Postgres rejects would be silently accepted here,
    breaking the "one-line swap to Postgres" claim in the module docstring.
    """
    url = get_database_url()
    is_sqlite = url.startswith("sqlite")
    connect_args = {"check_same_thread": False} if is_sqlite else {}
    engine = create_engine(url, connect_args=connect_args)
    if is_sqlite:
        event.listen(engine, "connect", _enable_sqlite_foreign_keys)
    return engine


def _enable_sqlite_foreign_keys(dbapi_connection, _connection_record) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def _strip_line_comments(sql: str) -> str:
    """
    Remove SQL line comments (`-- ...` to end of line) before statement
    splitting.

    Why this exists: naively splitting schema.sql on literal ';' characters
    breaks if a semicolon appears inside an inline comment (e.g.
    "-- note: X; not Y"), producing a truncated, syntactically invalid
    statement fragment. Stripping comments first removes that ambiguity.

    Known limitation: this does NOT handle semicolons inside string literal
    values (e.g. a DEFAULT 'a;b' clause), because it does not track quote
    state. That's an acceptable simplification for a DDL-only schema file
    with no such literals -- but if schema.sql ever needs a string literal
    containing a semicolon, this function must be upgraded to a real SQL
    tokenizer (e.g. `sqlparse.split`) rather than patched further.
    """
    lines = []
    for line in sql.splitlines():
        comment_idx = line.find("--")
        lines.append(line[:comment_idx] if comment_idx != -1 else line)
    return "\n".join(lines)


def init_schema(engine: Engine | None = None) -> None:
    """
    Apply db/schema.sql against the given engine (or a fresh default engine).

    Idempotent: every table in schema.sql uses `CREATE TABLE IF NOT EXISTS`,
    so calling this repeatedly (e.g., at the top of every test module) is
    safe and does not raise on existing tables.
    """
    engine = engine or get_engine()
    schema_sql = _strip_line_comments(_SCHEMA_PATH.read_text(encoding="utf-8"))
    with engine.begin() as conn:
        for statement in schema_sql.split(";"):
            statement = statement.strip()
            if statement:
                conn.execute(text(statement))
