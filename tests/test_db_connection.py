# tests/test_db_connection.py

from sqlalchemy import inspect, text

from db.connection import _strip_line_comments, init_schema

EXPECTED_TABLES = {
    "users",
    "pilots",
    "treatment_assignment",
    "outcomes",
    "data_splits",
    "model_runs",
    "uplift_predictions",
    "qini_curve_points",
    "policy_simulations",
}


def test_init_schema_creates_all_expected_tables(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    from db.connection import get_engine as fresh_get_engine

    engine = fresh_get_engine()
    init_schema(engine)
    tables = set(inspect(engine).get_table_names())
    assert EXPECTED_TABLES.issubset(tables)


def test_init_schema_is_idempotent(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    from db.connection import get_engine as fresh_get_engine

    engine = fresh_get_engine()
    init_schema(engine)
    init_schema(engine)  # must not raise on second call
    tables = set(inspect(engine).get_table_names())
    assert EXPECTED_TABLES.issubset(tables)


def test_strip_line_comments_handles_semicolon_inside_comment():
    """
    Regression test for a real bug found during Phase 0 verification: a
    semicolon inside an inline SQL comment ('-- ...only; unused on real data')
    broke naive split(';') statement parsing. This test pins the fix.
    """
    sql = (
        "CREATE TABLE t (\n"
        "    a TEXT,  -- comment with a semicolon; right here\n"
        "    b TEXT\n"
        ");"
    )
    stripped = _strip_line_comments(sql)
    assert stripped.count(";") == 1  # only the real statement terminator remains
    statements = [s.strip() for s in stripped.split(";") if s.strip()]
    assert len(statements) == 1
    assert "CREATE TABLE t" in statements[0]


def test_data_splits_table_has_expected_columns(monkeypatch):
    """
    Confirms the Phase 1.5 addition (data_splits table, not in the original
    spec) is present with the columns later phases will depend on.
    """
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    from db.connection import get_engine as fresh_get_engine

    engine = fresh_get_engine()
    init_schema(engine)
    with engine.connect() as conn:
        result = conn.execute(text("PRAGMA table_info(data_splits)"))
        columns = {row[1] for row in result}
    assert {"user_id", "pilot_id", "split", "split_seed"}.issubset(columns)
