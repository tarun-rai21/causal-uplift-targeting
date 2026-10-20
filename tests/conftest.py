"""Shared DB fixtures. Foreign keys are enforced (see db/connection.py), so
any test that writes child rows must first insert the parent pilot/users."""

from __future__ import annotations

from collections.abc import Iterable

import pytest
from sqlalchemy import text


@pytest.fixture
def engine(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    from db.connection import get_engine, init_schema

    eng = get_engine()
    init_schema(eng)
    return eng


def insert_pilot(engine, pilot_id: str, user_ids: Iterable[str]) -> None:
    """Insert a pilot plus minimal users rows -- just enough parent rows for
    FK-checked writes to data_splits / model_runs / uplift_predictions."""
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO pilots (pilot_id, name, start_date, budget_cap) "
                "VALUES (:p, 'test', '2023-01-01', NULL)"
            ),
            {"p": pilot_id},
        )
        conn.execute(
            text(
                "INSERT INTO users (user_id, signup_date, region, device_type) "
                "VALUES (:u, '2023-01-01', 'north', 'mobile')"
            ),
            [{"u": u} for u in user_ids],
        )
