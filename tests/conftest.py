"""Shared DB fixtures. Foreign keys are enforced (see db/connection.py), so
any test that writes child rows must first insert the parent pilot/users."""

from __future__ import annotations

from collections.abc import Iterable

import pytest
import yaml
from sqlalchemy import text

from config.schema import load_pilot_config


@pytest.fixture
def engine(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    from db.connection import get_engine, init_schema

    eng = get_engine()
    init_schema(eng)
    return eng


@pytest.fixture
def small_config_path(tmp_path) -> str:
    """The default config shrunk to 2,000 users, written to a temp YAML."""
    cfg = load_pilot_config("config/default_config.yaml").model_dump()
    cfg["n_users"] = 2_000
    path = tmp_path / "small.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return str(path)


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
        rows = [{"u": u} for u in user_ids]
        if rows:
            conn.execute(
                text(
                    "INSERT INTO users (user_id, signup_date, region, device_type) "
                    "VALUES (:u, '2023-01-01', 'north', 'mobile')"
                ),
                rows,
            )
