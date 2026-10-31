"""Round-trip + validation tests for core/model_store.py."""

from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import text

from core.model_store import load_predictions, persist_model_run
from tests.conftest import insert_pilot


def _seed_pilot(engine, pilot_id="p1"):
    insert_pilot(engine, pilot_id, ["u1", "u2", "u3"])


def test_round_trip(engine):
    _seed_pilot(engine)
    preds = pd.Series([0.3, 0.1, 0.2], index=["u1", "u2", "u3"])
    run_id = persist_model_run(engine, "p1", "t_learner", "gradient_boosting", preds)
    loaded = load_predictions(engine, run_id)
    assert loaded.sort_index().tolist() == preds.sort_index().tolist()


def test_rank_pct_is_descending_percentile(engine):
    _seed_pilot(engine)
    preds = pd.Series([0.3, 0.1, 0.2], index=["u1", "u2", "u3"])
    run_id = persist_model_run(engine, "p1", "t_learner", "gradient_boosting", preds)
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT user_id, rank_pct FROM uplift_predictions "
                "WHERE model_run_id = :r ORDER BY rank_pct"
            ),
            {"r": run_id},
        ).fetchall()
    # highest uplift (u1) gets smallest rank_pct
    assert [r[0] for r in rows] == ["u1", "u3", "u2"]


def test_rejects_predictions_for_unknown_users(engine):
    """FK enforcement regression guard: predictions for users that were
    never persisted must be rejected (as Postgres would), not stored as
    orphans -- and the model_runs insert must roll back with them."""
    _seed_pilot(engine)
    with pytest.raises(Exception, match="FOREIGN KEY"):
        persist_model_run(
            engine, "p1", "t_learner", "gradient_boosting", pd.Series([0.1], index=["ghost"])
        )
    with engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM model_runs")).scalar() == 0


def test_rejects_bad_model_type(engine):
    with pytest.raises(ValueError, match="model_type"):
        persist_model_run(
            engine, "p1", "s_learner", "gradient_boosting", pd.Series([0.1], index=["u1"])
        )


def test_rejects_duplicate_user_ids(engine):
    with pytest.raises(ValueError, match="duplicates"):
        persist_model_run(
            engine, "p1", "t_learner", "gradient_boosting",
            pd.Series([0.1, 0.2], index=["u1", "u1"]),
        )


def test_rejects_unknown_base_learner(engine):
    with pytest.raises(ValueError, match="base_learner"):
        persist_model_run(engine, "p1", "t_learner", "gbc", pd.Series([0.1], index=["u1"]))
