"""End-to-end integration: seed -> load from DB -> split -> train -> persist."""

import pandas as pd
import pytest
from sqlalchemy import text

from config.schema import load_pilot_config
from core.data_split import load_split_user_ids
from core.features import build_feature_matrix
from core.model_store import load_predictions
from core.pipeline import latest_pilot_id, load_pilot_frames, run_pipeline
from core.t_learner import fit_t_learner
from data_sim.pilot_simulator import PilotSimulator


@pytest.fixture
def seeded(engine, small_config_path):
    from db.seed import seed_demo_pilot

    return engine, seed_demo_pilot(small_config_path), small_config_path


def _in_memory_frames(config_path, pilot_id):
    """What the simulator produced, namespaced and ordered as seed + load do."""
    users, treatment, outcomes, _ = PilotSimulator(load_pilot_config(config_path)).generate()
    return tuple(
        df.assign(user_id=pilot_id + ":" + df["user_id"])
        .sort_values("user_id")
        .reset_index(drop=True)
        for df in (users, treatment, outcomes)
    )


def test_db_round_trip_preserves_what_models_see(seeded):
    engine, pilot_id, config_path = seeded
    db_users, db_treatment, db_outcomes = load_pilot_frames(engine, pilot_id)
    mem_users, mem_treatment, mem_outcomes = _in_memory_frames(config_path, pilot_id)

    pd.testing.assert_frame_equal(build_feature_matrix(db_users), build_feature_matrix(mem_users))
    assert db_treatment["treated"].dtype == bool and db_outcomes["converted"].dtype == bool
    assert db_treatment["treated"].tolist() == mem_treatment["treated"].tolist()
    assert db_outcomes["converted"].tolist() == mem_outcomes["converted"].tolist()


def test_pipeline_persists_test_set_predictions_for_each_model(seeded):
    engine, pilot_id, _ = seeded
    runs = run_pipeline(engine, pilot_id, test_size=0.2, split_seed=1)

    test_ids = set(load_split_user_ids(engine, pilot_id, "test"))
    train_ids = set(load_split_user_ids(engine, pilot_id, "train"))
    assert len(test_ids) == 400 and test_ids.isdisjoint(train_ids)
    assert set(runs) == {"baseline", "t_learner", "x_learner"}
    for run_id in runs.values():
        assert set(load_predictions(engine, run_id).index) == test_ids  # never train rows


def test_db_path_matches_in_memory_path_exactly(seeded):
    """The DB round trip must not change model inputs in any way: training
    through the pipeline and training directly on the simulator's frames
    (same split) must give bit-identical T-learner predictions."""
    engine, pilot_id, config_path = seeded
    runs = run_pipeline(engine, pilot_id, test_size=0.2, split_seed=1, random_state=7)
    db_preds = load_predictions(engine, runs["t_learner"]).sort_index()

    users, treatment, outcomes = _in_memory_frames(config_path, pilot_id)
    train = set(load_split_user_ids(engine, pilot_id, "train"))
    test = set(load_split_user_ids(engine, pilot_id, "test"))
    pick = lambda df, ids: df[df["user_id"].isin(ids)].reset_index(drop=True)  # noqa: E731
    model = fit_t_learner(
        pick(users, train), pick(treatment, train), pick(outcomes, train), random_state=7
    )
    mem_preds = model.predict_cate(pick(users, test)).sort_index()

    assert db_preds.index.tolist() == mem_preds.index.tolist()
    assert db_preds.tolist() == mem_preds.tolist()


def test_rerun_reuses_split_and_rejects_a_different_seed(seeded):
    engine, pilot_id, _ = seeded
    run_pipeline(engine, pilot_id, split_seed=1)
    first_test = set(load_split_user_ids(engine, pilot_id, "test"))
    run_pipeline(engine, pilot_id, split_seed=1)  # must not hit the data_splits PK

    with engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM data_splits")).scalar() == 2_000
        assert conn.execute(text("SELECT COUNT(*) FROM model_runs")).scalar() == 6
    assert set(load_split_user_ids(engine, pilot_id, "test")) == first_test
    with pytest.raises(ValueError, match="already has a split"):
        run_pipeline(engine, pilot_id, split_seed=2)


def test_latest_pilot_id_refuses_to_guess_between_same_day_pilots(seeded):
    from db.seed import seed_demo_pilot

    engine, pilot_id, config_path = seeded
    assert latest_pilot_id(engine) == pilot_id
    seed_demo_pilot(config_path)
    with pytest.raises(ValueError, match="pass --pilot-id"):
        latest_pilot_id(engine)


def test_load_pilot_frames_rejects_unknown_pilot(engine):
    with pytest.raises(ValueError, match="no users found"):
        load_pilot_frames(engine, "missing")
