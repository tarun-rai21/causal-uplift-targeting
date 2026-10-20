import pytest
import yaml
from sqlalchemy import text

from config.schema import load_pilot_config
from data_sim.ground_truth import load_ground_truth
from data_sim.pilot_simulator import PilotSimulator
from tests.conftest import insert_pilot


def _small_config(tmp_path, n_users=1_000):
    cfg = load_pilot_config("config/default_config.yaml").model_dump()
    cfg["n_users"] = n_users
    path = tmp_path / "small.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return str(path)


def test_seeding_two_pilots_into_one_db_does_not_collide(engine, tmp_path, monkeypatch):
    """Regression: every simulated pilot numbers its users u_00000000..; the
    seed step must namespace them by pilot so a second pilot neither hits
    the users PK nor overwrites/borrows the first pilot's covariates."""
    from db.seed import seed_demo_pilot

    small = _small_config(tmp_path)
    p1 = seed_demo_pilot(small)
    p2 = seed_demo_pilot(small)

    with engine.connect() as conn:
        n_users = conn.execute(text("SELECT COUNT(*) FROM users")).scalar()
        per_pilot = dict(
            conn.execute(
                text("SELECT pilot_id, COUNT(*) FROM outcomes GROUP BY pilot_id")
            ).fetchall()
        )
    assert n_users == 2_000
    assert per_pilot == {p1: 1_000, p2: 1_000}


def test_seeded_pilot_carries_the_ground_truth_actually_used(engine, tmp_path):
    from db.seed import seed_demo_pilot

    small = _small_config(tmp_path)
    pilot_id = seed_demo_pilot(small)
    _, _, _, expected = PilotSimulator(load_pilot_config(small)).generate()
    assert load_ground_truth(engine, pilot_id) == expected


def test_load_ground_truth_errors_are_explicit(engine):
    with pytest.raises(ValueError, match="no pilot found"):
        load_ground_truth(engine, "missing")
    insert_pilot(engine, "real_pilot", [])  # ground_truth_json left NULL
    with pytest.raises(ValueError, match="no ground truth"):
        load_ground_truth(engine, "real_pilot")
