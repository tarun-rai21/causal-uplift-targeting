import pytest
from sqlalchemy import text

from config.schema import load_pilot_config
from data_sim.ground_truth import load_ground_truth
from data_sim.pilot_simulator import PilotSimulator
from tests.conftest import insert_pilot


def test_seeding_two_pilots_into_one_db_does_not_collide(engine, small_config_path):
    """Regression: every simulated pilot numbers its users u_00000000..; the
    seed step must namespace them by pilot so a second pilot neither hits
    the users PK nor overwrites/borrows the first pilot's covariates."""
    from db.seed import seed_demo_pilot

    p1 = seed_demo_pilot(small_config_path)
    p2 = seed_demo_pilot(small_config_path)

    with engine.connect() as conn:
        n_users = conn.execute(text("SELECT COUNT(*) FROM users")).scalar()
        per_pilot = dict(
            conn.execute(
                text("SELECT pilot_id, COUNT(*) FROM outcomes GROUP BY pilot_id")
            ).fetchall()
        )
    assert n_users == 4_000
    assert per_pilot == {p1: 2_000, p2: 2_000}


def test_seeded_pilot_carries_the_ground_truth_actually_used(engine, small_config_path):
    from db.seed import seed_demo_pilot

    pilot_id = seed_demo_pilot(small_config_path)
    _, _, _, expected = PilotSimulator(load_pilot_config(small_config_path)).generate()
    assert load_ground_truth(engine, pilot_id) == expected


def test_load_ground_truth_errors_are_explicit(engine):
    with pytest.raises(ValueError, match="no pilot found"):
        load_ground_truth(engine, "missing")
    insert_pilot(engine, "real_pilot", [])  # ground_truth_json left NULL
    with pytest.raises(ValueError, match="no ground truth"):
        load_ground_truth(engine, "real_pilot")
