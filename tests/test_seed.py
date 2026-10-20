import yaml
from sqlalchemy import text

from config.schema import load_pilot_config


def test_seeding_two_pilots_into_one_db_does_not_collide(engine, tmp_path, monkeypatch):
    """Regression: every simulated pilot numbers its users u_00000000..; the
    seed step must namespace them by pilot so a second pilot neither hits
    the users PK nor overwrites/borrows the first pilot's covariates."""
    from db.seed import seed_demo_pilot

    cfg = load_pilot_config("config/default_config.yaml").model_dump()
    cfg["n_users"] = 1_000
    small = tmp_path / "small.yaml"
    small.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    p1 = seed_demo_pilot(str(small))
    p2 = seed_demo_pilot(str(small))

    with engine.connect() as conn:
        n_users = conn.execute(text("SELECT COUNT(*) FROM users")).scalar()
        per_pilot = dict(
            conn.execute(
                text("SELECT pilot_id, COUNT(*) FROM outcomes GROUP BY pilot_id")
            ).fetchall()
        )
    assert n_users == 2_000
    assert per_pilot == {p1: 1_000, p2: 1_000}
