"""
Phase 10: the whole system, end to end, through its real entrypoints.

Each stage runs as a separate `python -m ...` process -- exactly what a user
types -- in order, against one temp DB, from a temp working directory with
only PYTHONPATH pointing at the repo (so nothing relies on the cwd). After
every stage the test asserts the DB state the NEXT stage depends on, so a
contract break between stages fails here even when each stage's own unit
tests pass.

    db.seed -> core.pipeline -> core.evaluation -> core.policy
            -> validation.harness --pilot-id -> Streamlit page
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from sqlalchemy import create_engine, text
from streamlit.testing.v1 import AppTest

from config.schema import load_pilot_config

REPO = Path(__file__).resolve().parent.parent
N_USERS = 2_000
N_TEST = 400  # test_size 0.2


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    work = tmp_path / "elsewhere"  # deliberately NOT the repo root
    work.mkdir()
    db_url = f"sqlite:///{(tmp_path / 'e2e.db').as_posix()}"
    cfg = load_pilot_config().model_dump()
    cfg["n_users"] = N_USERS
    config_path = tmp_path / "pilot.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.setenv("DATABASE_URL", db_url)  # for the in-process AppTest stage
    env = {**os.environ, "DATABASE_URL": db_url, "PYTHONPATH": str(REPO),
           "PYTHONIOENCODING": "utf-8"}
    return work, env, db_url, config_path, tmp_path


def _run(module: str, *args: str, work: Path, env: dict) -> str:
    result = subprocess.run(
        [sys.executable, "-m", module, *args],
        cwd=work, env=env, capture_output=True, text=True, encoding="utf-8", timeout=600,
    )
    assert result.returncode == 0, f"{module} failed:\n{result.stdout}\n{result.stderr}"
    return result.stdout


def _count(engine, sql: str) -> int:
    with engine.connect() as conn:
        return conn.execute(text(sql)).scalar()


def test_full_system_through_its_entrypoints(workspace):
    work, env, db_url, config_path, tmp = workspace
    engine = create_engine(db_url)

    # 1. Simulate a pilot (with its frozen ground truth).
    out = _run("db.seed", "--config", str(config_path), work=work, env=env)
    assert f"with {N_USERS} users" in out
    with engine.connect() as conn:
        pilot_id, truth = conn.execute(
            text("SELECT pilot_id, ground_truth_json FROM pilots")
        ).one()
    assert truth is not None
    assert _count(engine, "SELECT COUNT(*) FROM outcomes") == N_USERS

    # 2. Split, train all three models + economics, store test predictions.
    out = _run("core.pipeline", work=work, env=env)
    assert {"baseline", "t_learner", "x_learner"} <= set(out.split())
    assert _count(engine, "SELECT COUNT(*) FROM data_splits") == N_USERS
    assert _count(engine, "SELECT COUNT(*) FROM model_runs") == 3
    assert _count(engine, "SELECT COUNT(*) FROM uplift_predictions") == 3 * N_TEST
    assert _count(engine, "SELECT COUNT(*) FROM user_economics") == N_TEST
    # Predictions exist ONLY for test-split users -- never training rows.
    assert _count(engine, """
        SELECT COUNT(*) FROM uplift_predictions p
        JOIN model_runs m ON m.model_run_id = p.model_run_id
        JOIN data_splits s ON s.user_id = p.user_id AND s.pilot_id = m.pilot_id
        WHERE s.split = 'train'""") == 0

    # 3. Qini / AUUC with paired bootstrap, persisted.
    out = _run("core.evaluation", "--n-bootstrap", "50", work=work, env=env)
    assert "Paired Qini-coefficient differences" in out
    assert _count(engine, "SELECT COUNT(*) FROM model_evaluations") == 3
    assert _count(engine, "SELECT COUNT(*) FROM qini_curve_points") == 3 * 101

    # 4. Budget policy curves, valued from the holdout, persisted.
    out = _run("core.policy", "--cashback-rate", "0.1", "--min-roi", "1.0",
               "--budget", "50", "--n-bootstrap", "50", work=work, env=env)
    assert out.count("-> budget 50: target") == 3
    assert _count(engine, "SELECT COUNT(*) FROM policy_simulations") == 3 * 21
    # The ROI floor applies to uplift models only; naive has no uplift estimate.
    assert _count(engine, """
        SELECT COUNT(DISTINCT p.model_run_id) FROM policy_simulations p
        JOIN model_runs m ON m.model_run_id = p.model_run_id
        WHERE m.model_type = 'baseline' AND p.min_roi IS NULL""") == 1

    # 5. Stored models validated against the stored ground truth.
    report = tmp / "report.md"
    _run("validation.harness", "--pilot-id", pilot_id, "--n-bootstrap", "50",
         "--out", str(report), work=work, env=env)
    text_report = report.read_text(encoding="utf-8")
    assert f"Stored pilot `{pilot_id}`" in text_report
    assert "### Claims" in text_report and "Policy: cashback 10%" in text_report

    # 6. The app renders every tab for this pilot without an exception.
    at = AppTest.from_file(str(REPO / "app" / "streamlit_app.py"), default_timeout=300).run()
    assert not at.exception, at.exception
    assert pilot_id in at.selectbox[0].options[0]
    assert len(at.metric) == 4  # simulated -> true-net headline tiles
