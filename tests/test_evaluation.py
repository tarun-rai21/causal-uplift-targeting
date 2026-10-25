"""Closed-form, cross-check, and DB-integration tests for core/evaluation.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

from core.evaluation import (
    GRID,
    evaluate_model_runs,
    evaluate_scores,
    gain_curve,
    load_evaluation_frame,
    persist_evaluation,
    qini_curve,
    score_sorted,
)


def test_hand_computed_four_user_example():
    """Sorted users: (T, 1), (C, 0), (T, 0), (C, 1).
    Q(t) = Y_T - Y_C * n_T / n_C -> [0, 1, 1, 1, 0]; random line is 0
    since Q(N) = 0, so qini_coefficient = 3/5.
    G(t) = (Y_T/n_T - Y_C/n_C) * t -> [0, 1, 2, 1.5, 0]; auuc = 4.5/5."""
    treated = np.array([1, 0, 1, 0])
    converted = np.array([1, 0, 0, 1])
    assert qini_curve(treated, converted).tolist() == [0, 1, 1, 1, 0]
    assert gain_curve(treated, converted).tolist() == [0, 1, 2, 1.5, 0]
    s = score_sorted(treated, converted)
    assert s.qini_coefficient == pytest.approx(0.6)
    assert s.auuc == pytest.approx(0.9)


def _random_frame(n=2_000, seed=0):
    rng = np.random.default_rng(seed)
    score = rng.normal(size=n)
    treated = rng.random(n) < 0.5
    converted = rng.random(n) < 0.1 + 0.1 * treated * (score > 0)
    # Top two users: one treated, one control, neither converted -- removes
    # the only point where causalml (nan-interpolates an empty arm) and this
    # module (treats it as 0) legitimately differ.
    top = np.argsort(-score)[:2]
    treated[top], converted[top] = [True, False], [False, False]
    return pd.DataFrame({
        "user_id": [f"u{i:05d}" for i in range(n)],
        "treated": treated, "converted": converted, "score": score,
    })


def test_matches_causalml_exactly():
    metrics = pytest.importorskip("causalml.metrics")
    frame = _random_frame()
    cm = pd.DataFrame({
        "y": frame["converted"].astype(int), "w": frame["treated"].astype(int),
        "score": frame["score"],
    })
    order = np.argsort(-frame["score"].to_numpy())
    ours_q = qini_curve(frame["treated"].to_numpy()[order], frame["converted"].to_numpy()[order])
    ours_g = gain_curve(frame["treated"].to_numpy()[order], frame["converted"].to_numpy()[order])
    ours = score_sorted(frame["treated"].to_numpy()[order], frame["converted"].to_numpy()[order])

    assert np.allclose(ours_q, metrics.get_qini(cm, normalize=False)["score"].to_numpy())
    assert np.allclose(ours_g, metrics.get_cumgain(cm, normalize=False)["score"].to_numpy())
    assert ours.qini_coefficient == pytest.approx(
        float(metrics.qini_score(cm, normalize=False)["score"])
    )
    assert ours.auuc == pytest.approx(float(metrics.auuc_score(cm, normalize=False)["score"]))


def test_scores_ignore_row_order_and_break_ties_by_user_id():
    frame = _random_frame(n=500)
    frame["tied"] = np.round(frame["score"])  # heavy ties
    a, _ = evaluate_scores(frame, ["score", "tied"], n_bootstrap=20)
    b, _ = evaluate_scores(frame.sample(frac=1, random_state=3), ["score", "tied"], n_bootstrap=20)
    for col in ("score", "tied"):
        assert a[col]["qini"] == b[col]["qini"]
        assert a[col]["auuc"] == b[col]["auuc"]


def test_bootstrap_interval_brackets_estimate_and_self_difference_is_zero():
    frame = _random_frame()
    frame["copy"] = frame["score"]
    results, diffs = evaluate_scores(frame, ["score", "copy"], n_bootstrap=200)
    q = results["score"]["qini"]
    assert q.lower <= q.estimate <= q.upper
    assert diffs[("score", "copy")].estimate == 0
    assert diffs[("score", "copy")].lower == diffs[("score", "copy")].upper == 0  # paired
    lo, hi = results["score"]["curve_lower"], results["score"]["curve_upper"]
    assert lo.shape == hi.shape == GRID.shape and (lo <= hi).all()


def test_informative_score_beats_its_negation():
    frame = _random_frame(n=5_000)
    frame["reversed"] = -frame["score"]
    results, diffs = evaluate_scores(frame, ["score", "reversed"], n_bootstrap=200)
    assert results["score"]["qini"].estimate > 0 > results["reversed"]["qini"].estimate
    assert diffs[("score", "reversed")].excludes_zero()


def test_rejects_single_arm_and_duplicate_users():
    frame = _random_frame(n=50)
    with pytest.raises(ValueError, match="both treated and control"):
        evaluate_scores(frame.assign(treated=True), ["score"], n_bootstrap=5)
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_scores(pd.concat([frame, frame.head(1)]), ["score"], n_bootstrap=5)


# --- DB integration ---------------------------------------------------------


@pytest.fixture
def trained(engine, small_config_path):
    from core.pipeline import run_pipeline
    from db.seed import seed_demo_pilot

    pilot_id = seed_demo_pilot(small_config_path)
    return engine, pilot_id, run_pipeline(engine, pilot_id, split_seed=1)


def test_evaluate_and_persist_round_trip(trained):
    engine, _, runs = trained
    run_ids = list(runs.values())
    evaluations, diffs = evaluate_model_runs(engine, run_ids, n_bootstrap=50)
    assert {ev.model_type for ev in evaluations} == set(runs)
    assert all(ev.n_users == 400 for ev in evaluations)  # test split only
    assert len(diffs) == 6

    for _ in range(2):  # re-persisting replaces rather than duplicating
        for ev in evaluations:
            persist_evaluation(engine, ev, n_bootstrap=50, ci_level=0.95, seed=42)
    with engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM model_evaluations")).scalar() == 3
        assert conn.execute(text("SELECT COUNT(*) FROM qini_curve_points")).scalar() == 3 * 101


def test_db_frame_matches_scores_and_outcomes(trained):
    engine, pilot_id, runs = trained
    from core.model_store import load_predictions
    from core.pipeline import load_pilot_frames

    frame = load_evaluation_frame(engine, [runs["t_learner"]])
    preds = load_predictions(engine, runs["t_learner"]).sort_index()
    _, treatment, outcomes = load_pilot_frames(engine, pilot_id)
    joined = treatment.merge(outcomes, on="user_id").set_index("user_id").loc[frame["user_id"]]
    assert frame[runs["t_learner"]].tolist() == preds.loc[frame["user_id"]].tolist()
    assert frame["treated"].tolist() == joined["treated"].tolist()
    assert frame["converted"].tolist() == joined["converted"].tolist()


def test_refuses_to_pair_runs_over_different_users(trained):
    engine, pilot_id, runs = trained
    from core.model_store import persist_model_run

    partial = load_evaluation_frame(engine, [runs["baseline"]]).head(100)
    other = persist_model_run(
        engine, pilot_id, "baseline", "gradient_boosting",
        pd.Series(np.arange(100.0), index=partial["user_id"].to_numpy()),
    )
    with pytest.raises(ValueError, match="different user set"):
        load_evaluation_frame(engine, [runs["baseline"], other])
