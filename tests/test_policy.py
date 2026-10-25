"""Closed-form, property, and DB tests for core/economics.py + core/policy.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

from core.policy import (
    BUDGET_GRID,
    build_policy,
    load_policy_frame,
    outcome_estimate,
    persist_policy_curve,
    policy_curve,
    policy_inputs,
    roi_ranking,
    select_targets,
)

# Three users at cashback rate 0.1, hand-computable:
#   a: tau 0.15, p1 0.23, m 40 -> cost 0.92, incr rev 6.0,  roi 6.52
#   b: tau 0.01, p1 0.41, m 60 -> cost 2.46, incr rev 0.6,  roi 0.24
#   c: tau 0.02, p1 0.02, m 25 -> cost 0.05, incr rev 0.5,  roi 10.0
TOY = dict(
    user_ids=["a", "b", "c"],
    tau=np.array([0.15, 0.01, 0.02]),
    p1=np.array([0.23, 0.41, 0.02]),
    m=np.array([40.0, 60.0, 25.0]),
)


def _toy_inputs(rate=0.1):
    t = TOY
    return policy_inputs(
        t["user_ids"], roi_ranking(t["tau"], t["p1"], rate), t["p1"], t["m"], rate,
        uplift=t["tau"],
    )


def test_hand_computed_economics():
    inputs = _toy_inputs()
    assert inputs["expected_cost"].tolist() == pytest.approx([0.92, 2.46, 0.05])
    assert inputs["expected_incremental_revenue"].tolist() == pytest.approx([6.0, 0.6, 0.5])
    assert inputs["roi"].tolist() == pytest.approx([0.15 / 0.023, 0.01 / 0.041, 10.0])


def test_selection_is_a_budget_fitting_prefix_of_the_roi_ranking():
    inputs = _toy_inputs()
    assert select_targets(inputs, 0.0) == []
    assert select_targets(inputs, 0.05) == ["c"]
    assert select_targets(inputs, 0.96) == ["c"]  # a would bring spend to 0.97
    assert select_targets(inputs, 0.97) == ["c", "a"]
    # b alone would fit a budget of 2.5 -- but the walk stops at the first
    # user who does not fit, so the selected set stays a prefix.
    assert select_targets(inputs, 2.5) == ["c", "a"]
    assert select_targets(inputs, 10.0) == ["c", "a", "b"]


def test_roi_floor_drops_users_who_return_less_than_their_cashback():
    inputs = _toy_inputs()
    assert select_targets(inputs, 10.0, min_roi=1.0) == ["c", "a"]  # b: ROI 0.24


def test_naive_policy_has_no_revenue_estimate_and_rejects_a_roi_floor():
    t = TOY
    naive = policy_inputs(t["user_ids"], [0.4, 0.9, 0.1], t["p1"], t["m"], 0.1)
    assert select_targets(naive, 2.46) == ["b"]  # ranks by its score: b first
    curve = policy_curve(naive)
    targeted = curve["n_targeted"] > 0  # nobody targeted -> nothing earned, legitimately 0
    assert curve.loc[targeted, "model_incremental_revenue"].isna().all()
    assert (curve.loc[~targeted, "model_incremental_revenue"] == 0).all()
    with pytest.raises(ValueError, match="naive policy has none"):
        select_targets(naive, 1.0, min_roi=1.0)


def test_curve_is_monotone_with_diminishing_marginal_roi():
    rng = np.random.default_rng(0)
    n = 5_000
    tau, p1 = rng.normal(0.03, 0.05, n), rng.uniform(0.02, 0.5, n)
    m = rng.uniform(20, 70, n)
    inputs = policy_inputs(
        [f"u{i}" for i in range(n)], roi_ranking(tau, p1, 0.1), p1, m, 0.1, uplift=tau
    )
    curve = policy_curve(inputs)
    assert curve["budget_pct"].tolist() == BUDGET_GRID.tolist()
    assert (curve["expected_spend"] <= curve["budget"] + 1e-9).all()
    assert curve["n_targeted"].is_monotonic_increasing
    assert curve["expected_spend"].is_monotonic_increasing
    marginal = curve["model_marginal_roi"].dropna().to_numpy()
    assert (np.diff(marginal) <= 1e-9).all()  # ranked by ROI -> diminishing returns
    assert curve["n_targeted"].iloc[-1] == n


def test_outcome_estimate_hand_computed():
    """Treated revenues [50, 0], control [10, 0]: means 25 and 5 over n = 4.
    incremental = 4 * (25 - 5) = 80; spend = 4 * 0.1 * 25 = 10; net = 70."""
    est = outcome_estimate(
        treated=[True, True, False, False], revenue=[50.0, 0.0, 10.0, 0.0],
        cashback_rate=0.1, n_bootstrap=200,
    )
    assert est["incremental_revenue"] == pytest.approx(80.0)
    assert est["spend"] == pytest.approx(10.0)
    assert est["net"] == pytest.approx(70.0)
    assert est["net_lower"] <= est["net"] <= est["net_upper"]


def test_outcome_estimate_is_missing_when_an_arm_is_empty():
    est = outcome_estimate([True, True], [10.0, 0.0], cashback_rate=0.1)
    assert all(np.isnan(v) for v in est.values())


def test_outcome_estimate_is_unbiased_under_randomization():
    """Each user's true incremental revenue is 3 and cashback 0.1 * 20 = 2,
    so true net per user is 1. Averaged over many randomized assignments,
    the estimate must center on n * 1."""
    rng = np.random.default_rng(0)
    n, nets = 400, []
    for _ in range(300):
        treated = rng.random(n) < 0.5
        revenue = np.where(treated, rng.normal(20, 5, n), rng.normal(17, 5, n))
        nets.append(outcome_estimate(treated, revenue, 0.1, n_bootstrap=2)["net"])
    assert np.mean(nets) == pytest.approx(n * 1.0, rel=0.05)


def test_curve_values_each_budget_prefix_from_outcomes():
    t = TOY
    inputs = _toy_inputs()
    outcomes = pd.DataFrame({
        "user_id": t["user_ids"], "treated": [True, False, True], "revenue": [40.0, 0.0, 25.0],
    })
    curve = policy_curve(inputs, outcomes=outcomes, cashback_rate=0.1, n_bootstrap=20)
    # Small budgets select only "c" (treated): no control user -> no estimate.
    only_c = curve["n_targeted"] == 1
    assert curve.loc[only_c, "ab_net_revenue"].isna().all()
    # Everyone: treated mean (40 + 25) / 2 = 32.5, control mean 0 -> n=3.
    full = curve.iloc[-1]
    assert full["ab_incremental_revenue"] == pytest.approx(3 * 32.5)
    assert full["ab_spend"] == pytest.approx(3 * 0.1 * 32.5)
    with pytest.raises(ValueError, match="cashback_rate is required"):
        policy_curve(inputs, outcomes=outcomes)


@pytest.mark.parametrize("rate", [0.0, 1.5])
def test_rejects_invalid_cashback_rate(rate):
    with pytest.raises(ValueError, match="cashback_rate"):
        _toy_inputs(rate)


def test_economics_recover_segment_revenue_and_treated_conversion():
    from config.schema import load_pilot_config
    from core.data_split import compute_data_split, subset_to_split
    from core.economics import fit_economics
    from data_sim.pilot_simulator import PilotSimulator

    cfg = load_pilot_config("config/default_config.yaml").model_copy(update={"n_users": 20_000})
    users, treatment, outcomes, gt = PilotSimulator(cfg).generate()
    split = compute_data_split(treatment, outcomes, "p", test_size=0.2, split_seed=1)
    train = subset_to_split(split, "train", users, treatment, outcomes)
    (test_users,) = subset_to_split(split, "test", users)

    est = fit_economics(*train).predict(test_users)
    assert est["p_convert_if_treated"].between(0, 1).all()
    assert (est["revenue_per_conversion"] >= 0).all()
    # Population-level calibration against the truth (individual estimates
    # are noisy; segment is hidden from the model).
    seg = test_users["segment"]
    true_p1 = seg.map(gt.true_treated_probability).mean()
    true_m = (seg.map(gt.expected_treated_revenue).sum()
              / seg.map(gt.true_treated_probability).sum())
    assert est["p_convert_if_treated"].mean() == pytest.approx(true_p1, abs=0.02)
    weighted_m = (est["revenue_per_conversion"] * est["p_convert_if_treated"]).sum() / est[
        "p_convert_if_treated"
    ].sum()
    assert weighted_m == pytest.approx(true_m, rel=0.08)


def test_pipeline_stores_economics_and_policy_curves_persist(engine, small_config_path):
    from core.pipeline import run_pipeline
    from db.seed import seed_demo_pilot

    pilot_id = seed_demo_pilot(small_config_path)
    runs = run_pipeline(engine, pilot_id, split_seed=1)
    run_pipeline(engine, pilot_id, split_seed=1)  # economics are replaced, not duplicated

    with engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM user_economics")).scalar() == 400

    for model_type, run_id in runs.items():
        frame = load_policy_frame(engine, run_id)
        assert len(frame) == 400 and (frame["model_type"] == model_type).all()
        assert frame["treated"].dtype == bool
        curve = policy_curve(
            build_policy(frame, 0.1), outcomes=frame[["user_id", "treated", "revenue"]],
            cashback_rate=0.1, n_bootstrap=20,
        )
        assert curve["ab_net_revenue"].iloc[-1] == pytest.approx(
            curve["ab_incremental_revenue"].iloc[-1] - curve["ab_spend"].iloc[-1]
        )
        persist_policy_curve(engine, run_id, curve, cashback_rate=0.1, min_roi=None)
        if model_type == "baseline":
            targeted = curve["n_targeted"] > 0
            assert curve.loc[targeted, "model_incremental_revenue"].isna().all()
        else:
            assert curve["model_incremental_revenue"].notna().all()
    with engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM policy_simulations")).scalar() == 3 * 21
        assert conn.execute(
            text("SELECT COUNT(DISTINCT policy_curve_id) FROM policy_simulations")
        ).scalar() == 3


def test_policy_frame_requires_stored_economics(engine):
    from core.model_store import persist_model_run
    from tests.conftest import insert_pilot

    insert_pilot(engine, "p", ["u1"])
    run_id = persist_model_run(engine, "p", "t_learner", "gradient_boosting",
                               pd.Series([0.1], index=["u1"]))
    with pytest.raises(ValueError, match="run core.pipeline"):
        load_policy_frame(engine, run_id)
