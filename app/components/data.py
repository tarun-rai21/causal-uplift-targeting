"""
Data layer for the Streamlit app: everything the pages show, computed from
the DB with the project's own modules. Plain functions of an engine (no
Streamlit imports), so they are testable without a browser and the app
stays a thin presentation layer -- per the README architecture, it holds no
modeling logic of its own.

What a page can show depends on what the pilot has:
    model runs + economics   -> policy curves (core.pipeline)
    model_evaluations        -> Qini curves   (core.evaluation)
    ground_truth_json        -> TRUE values   (simulated pilots only; real
                                pilots fall back to outcome-based estimates)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sqlalchemy import text

from core.evaluation import latest_runs_per_model_type
from core.policy import build_policy, load_policy_frame, policy_curve, select_targets
from data_sim.ground_truth import GroundTruth, load_ground_truth

MODEL_ORDER = ("baseline", "t_learner", "x_learner")
MODEL_LABELS = {
    "baseline": "Naive (propensity)", "t_learner": "T-learner", "x_learner": "X-learner",
}
SEGMENT_ORDER = ("persuadable", "sure_thing", "lost_cause", "sleeping_dog")


def list_pilots(engine) -> pd.DataFrame:
    """One row per pilot with what it supports, newest first."""
    with engine.connect() as conn:
        return pd.read_sql(
            text(
                "SELECT p.pilot_id, p.name, p.start_date, "
                "  (SELECT COUNT(*) FROM treatment_assignment t "
                "   WHERE t.pilot_id = p.pilot_id) AS n_users, "
                "  (SELECT COUNT(DISTINCT model_type) FROM model_runs m "
                "   WHERE m.pilot_id = p.pilot_id) AS n_model_types, "
                "  (SELECT COUNT(*) FROM user_economics e "
                "   WHERE e.pilot_id = p.pilot_id) AS n_economics, "
                "  CASE WHEN p.ground_truth_json IS NULL THEN 0 ELSE 1 END AS simulated "
                "FROM pilots p ORDER BY p.start_date DESC, p.pilot_id"
            ),
            conn,
        )


def ground_truth_or_none(engine, pilot_id: str) -> GroundTruth | None:
    """The pilot's stored truth if it is simulated AND has revenue truth."""
    try:
        gt = load_ground_truth(engine, pilot_id)
    except ValueError:
        return None
    return gt if gt.revenue_means else None


@dataclass
class PolicyView:
    """Everything the policy page needs for one model. Curves are the
    expensive part (bootstrap per budget point), so they are built once per
    (pilot, rate, floor); picking users at a budget is a cheap prefix walk."""

    model_type: str
    curve: pd.DataFrame  # core.policy.policy_curve (+ true_net when simulated)
    inputs: pd.DataFrame  # core.policy.policy_inputs, for selection at any budget
    min_roi: float | None

    def selected(self, budget_pct: float) -> list[str]:
        budget = budget_pct * float(self.inputs["expected_cost"].sum())
        return select_targets(self.inputs, budget, min_roi=self.min_roi)


def policy_views(
    engine,
    pilot_id: str,
    cashback_rate: float,
    min_roi: float | None,
    n_bootstrap: int = 300,
) -> list[PolicyView]:
    """Policy curve per model, with TRUE net added for simulated pilots."""
    gt = ground_truth_or_none(engine, pilot_id)
    users = _pilot_segments(engine, pilot_id) if gt is not None else None
    views = []
    for run_id in latest_runs_per_model_type(engine, pilot_id):
        frame = load_policy_frame(engine, run_id)
        model_type = frame["model_type"].iloc[0]
        inputs = build_policy(frame, cashback_rate)
        floor = None if model_type == "baseline" else min_roi
        curve = policy_curve(
            inputs, min_roi=floor, outcomes=frame[["user_id", "treated", "revenue"]],
            cashback_rate=cashback_rate, n_bootstrap=n_bootstrap,
        )
        if gt is not None:
            curve["true_net"] = [
                _true_net(select_targets(inputs, b, min_roi=floor), users, gt, cashback_rate)
                for b in curve["budget"]
            ]
        views.append(PolicyView(model_type, curve, inputs, floor))
    return sorted(views, key=lambda v: MODEL_ORDER.index(v.model_type))


def random_true_net(engine, pilot_id: str, cashback_rate: float, budgets) -> np.ndarray | None:
    """
    Expected TRUE net of random targeting at each absolute budget: a random
    user costs the average true cost, so a budget B treats a B / (true cost
    of everyone) share -- capped at everyone. Budgets are absolute so this
    lines up with the policy curves, whose budget_pct is relative to the
    ESTIMATED cost of treating everyone.
    """
    gt = ground_truth_or_none(engine, pilot_id)
    if gt is None:
        return None
    seg = _pilot_segments(engine, pilot_id, test_only=True)["segment"]
    true_cost = cashback_rate * seg.map(gt.expected_treated_revenue).sum()
    true_net = seg.map(gt.true_revenue_uplift).sum() - true_cost
    share = np.minimum(np.asarray(budgets, dtype=float) / true_cost, 1.0)
    return share * true_net


def segment_mix(engine, pilot_id: str, selected: list[str]) -> pd.Series:
    """Share of each true segment among selected users (simulated pilots)."""
    users = _pilot_segments(engine, pilot_id).set_index("user_id")
    if not selected:
        return pd.Series(0.0, index=list(SEGMENT_ORDER))
    shares = users.loc[selected, "segment"].value_counts(normalize=True)
    return shares.reindex(SEGMENT_ORDER, fill_value=0.0)


def _pilot_segments(engine, pilot_id: str, test_only: bool = False) -> pd.DataFrame:
    sql = (
        "SELECT u.user_id, u.segment FROM users u "
        "JOIN treatment_assignment t ON t.user_id = u.user_id AND t.pilot_id = :p"
    )
    if test_only:
        sql += (
            " JOIN data_splits s ON s.user_id = u.user_id AND s.pilot_id = :p "
            "AND s.split = 'test'"
        )
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params={"p": pilot_id})


def _true_net(selected, users: pd.DataFrame, gt: GroundTruth, cashback_rate: float) -> float:
    if not selected:
        return 0.0
    seg = users.set_index("user_id").loc[selected, "segment"]
    cost = cashback_rate * seg.map(gt.expected_treated_revenue)
    return float((seg.map(gt.true_revenue_uplift) - cost).sum())


def qini_view(engine, pilot_id: str):
    """(scores, curves) from persisted evaluations, or (None, None) if absent."""
    run_ids = latest_runs_per_model_type(engine, pilot_id)
    params = {f"r{i}": r for i, r in enumerate(run_ids)}
    marks = ", ".join(f":{k}" for k in params)
    with engine.connect() as conn:
        scores = pd.read_sql(
            text(
                "SELECT m.model_type, e.* FROM model_evaluations e "
                f"JOIN model_runs m ON m.model_run_id = e.model_run_id "
                f"WHERE e.model_run_id IN ({marks})"
            ),
            conn,
            params=params,
        )
        curves = pd.read_sql(
            text(
                "SELECT m.model_type, q.pct_targeted, q.cumulative_incremental_gain, "
                "q.ci_lower, q.ci_upper FROM qini_curve_points q "
                "JOIN model_runs m ON m.model_run_id = q.model_run_id "
                f"WHERE q.model_run_id IN ({marks}) ORDER BY m.model_type, q.pct_targeted"
            ),
            conn,
            params=params,
        )
    if scores.empty:
        return None, None
    return scores, curves


REFERENCE_LABELS = {
    "random": "Random targeting", "x_ceiling": "Best possible from covariates",
    "segment_oracle": "Segment oracle (unattainable)",
}


def qini_table(scores: pd.DataFrame) -> pd.DataFrame:
    """Stored evaluations as a readable table, in model order."""
    ordered = scores.set_index("model_type").loc[
        [m for m in MODEL_ORDER if m in set(scores["model_type"])]
    ]
    return pd.DataFrame({
        "model": [MODEL_LABELS[m] for m in ordered.index],
        "Qini coefficient [95% CI]": [
            f"{r.qini_coefficient:.1f} [{r.qini_ci_lower:.1f}, {r.qini_ci_upper:.1f}]"
            for r in ordered.itertuples()
        ],
        "AUUC [95% CI]": [
            f"{r.auuc:.1f} [{r.auuc_ci_lower:.1f}, {r.auuc_ci_upper:.1f}]"
            for r in ordered.itertuples()
        ],
        "test users": ordered["n_users"].to_numpy(),
    })


def ranking_table(oracle: pd.DataFrame) -> pd.DataFrame:
    """The harness's oracle metrics with readable row and column names."""
    labels = {**MODEL_LABELS, **REFERENCE_LABELS}
    table = oracle.rename(index=labels).rename(columns={
        "top_10%": "top 10%", "top_20%": "top 20%", "top_30%": "top 30%", "top_50%": "top 50%",
        "whole_curve": "whole-curve area", "lift_share": "share of achievable lift",
        "spearman": "rank corr. with E[tau|X]", "rmse": "RMSE vs E[tau|X]",
    })
    return table.round(3)


def headline(views: list[PolicyView], budget_pct: float) -> pd.DataFrame:
    """The policy table at the chosen budget: one row per model."""
    rows = []
    for v in views:
        point = v.curve.iloc[int(np.argmin(np.abs(v.curve["budget_pct"] - budget_pct)))]
        rows.append({
            "model": MODEL_LABELS[v.model_type],
            "users targeted": int(point["n_targeted"]),
            "planned spend": point["expected_spend"],
            "A/B net": point["ab_net_revenue"],
            "A/B 95% CI low": point["ab_net_ci_lower"],
            "A/B 95% CI high": point["ab_net_ci_upper"],
            "model-claimed net": point["model_net_revenue"],
            **({"true net": point["true_net"]} if "true_net" in v.curve else {}),
        })
    return pd.DataFrame(rows)
