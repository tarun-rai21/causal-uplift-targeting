"""
Budget-constrained targeting under a cashback-on-conversion incentive
(Phase 7). Consumes per-user estimates -- predicted uplift from any model,
plus the shared economics from core/economics.py -- and contains no
modeling logic itself, so the objective can change without touching any
causal estimation code.

Economics of treating user i at cashback rate r (tau = conversion uplift,
p1 = P(convert | treated), m = revenue per conversion):

    expected cost              r * p1 * m       paid on EVERY treated conversion
    expected incr. revenue     tau * m
    ROI                        tau / (r * p1)   (m cancels)

Ranking: by estimated ROI, then walk down the ranking until estimated spend
reaches the budget. For a spend cap this is the greedy fractional-knapsack
order -- best incremental revenue per unit of cashback first. It beats
ranking by uplift or by net value under every constrained budget tried
(see reports/validation_report.md) because it avoids sure_things: they
convert anyway, so they collect cashback while adding almost nothing,
and each one displaces several cheaper, responsive users.

min_roi optionally removes users below a floor (1.0 = break-even: only
treat users expected to return at least their cashback). The selected set
is always a PREFIX of the ranking: the walk stops at the first user who no
longer fits the budget rather than skipping ahead to cheaper ones, which
keeps the marginal-ROI curve monotone and the policy easy to explain.

Planned vs. paced spend: the walk here stops where ESTIMATED spend reaches
the budget, which is the right plan for a fixed, up-front target list --
but such a list systematically UNDER-spends. Selecting users by estimated
ROI favors users whose uplift was over-estimated, and the same noise
over-estimates their cost (the T-learner's tau and the economics' p1 share
one fitted mu1): a winner's curse. Across all users p1 is unbiased; among
selected users it runs high (about +0.04 to +0.06 at e=0.2 in the
validation harness). A rollout that keeps treating in rank order until
REALIZED cashback hits the budget avoids the shortfall, so deploy the
ranking with spend pacing and treat n_targeted as a lower bound.

Valuing a policy -- models choose WHO, outcomes say WHAT IT IS WORTH. A
model's own sum of tau_hat * m_hat over the users it selected is badly
optimistic: the users were picked BECAUSE their tau_hat was high, so the
sum inherits that selection (3-4x the true net at a 10% budget on the demo
pilot). Those numbers are kept only as model_* columns. The honest value
comes from the evaluation set's randomized outcomes: the models never saw
those users, and within any set chosen from covariates alone treatment is
still randomized, so for a selected set S of n users

    incremental revenue  ~=  n * (mean revenue | S, treated - mean revenue | S, control)
    cashback spend       ~=  n * r * (mean revenue | S, treated)

are unbiased (ab_* columns), with a bootstrap interval on net. The
validation harness checks that interval's coverage against the truth.

The naive policy is the same walk with the baseline's propensity score as
the ranking; it has no uplift estimate, so its model_* revenue columns are
missing rather than invented -- its ab_* columns value it like any other.

All quantities are on the evaluation set (the persisted test split);
budget_pct is budget / (estimated cost of treating every user in it), so
curves are scale-free and comparable across pilots.
"""

from __future__ import annotations

import argparse
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sqlalchemy import text

BUDGET_GRID = np.round(np.linspace(0.0, 1.0, 21), 2)  # budget_pct points per curve
_MIN_P = 1e-9  # guards the ROI denominator; p1 is a probability, never truly 0


def _check_rate(cashback_rate: float) -> None:
    if not 0.0 < cashback_rate <= 1.0:
        raise ValueError(f"cashback_rate must be in (0, 1], got {cashback_rate}")


def _fits(budget: float) -> float:
    """Budget plus a relative float tolerance: summed costs like 0.05 + 0.92
    give 0.9700000000000001, and spend exactly equal to the budget must fit."""
    return budget * (1 + 1e-9) + 1e-12


def policy_inputs(
    user_ids,
    ranking_score,
    p_convert_if_treated,
    revenue_per_conversion,
    cashback_rate: float,
    uplift=None,
) -> pd.DataFrame:
    """
    Assemble per-user policy inputs. uplift (conversion-probability CATE)
    is None for the naive policy, which has no causal estimate.
    """
    _check_rate(cashback_rate)
    p1 = np.asarray(p_convert_if_treated, dtype=float)
    m = np.asarray(revenue_per_conversion, dtype=float)
    frame = pd.DataFrame({
        "user_id": np.asarray(user_ids),
        "ranking_score": np.asarray(ranking_score, dtype=float),
        "expected_cost": cashback_rate * p1 * m,
    })
    if uplift is None:
        frame["expected_incremental_revenue"] = np.nan
        frame["roi"] = np.nan
    else:
        tau = np.asarray(uplift, dtype=float)
        frame["expected_incremental_revenue"] = tau * m
        frame["roi"] = tau / (cashback_rate * np.maximum(p1, _MIN_P))
    if frame["user_id"].duplicated().any():
        raise ValueError("duplicate user_ids in policy inputs")
    return frame


def roi_ranking(uplift, p_convert_if_treated, cashback_rate: float) -> np.ndarray:
    """The ranking score uplift-model policies use: estimated ROI."""
    _check_rate(cashback_rate)
    return np.asarray(uplift, dtype=float) / (
        cashback_rate * np.maximum(np.asarray(p_convert_if_treated, dtype=float), _MIN_P)
    )


def outcome_estimate(
    treated, revenue, cashback_rate: float, n_bootstrap: int = 500,
    ci_level: float = 0.95, seed: int = 42,
) -> dict[str, float]:
    """
    Randomized-outcome estimate of treating ALL of a selected set: per-arm
    revenue means scaled to the set's size. NaN when either arm is empty.
    """
    _check_rate(cashback_rate)
    t = np.asarray(treated, dtype=bool)
    y = np.asarray(revenue, dtype=float)
    nan = dict.fromkeys(
        ("incremental_revenue", "spend", "net", "net_lower", "net_upper"), float("nan")
    )
    if t.all() or not t.any():
        return nan

    def estimate(t_, y_):
        n_t = t_.sum(axis=-1)
        n_c = t_.shape[-1] - n_t
        mean_t = (y_ * t_).sum(axis=-1) / np.where(n_t > 0, n_t, np.nan)
        mean_c = (y_ * ~t_).sum(axis=-1) / np.where(n_c > 0, n_c, np.nan)
        n = t_.shape[-1]
        return n * (mean_t - mean_c), n * cashback_rate * mean_t

    incr, spend = estimate(t, y)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(t), size=(n_bootstrap, len(t)))
    b_incr, b_spend = estimate(t[idx], y[idx])
    alpha = (1 - ci_level) / 2
    lower, upper = np.nanquantile(b_incr - b_spend, [alpha, 1 - alpha])
    return {
        "incremental_revenue": float(incr), "spend": float(spend), "net": float(incr - spend),
        "net_lower": float(lower), "net_upper": float(upper),
    }


def _ranked(inputs: pd.DataFrame, min_roi: float | None) -> pd.DataFrame:
    if min_roi is not None:
        if inputs["roi"].isna().all():
            raise ValueError("min_roi needs an uplift estimate; the naive policy has none")
        inputs = inputs[inputs["roi"] >= min_roi]
    order = np.lexsort((inputs["user_id"].to_numpy(), -inputs["ranking_score"].to_numpy()))
    return inputs.iloc[order].reset_index(drop=True)


def select_targets(
    inputs: pd.DataFrame, budget: float, min_roi: float | None = None
) -> list[str]:
    """The longest prefix of the ranking whose estimated spend fits the budget."""
    if budget < 0:
        raise ValueError(f"budget must be >= 0, got {budget}")
    ranked = _ranked(inputs, min_roi)
    k = int(np.searchsorted(
        ranked["expected_cost"].cumsum().to_numpy(), _fits(budget), side="right"
    ))
    return ranked["user_id"].iloc[:k].tolist()


def policy_curve(
    inputs: pd.DataFrame,
    min_roi: float | None = None,
    grid=BUDGET_GRID,
    outcomes: pd.DataFrame | None = None,
    cashback_rate: float | None = None,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """
    The policy at each budget_pct in grid. expected_spend is the planning
    estimate the walk stops on; model_* are the model's own (optimistic)
    valuations; model_marginal_roi is d(model revenue) / d(spend) between
    grid points -- the diminishing-returns SHAPE, missing where spend did
    not grow. With outcomes (user_id, treated, revenue) and cashback_rate,
    ab_* add the unbiased randomized-outcome valuation with a CI on net.
    """
    total_cost = float(inputs["expected_cost"].sum())  # cost of treating everyone
    ranked = _ranked(inputs, min_roi)
    cum_cost = np.concatenate([[0.0], ranked["expected_cost"].cumsum().to_numpy()])
    cum_rev = np.concatenate(
        [[0.0], ranked["expected_incremental_revenue"].cumsum().to_numpy()]
    )
    rows = []
    for pct in grid:
        budget = float(pct) * total_cost
        k = int(np.searchsorted(cum_cost, _fits(budget), side="right")) - 1
        rows.append({
            "budget_pct": float(pct),
            "budget": budget,
            "n_targeted": k,
            "expected_spend": cum_cost[k],
            "expected_incremental_revenue": cum_rev[k],
        })
    curve = pd.DataFrame(rows).rename(
        columns={"expected_incremental_revenue": "model_incremental_revenue"}
    )
    curve["model_net_revenue"] = curve["model_incremental_revenue"] - curve["expected_spend"]
    d_rev = curve["model_incremental_revenue"].diff()
    d_spend = curve["expected_spend"].diff()
    curve["model_marginal_roi"] = (d_rev / d_spend).where(d_spend > 0)

    if outcomes is not None:
        if cashback_rate is None:
            raise ValueError("cashback_rate is required to value a policy from outcomes")
        aligned = outcomes.set_index("user_id").loc[ranked["user_id"]]
        treated = aligned["treated"].to_numpy(dtype=bool)
        revenue = aligned["revenue"].to_numpy(dtype=float)
        estimates = [
            outcome_estimate(
                treated[:k], revenue[:k], cashback_rate, n_bootstrap=n_bootstrap, seed=seed
            )
            for k in curve["n_targeted"]
        ]
        for key in ("incremental_revenue", "spend", "net", "net_lower", "net_upper"):
            name = {"net_lower": "ab_net_ci_lower", "net_upper": "ab_net_ci_upper"}.get(
                key, f"ab_{key}" if key != "net" else "ab_net_revenue"
            )
            curve[name] = [e[key] for e in estimates]
    return curve


# --- DB layer ---------------------------------------------------------------


def persist_economics(engine, pilot_id: str, economics: pd.DataFrame) -> None:
    """Replace a pilot's stored economics estimates (one row per test user)."""
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM user_economics WHERE pilot_id = :p"), {"p": pilot_id})
        economics.assign(pilot_id=pilot_id).to_sql(
            "user_economics", conn, if_exists="append", index=False
        )


def load_policy_frame(engine, model_run_id: str) -> pd.DataFrame:
    """A model_run's scores joined to the pilot's stored economics."""
    with engine.connect() as conn:
        frame = pd.read_sql(
            text(
                "SELECT p.user_id, p.predicted_uplift AS score, m.model_type, "
                "e.p_convert_if_treated, e.revenue_per_conversion, t.treated, o.revenue "
                "FROM uplift_predictions p "
                "JOIN model_runs m ON m.model_run_id = p.model_run_id "
                "JOIN user_economics e "
                "  ON e.user_id = p.user_id AND e.pilot_id = m.pilot_id "
                "JOIN treatment_assignment t "
                "  ON t.user_id = p.user_id AND t.pilot_id = m.pilot_id "
                "JOIN outcomes o ON o.user_id = p.user_id AND o.pilot_id = m.pilot_id "
                "WHERE p.model_run_id = :run ORDER BY p.user_id"
            ),
            conn,
            params={"run": model_run_id},
        )
    if frame.empty:
        raise ValueError(
            f"no scored users with economics for model_run_id={model_run_id!r} -- "
            "run core.pipeline (which fits and stores the economics) first"
        )
    frame["treated"] = frame["treated"].astype(bool)
    return frame


def build_policy(frame: pd.DataFrame, cashback_rate: float) -> pd.DataFrame:
    """Uplift models rank by estimated ROI; the naive baseline by its own score."""
    naive = (frame["model_type"] == "baseline").all()
    uplift = None if naive else frame["score"]
    ranking = (
        frame["score"]
        if naive
        else roi_ranking(frame["score"], frame["p_convert_if_treated"], cashback_rate)
    )
    return policy_inputs(
        frame["user_id"], ranking, frame["p_convert_if_treated"],
        frame["revenue_per_conversion"], cashback_rate, uplift=uplift,
    )


def persist_policy_curve(
    engine, model_run_id: str, curve: pd.DataFrame, cashback_rate: float,
    min_roi: float | None,
) -> str:
    curve_id = str(uuid.uuid4())
    rows = curve.assign(
        policy_sim_id=[str(uuid.uuid4()) for _ in range(len(curve))],
        policy_curve_id=curve_id,
        model_run_id=model_run_id,
        cashback_rate=cashback_rate,
        min_roi=min_roi,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    with engine.begin() as conn:
        rows.to_sql("policy_simulations", conn, if_exists="append", index=False)
    return curve_id


def _fmt(x) -> str:
    return "n/a" if pd.isna(x) else f"{x:,.0f}"


def main() -> None:
    from core.evaluation import latest_runs_per_model_type
    from core.pipeline import latest_pilot_id
    from db.connection import get_engine, init_schema

    parser = argparse.ArgumentParser(description="Budget-constrained cashback targeting")
    parser.add_argument("--cashback-rate", type=float, default=0.10)
    parser.add_argument("--min-roi", type=float, help="e.g. 1.0 = break-even floor")
    parser.add_argument("--budget", type=float, help="absolute budget to report targets for")
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument("--pilot-id", help="defaults to the most recently started pilot")
    args = parser.parse_args()

    engine = get_engine()
    init_schema(engine)
    pilot_id = args.pilot_id or latest_pilot_id(engine)
    print(f"pilot {pilot_id}, cashback rate {args.cashback_rate:.0%}"
          + (f", ROI floor {args.min_roi}" if args.min_roi is not None else "") + "\n")
    show = [0.05, 0.10, 0.25, 0.50, 1.00]
    for run_id in latest_runs_per_model_type(engine, pilot_id):
        frame = load_policy_frame(engine, run_id)
        model_type = frame["model_type"].iloc[0]
        inputs = build_policy(frame, args.cashback_rate)
        floor = None if model_type == "baseline" else args.min_roi
        curve = policy_curve(
            inputs, min_roi=floor, outcomes=frame[["user_id", "treated", "revenue"]],
            cashback_rate=args.cashback_rate, n_bootstrap=args.n_bootstrap,
        )
        persist_policy_curve(engine, run_id, curve, args.cashback_rate, floor)

        print(f"{model_type}  (budget scale: {curve['budget'].iloc[-1]:,.0f} = treat everyone)")
        print(f"  {'budget':>7} {'targeted':>9} {'plan spend':>11} {'A/B spend':>10} "
              f"{'A/B net':>8} {'95% CI':>18} {'model-claimed net':>18}")
        for _, r in curve[curve["budget_pct"].isin(show)].iterrows():
            ci = (f"[{_fmt(r.ab_net_ci_lower)}, {_fmt(r.ab_net_ci_upper)}]"
                  if not pd.isna(r.ab_net_ci_lower) else "n/a")
            print(f"  {r.budget_pct:>7.0%} {int(r.n_targeted):>9,} {_fmt(r.expected_spend):>11} "
                  f"{_fmt(r.ab_spend):>10} {_fmt(r.ab_net_revenue):>8} {ci:>18} "
                  f"{_fmt(r.model_net_revenue):>18}")
        if args.budget is not None:
            targets = select_targets(inputs, args.budget, min_roi=floor)
            print(f"  -> budget {args.budget:,.0f}: target {len(targets):,} users")
        print()
    print("A/B = valued from the evaluation set's randomized outcomes: unbiased, but read the")
    print("      CI -- on a ~10k-user holdout it is too wide to separate close policies.")
    print("model-claimed = the model's own estimates over the users it picked; optimistic by")
    print("construction (selection on estimated uplift). Planned lists also tend to")
    print("under-spend -- deploy the ranking with spend paced on realized cashback.")


if __name__ == "__main__":
    main()
