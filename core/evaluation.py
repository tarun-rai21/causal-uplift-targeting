"""
Qini / AUUC evaluation with paired bootstrap confidence intervals (Phase 5).

Unlike validation/oracle_evaluation.py, nothing here uses ground truth: it
needs only each test user's score, treatment flag, and observed outcome,
so it works unchanged on real pilot data. That is the whole point -- the
oracle exists to check that THIS module ranks models the same way the
truth does.

Curves (users sorted by score, descending; t = number targeted):

    Qini (Radcliffe 2007):   Q(t) = Y_T(t) - Y_C(t) * n_T(t) / n_C(t)
    Cumulative gain:         G(t) = (Y_T(t) / n_T(t) - Y_C(t) / n_C(t)) * t

Y_T / Y_C are conversions and n_T / n_C counts among the top-t treated /
control users. They differ in SCALE, which matters when reading numbers:
Q counts incremental conversions among the TREATED users in the top t
(about e * t users), while G estimates incremental conversions had ALL t
been treated -- so G, not Q, is comparable to the oracle's
true_incremental_outcome. Where an arm is still empty at small t the
undefined term is taken as 0.

Scores (both match causalml.metrics with normalize=False exactly):

    qini_coefficient = mean over t = 0..N of  Q(t) - (t / N) * Q(N)
    auuc             = mean over t = 0..N of  G(t)

Ordering: score descending, ties broken by user_id, so a curve is a
deterministic function of the data (causalml's unstable sort is not).

Bootstrap: resample test users with replacement and recompute every curve
through the same code path as the point estimate. The resample is PAIRED
-- one set of resampled users per draw, shared by every model -- so a
difference between two models' scores gets its own valid interval. The
intervals reflect evaluation-sample noise only: the fitted models are
held fixed, so training variance is not included.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sqlalchemy import text

GRID = np.linspace(0.0, 1.0, 101)  # pct_targeted points persisted per curve


@dataclass(frozen=True)
class CurveScores:
    qini_coefficient: float
    auuc: float
    qini_at_grid: np.ndarray  # Q(round(p * N)) for p in GRID


@dataclass(frozen=True)
class Interval:
    estimate: float
    lower: float
    upper: float

    def excludes_zero(self) -> bool:
        return self.lower > 0 or self.upper < 0


@dataclass(frozen=True)
class ModelEvaluation:
    model_run_id: str
    model_type: str
    n_users: int
    qini_coefficient: Interval
    auuc: Interval
    qini_curve: pd.DataFrame  # pct_targeted, cumulative_incremental_gain, ci_lower, ci_upper


def qini_curve(treated: np.ndarray, converted: np.ndarray) -> np.ndarray:
    """Q(t) for t = 0..N, for users ALREADY sorted by score descending."""
    n_t, n_c, y_t, y_c = _cumulative_counts(treated, converted)
    control_scaled = np.divide(y_c * n_t, n_c, out=np.zeros_like(y_c), where=n_c > 0)
    return y_t - control_scaled


def gain_curve(treated: np.ndarray, converted: np.ndarray) -> np.ndarray:
    """G(t) for t = 0..N, for users ALREADY sorted by score descending."""
    n_t, n_c, y_t, y_c = _cumulative_counts(treated, converted)
    rate_t = np.divide(y_t, n_t, out=np.zeros_like(y_t), where=n_t > 0)
    rate_c = np.divide(y_c, n_c, out=np.zeros_like(y_c), where=n_c > 0)
    return (rate_t - rate_c) * np.arange(len(n_t))


def score_sorted(treated: np.ndarray, converted: np.ndarray) -> CurveScores:
    """Qini coefficient, AUUC and grid-sampled Qini for pre-sorted users."""
    q = qini_curve(treated, converted)
    n = len(q) - 1
    random_line = np.arange(n + 1) / n * q[-1]
    return CurveScores(
        qini_coefficient=float((q - random_line).mean()),
        auuc=float(gain_curve(treated, converted).mean()),
        qini_at_grid=q[np.rint(GRID * n).astype(int)],
    )


def _cumulative_counts(treated, converted):
    w = np.asarray(treated, dtype=float)
    y = np.asarray(converted, dtype=float)
    zero = np.zeros(1)
    n_t = np.concatenate([zero, np.cumsum(w)])
    n_c = np.concatenate([zero, np.cumsum(1.0 - w)])
    y_t = np.concatenate([zero, np.cumsum(y * w)])
    y_c = np.concatenate([zero, np.cumsum(y * (1.0 - w))])
    return n_t, n_c, y_t, y_c


def evaluate_scores(
    frame: pd.DataFrame,
    score_columns: list[str],
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> tuple[dict[str, dict], dict[tuple[str, str], Interval]]:
    """
    Evaluate several score columns on ONE set of users with a paired bootstrap.

    frame: one row per user with user_id, treated, converted and each score
    column. Returns ({column: {"scores": CurveScores, "qini": Interval,
    "auuc": Interval, "curve_lower": arr, "curve_upper": arr}},
    {(a, b): Interval for qini_coefficient[a] - qini_coefficient[b]}).
    """
    if not 0.0 < ci_level < 1.0:
        raise ValueError(f"ci_level must be in (0, 1), got {ci_level}")
    if frame["user_id"].duplicated().any():
        raise ValueError("frame has duplicate user_ids")
    # The bootstrap draws row POSITIONS, so without a canonical row order the
    # same data passed in a different order would get different intervals.
    frame = frame.sort_values("user_id", kind="stable").reset_index(drop=True)
    treated = frame["treated"].astype(bool).to_numpy()
    converted = frame["converted"].astype(bool).to_numpy()
    if treated.all() or not treated.any():
        raise ValueError("evaluation set must contain both treated and control users")

    # Per-model ordering, as positions into frame: score desc, user_id asc.
    orders = {
        col: np.lexsort((frame["user_id"].to_numpy(), -frame[col].to_numpy()))
        for col in score_columns
    }
    # rank[col][i] = where user i sits in that model's ordering.
    ranks = {col: np.argsort(order) for col, order in orders.items()}

    point = {col: score_sorted(treated[o], converted[o]) for col, o in orders.items()}

    n = len(frame)
    rng = np.random.default_rng(seed)
    boot = {col: {"qini": np.empty(n_bootstrap), "auuc": np.empty(n_bootstrap),
                  "grid": np.empty((n_bootstrap, len(GRID)))} for col in score_columns}
    for b in range(n_bootstrap):
        sample = rng.integers(0, n, size=n)  # shared by every model -> paired
        for col in score_columns:
            # Sorting the resampled users by their rank in this model's
            # ordering reproduces exactly how the model would order the
            # resampled population (duplicates sit next to each other).
            resampled = orders[col][np.sort(ranks[col][sample])]
            s = score_sorted(treated[resampled], converted[resampled])
            boot[col]["qini"][b] = s.qini_coefficient
            boot[col]["auuc"][b] = s.auuc
            boot[col]["grid"][b] = s.qini_at_grid

    alpha = (1.0 - ci_level) / 2.0
    q_lo, q_hi = alpha, 1.0 - alpha

    def interval(estimate, draws) -> Interval:
        lo, hi = np.quantile(draws, [q_lo, q_hi])
        return Interval(float(estimate), float(lo), float(hi))

    results = {
        col: {
            "scores": point[col],
            "qini": interval(point[col].qini_coefficient, boot[col]["qini"]),
            "auuc": interval(point[col].auuc, boot[col]["auuc"]),
            "curve_lower": np.quantile(boot[col]["grid"], q_lo, axis=0),
            "curve_upper": np.quantile(boot[col]["grid"], q_hi, axis=0),
        }
        for col in score_columns
    }
    differences = {
        (a, b): interval(
            point[a].qini_coefficient - point[b].qini_coefficient,
            boot[a]["qini"] - boot[b]["qini"],
        )
        for a in score_columns
        for b in score_columns
        if a != b
    }
    return results, differences


# --- DB layer -------------------------------------------------------------


def load_evaluation_frame(engine, model_run_ids: list[str]) -> pd.DataFrame:
    """
    One row per test user: user_id, treated, converted, and one score column
    per model_run_id. All runs must belong to the same pilot and cover the
    same users -- a paired comparison over different user sets would be
    meaningless, so that is an error rather than a silent inner join.
    """
    if not model_run_ids:
        raise ValueError("no model_run_ids given")
    frames = []
    pilots = set()
    with engine.connect() as conn:
        for run_id in model_run_ids:
            df = pd.read_sql(
                text(
                    "SELECT p.user_id, m.pilot_id, t.treated, o.converted, "
                    "p.predicted_uplift AS score "
                    "FROM uplift_predictions p "
                    "JOIN model_runs m ON m.model_run_id = p.model_run_id "
                    "JOIN treatment_assignment t "
                    "  ON t.user_id = p.user_id AND t.pilot_id = m.pilot_id "
                    "JOIN outcomes o ON o.user_id = p.user_id AND o.pilot_id = m.pilot_id "
                    "WHERE p.model_run_id = :run ORDER BY p.user_id"
                ),
                conn,
                params={"run": run_id},
            )
            if df.empty:
                raise ValueError(f"no predictions found for model_run_id={run_id!r}")
            pilots.update(df["pilot_id"].unique())
            frames.append(df.drop(columns="pilot_id").rename(columns={"score": run_id}))
    if len(pilots) != 1:
        raise ValueError(f"model runs span several pilots: {sorted(pilots)}")
    base = frames[0]
    for other, run_id in zip(frames[1:], model_run_ids[1:]):
        if other["user_id"].tolist() != base["user_id"].tolist():
            raise ValueError(
                f"model_run {run_id!r} was scored on a different user set than "
                f"{model_run_ids[0]!r}; paired evaluation needs identical users"
            )
        base[run_id] = other[run_id].to_numpy()
    base["treated"] = base["treated"].astype(bool)
    base["converted"] = base["converted"].astype(bool)
    return base


def evaluate_model_runs(
    engine,
    model_run_ids: list[str],
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> tuple[list[ModelEvaluation], dict[tuple[str, str], Interval]]:
    """Evaluate runs on their shared test users; differences keyed by run id."""
    frame = load_evaluation_frame(engine, model_run_ids)
    with engine.connect() as conn:
        types = dict(
            conn.execute(
                text("SELECT model_run_id, model_type FROM model_runs")
            ).fetchall()
        )
    results, differences = evaluate_scores(
        frame, model_run_ids, n_bootstrap=n_bootstrap, ci_level=ci_level, seed=seed
    )
    evaluations = [
        ModelEvaluation(
            model_run_id=run_id,
            model_type=types[run_id],
            n_users=len(frame),
            qini_coefficient=r["qini"],
            auuc=r["auuc"],
            qini_curve=pd.DataFrame({
                "pct_targeted": GRID,
                "cumulative_incremental_gain": r["scores"].qini_at_grid,
                "ci_lower": r["curve_lower"],
                "ci_upper": r["curve_upper"],
            }),
        )
        for run_id, r in results.items()
    ]
    return evaluations, differences


def persist_evaluation(
    engine, evaluation: ModelEvaluation, n_bootstrap: int, ci_level: float, seed: int
) -> None:
    """Replace any previous evaluation of this run (re-evaluating is idempotent)."""
    params = {"run": evaluation.model_run_id}
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM qini_curve_points WHERE model_run_id = :run"), params)
        conn.execute(text("DELETE FROM model_evaluations WHERE model_run_id = :run"), params)
        conn.execute(
            text(
                "INSERT INTO model_evaluations (model_run_id, n_users, "
                "qini_coefficient, qini_ci_lower, qini_ci_upper, "
                "auuc, auuc_ci_lower, auuc_ci_upper, "
                "n_bootstrap, ci_level, bootstrap_seed, evaluated_at) VALUES "
                "(:run, :n, :q, :ql, :qu, :a, :al, :au, :nb, :lvl, :seed, :at)"
            ),
            {
                **params,
                "n": evaluation.n_users,
                "q": evaluation.qini_coefficient.estimate,
                "ql": evaluation.qini_coefficient.lower,
                "qu": evaluation.qini_coefficient.upper,
                "a": evaluation.auuc.estimate,
                "al": evaluation.auuc.lower,
                "au": evaluation.auuc.upper,
                "nb": n_bootstrap,
                "lvl": ci_level,
                "seed": seed,
                "at": datetime.now(timezone.utc).isoformat(),
            },
        )
        evaluation.qini_curve.assign(model_run_id=evaluation.model_run_id).to_sql(
            "qini_curve_points", conn, if_exists="append", index=False
        )


def latest_runs_per_model_type(engine, pilot_id: str) -> list[str]:
    """The most recent model_run of each model_type for a pilot."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT m.model_run_id FROM model_runs m "
                "WHERE m.pilot_id = :p AND m.trained_at = ("
                "  SELECT MAX(m2.trained_at) FROM model_runs m2 "
                "  WHERE m2.pilot_id = m.pilot_id AND m2.model_type = m.model_type"
                ") ORDER BY m.model_type"
            ),
            {"p": pilot_id},
        ).fetchall()
    if not rows:
        raise ValueError(f"no model runs for pilot {pilot_id!r} -- run core.pipeline first")
    return [r[0] for r in rows]


def format_report(evaluations, differences) -> str:
    lines = [
        f"{'model':<10} {'qini_coef':>10} {'95% CI':>20} {'AUUC':>8} {'95% CI':>20}",
    ]
    for ev in evaluations:
        q, a = ev.qini_coefficient, ev.auuc
        lines.append(
            f"{ev.model_type:<10} {q.estimate:10.2f} [{q.lower:8.2f}, {q.upper:8.2f}] "
            f"{a.estimate:8.2f} [{a.lower:8.2f}, {a.upper:8.2f}]"
        )
    by_id = {ev.model_run_id: ev.model_type for ev in evaluations}
    lines.append("\nPaired Qini-coefficient differences (row minus column):")
    for (a, b), d in sorted(differences.items(), key=lambda kv: (by_id[kv[0][0]], by_id[kv[0][1]])):
        if by_id[a] < by_id[b]:
            flag = "  *" if d.excludes_zero() else ""
            lines.append(
                f"  {by_id[a]} - {by_id[b]}: {d.estimate:+.2f} "
                f"[{d.lower:+.2f}, {d.upper:+.2f}]{flag}"
            )
    lines.append("  (* = 95% interval excludes 0)")
    return "\n".join(lines)


def main() -> None:
    from core.pipeline import latest_pilot_id
    from db.connection import get_engine, init_schema

    parser = argparse.ArgumentParser(description="Qini/AUUC evaluation of a pilot's models")
    parser.add_argument("--pilot-id", help="defaults to the most recently started pilot")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    engine = get_engine()
    init_schema(engine)
    pilot_id = args.pilot_id or latest_pilot_id(engine)
    run_ids = latest_runs_per_model_type(engine, pilot_id)
    evaluations, differences = evaluate_model_runs(
        engine, run_ids, n_bootstrap=args.n_bootstrap, seed=args.seed
    )
    for ev in evaluations:
        persist_evaluation(engine, ev, args.n_bootstrap, 0.95, args.seed)
    print(f"pilot {pilot_id}, {evaluations[0].n_users} test users\n")
    print(format_report(evaluations, differences))


if __name__ == "__main__":
    main()
