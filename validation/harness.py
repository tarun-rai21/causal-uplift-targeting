"""
Ground-truth validation harness (Phase 6): one standing report that checks
every model, and every claim the README makes about them, against the
simulator's known truth -- repeated over seeds so each claim is reported
as "holds in k/n runs" instead of resting on one lucky draw.

Two modes:

    python -m validation.harness [--seeds 1 2 3 4 5] [--scenarios balanced imbalanced]
        Simulate, split, fit and score in memory, per (scenario, seed).
        The DB is deliberately skipped: tests/test_pipeline.py proves the DB
        path yields bit-identical predictions, so validating in memory
        validates the methods without coupling the report to a database.

    python -m validation.harness --pilot-id PILOT_ID
        Validate the model runs STORED for one pilot against that pilot's
        STORED ground truth (pilots.ground_truth_json).

What each metric is measured against (see data_sim/ground_truth.py):

    top_<d>%       true incremental conversions if the model's top d% of
                   test users were targeted (sum of true segment uplift)
    whole_curve    mean over targeting depths of that cumulative true gain,
                   minus the random line -- the oracle analogue of AUUC
    lift_share     (model - random) / (X-ceiling - random) at the top
                   decile: share of the achievable lift captured
    spearman       rank correlation with E[tau | X]; defined for every
                   model, including the baseline whose score is not a CATE
    rmse           error vs. E[tau | X]; CATE learners only
    qini           observed-outcome Qini coefficient (no oracle), from
                   core/evaluation.py -- the number a real pilot would see

References: random (expected value of a random list), x_ceiling (ranking
by true E[tau | X], the best any model can do from covariates), and
segment_oracle (ranking by hidden segment, unattainable).
"""

from __future__ import annotations

import argparse
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config.schema import load_pilot_config
from core.baseline import fit_propensity_model, predict_propensity
from core.data_split import compute_data_split, subset_to_split
from core.evaluation import evaluate_scores
from core.t_learner import fit_t_learner
from core.x_learner import fit_x_learner
from data_sim.ground_truth import GroundTruth
from data_sim.pilot_simulator import PilotSimulator

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "default_config.yaml"
SCENARIOS = {"balanced": 0.5, "imbalanced": 0.2}  # name -> propensity
DEPTHS = (0.1, 0.2, 0.3, 0.5)
MODELS = ("baseline", "t_learner", "x_learner")
REFERENCES = ("random", "x_ceiling", "segment_oracle")
SPLIT_SEED = 1
TEST_SIZE = 0.2


# --- fitting ----------------------------------------------------------------


@dataclass
class SeedRun:
    """Everything one (scenario, seed) run produced -- held only in-process
    (the slow tests reuse it); workers return the lighter SeedResult."""

    scenario: str
    seed: int
    test_users: pd.DataFrame
    test_treatment: pd.DataFrame
    test_outcomes: pd.DataFrame
    ground_truth: GroundTruth
    scores: dict[str, pd.Series]  # model -> user_id-indexed score on test users


def run_seed(scenario: str, seed: int, config_path: Path = DEFAULT_CONFIG) -> SeedRun:
    cfg = load_pilot_config(str(config_path)).model_copy(
        update={"seed": seed, "propensity": SCENARIOS[scenario]}
    )
    users, treatment, outcomes, gt = PilotSimulator(cfg).generate()
    split = compute_data_split(
        treatment, outcomes, pilot_id="harness", test_size=TEST_SIZE, split_seed=SPLIT_SEED
    )
    train = subset_to_split(split, "train", users, treatment, outcomes)
    test_users, test_treatment, test_outcomes = subset_to_split(
        split, "test", users, treatment, outcomes
    )
    baseline = fit_propensity_model(train[0], train[2], random_state=seed)
    scores = {
        "baseline": predict_propensity(baseline, test_users),
        "t_learner": fit_t_learner(*train, random_state=seed).predict_cate(test_users),
        "x_learner": fit_x_learner(*train, random_state=seed).predict_cate(test_users),
    }
    return SeedRun(scenario, seed, test_users, test_treatment, test_outcomes, gt, scores)


# --- metrics ----------------------------------------------------------------


def _order(user_ids: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Score descending, ties by user_id -- the same rule core/evaluation uses."""
    return np.lexsort((user_ids, -score))


def oracle_metrics(
    test_users: pd.DataFrame, gt: GroundTruth, scores: dict[str, pd.Series]
) -> pd.DataFrame:
    """One row per model + reference, columns as in the module docstring."""
    ids = test_users["user_id"].to_numpy()
    tau = test_users["segment"].map(gt.segment_effects).to_numpy()
    cate_x = gt.true_cate_given_covariates(test_users).to_numpy()
    n = len(ids)

    def ranked(score: np.ndarray) -> dict:
        gain = np.concatenate([[0.0], np.cumsum(tau[_order(ids, score)])])
        row = {f"top_{int(d * 100)}%": gain[int(round(d * n))] for d in DEPTHS}
        row["whole_curve"] = float((gain - np.arange(n + 1) / n * gain[-1]).mean())
        row["spearman"] = float(spearmanr(score, cate_x).statistic)
        return row

    rows = {
        m: ranked(scores[m].loc[ids].to_numpy()) for m in scores
    }
    rows["x_ceiling"] = ranked(cate_x)
    rows["segment_oracle"] = ranked(tau)
    rows["random"] = {f"top_{int(d * 100)}%": d * tau.sum() for d in DEPTHS}
    rows["random"].update({"whole_curve": 0.0, "spearman": 0.0})
    table = pd.DataFrame.from_dict(rows, orient="index")

    top = "top_10%"
    span = table.loc["x_ceiling", top] - table.loc["random", top]
    table["lift_share"] = (table[top] - table.loc["random", top]) / span
    table["rmse"] = np.nan
    for m in ("t_learner", "x_learner"):
        if m in scores:
            table.loc[m, "rmse"] = float(np.sqrt(((scores[m].loc[ids] - cate_x) ** 2).mean()))
    return table


def observed_qini(
    test_users, test_treatment, test_outcomes, scores, n_bootstrap: int, seed: int
):
    frame = pd.DataFrame({
        "user_id": test_users["user_id"].to_numpy(),
        "treated": test_treatment["treated"].to_numpy(),
        "converted": test_outcomes["converted"].to_numpy(),
        **{m: s.loc[test_users["user_id"]].to_numpy() for m, s in scores.items()},
    })
    return evaluate_scores(frame, list(scores), n_bootstrap=n_bootstrap, seed=seed)


# --- claims -----------------------------------------------------------------

# name -> (description, predicate over a SeedResult). Each is a claim made in
# the README; the harness reports how often it holds rather than asserting.
CLAIMS = {
    "naive_not_above_random_top10": (
        "Naive top decile is at or below random (true incremental)",
        lambda r: r.oracle.loc["baseline", "top_10%"] <= r.oracle.loc["random", "top_10%"],
    ),
    "t_beats_naive_top10": (
        "T-learner top decile beats naive",
        lambda r: r.oracle.loc["t_learner", "top_10%"] > r.oracle.loc["baseline", "top_10%"],
    ),
    "x_beats_naive_top10": (
        "X-learner top decile beats naive",
        lambda r: r.oracle.loc["x_learner", "top_10%"] > r.oracle.loc["baseline", "top_10%"],
    ),
    "learners_beat_random_top10": (
        "Both CATE learners' top decile beats random",
        lambda r: all(
            r.oracle.loc[m, "top_10%"] > r.oracle.loc["random", "top_10%"]
            for m in ("t_learner", "x_learner")
        ),
    ),
    "x_lower_rmse_than_t": (
        "X-learner has lower error vs E[tau|X] than T-learner",
        lambda r: r.oracle.loc["x_learner", "rmse"] < r.oracle.loc["t_learner", "rmse"],
    ),
    "naive_beats_learners_whole_curve": (
        "Naive beats both learners on true whole-curve area (a finding, not a goal)",
        lambda r: all(
            r.oracle.loc["baseline", "whole_curve"] > r.oracle.loc[m, "whole_curve"]
            for m in ("t_learner", "x_learner")
        ),
    ),
    "qini_agrees_naive_vs_t": (
        "Observed Qini orders naive vs T-learner the same way as true whole-curve area",
        lambda r: np.sign(r.qini["baseline"] - r.qini["t_learner"])
        == np.sign(
            r.oracle.loc["baseline", "whole_curve"] - r.oracle.loc["t_learner", "whole_curve"]
        ),
    ),
}


@dataclass
class SeedResult:
    scenario: str
    seed: int
    propensity: float
    n_test: int
    oracle: pd.DataFrame
    qini: dict[str, float]
    qini_interval: dict[str, tuple[float, float]]
    claims: dict[str, bool] = field(default_factory=dict)


def evaluate_run(run: SeedRun, n_bootstrap: int = 200) -> SeedResult:
    results, _ = observed_qini(
        run.test_users, run.test_treatment, run.test_outcomes, run.scores,
        n_bootstrap=n_bootstrap, seed=run.seed,
    )
    result = SeedResult(
        scenario=run.scenario,
        seed=run.seed,
        propensity=run.ground_truth.propensity,
        n_test=len(run.test_users),
        oracle=oracle_metrics(run.test_users, run.ground_truth, run.scores),
        qini={m: r["qini"].estimate for m, r in results.items()},
        qini_interval={m: (r["qini"].lower, r["qini"].upper) for m, r in results.items()},
    )
    result.claims = {name: bool(pred(result)) for name, (_, pred) in CLAIMS.items()}
    return result


def _run_and_evaluate(args: tuple[str, int, int]) -> SeedResult:
    scenario, seed, n_bootstrap = args
    return evaluate_run(run_seed(scenario, seed), n_bootstrap=n_bootstrap)


def run_harness(
    seeds: list[int], scenarios: list[str], n_bootstrap: int = 200, jobs: int = 1
) -> list[SeedResult]:
    tasks = [(sc, s, n_bootstrap) for sc in scenarios for s in seeds]
    if jobs <= 1:
        return [_run_and_evaluate(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(_run_and_evaluate, tasks))  # map preserves task order


# --- report -----------------------------------------------------------------


def _fmt(x: float) -> str:
    return "n/a" if pd.isna(x) else f"{x:.3f}" if abs(x) < 1 else f"{x:.1f}"


def _markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for idx, row in df.iterrows():
        lines.append(f"| {idx} | " + " | ".join(_fmt(v) for v in row) + " |")
    return "\n".join(lines)


_COLUMNS = ["top_10%", "top_20%", "top_30%", "top_50%", "whole_curve",
            "lift_share", "spearman", "rmse", "qini"]


def _with_qini(r: SeedResult) -> pd.DataFrame:
    t = r.oracle.copy()
    t["qini"] = pd.Series(r.qini)
    return t[_COLUMNS]


def format_report(results: list[SeedResult], header: str) -> str:
    out = [f"# Validation report\n\n{header}\n"]
    for scenario in dict.fromkeys(r.scenario for r in results):
        runs = [r for r in results if r.scenario == scenario]
        out.append(
            f"## Scenario `{scenario}` (propensity {runs[0].propensity}, "
            f"seeds {[r.seed for r in runs]}, {runs[0].n_test:,} test users each)\n"
        )
        out.append("### Claims\n\n| claim | holds |\n|---|---|")
        for name, (desc, _) in CLAIMS.items():
            k = sum(r.claims[name] for r in runs)
            out.append(f"| {desc} | {k}/{len(runs)} |")
        mean = sum(_with_qini(r) for r in runs) / len(runs)
        order = [*MODELS, *REFERENCES]
        out.append(
            f"\n### Metrics (mean over {len(runs)} seeds)\n\n"
            + _markdown_table(mean.loc[[i for i in order if i in mean.index]])
        )
        out.append("\n### Per-seed top decile and whole-curve area\n")
        per_seed = pd.DataFrame({
            f"seed {r.seed}": {
                **{f"{m} top10": r.oracle.loc[m, "top_10%"] for m in (*MODELS, "random")},
                **{f"{m} curve": r.oracle.loc[m, "whole_curve"] for m in MODELS},
            }
            for r in runs
        })
        out.append(_markdown_table(per_seed) + "\n")
    out.append(
        "Qini is the observed-outcome coefficient (Q-scale, about e x the oracle "
        "G-scale, so compare it across models, not against oracle columns). "
        "lift_share = (model - random) / (x_ceiling - random) at the top decile.\n"
    )
    return "\n".join(out)


def validate_stored_pilot(engine, pilot_id: str, n_bootstrap: int = 200) -> SeedResult:
    """Score a pilot's STORED latest model runs against its STORED truth."""
    from sqlalchemy import text

    from core.evaluation import latest_runs_per_model_type
    from core.model_store import load_predictions
    from core.pipeline import load_pilot_frames
    from data_sim.ground_truth import load_ground_truth

    gt = load_ground_truth(engine, pilot_id)
    users, treatment, outcomes = load_pilot_frames(engine, pilot_id)
    run_ids = latest_runs_per_model_type(engine, pilot_id)
    with engine.connect() as conn:
        types = dict(
            conn.execute(text("SELECT model_run_id, model_type FROM model_runs")).fetchall()
        )
    scores = {types[r]: load_predictions(engine, r) for r in run_ids}
    if set(scores) != set(MODELS):
        raise ValueError(f"pilot {pilot_id!r} lacks runs for {set(MODELS) - set(scores)}")
    test_ids = set.intersection(*(set(s.index) for s in scores.values()))
    subset = [df[df["user_id"].isin(test_ids)].reset_index(drop=True)
              for df in (users, treatment, outcomes)]
    run = SeedRun("stored", gt.seed, *subset, gt, scores)
    return evaluate_run(run, n_bootstrap=n_bootstrap)


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
            cwd=Path(__file__).resolve().parent, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Ground-truth validation harness")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--scenarios", nargs="+", choices=list(SCENARIOS),
                        default=list(SCENARIOS))
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--pilot-id", help="validate a stored pilot instead of simulating")
    parser.add_argument("--out", type=Path, help="also write the markdown report here")
    args = parser.parse_args()

    if args.pilot_id:
        from db.connection import get_engine

        results = [validate_stored_pilot(get_engine(), args.pilot_id, args.n_bootstrap)]
        header = f"Stored pilot `{args.pilot_id}`, commit `{_git_commit()}`."
    else:
        results = run_harness(args.seeds, args.scenarios, args.n_bootstrap, args.jobs)
        header = (
            f"Generated by `python -m validation.harness` at commit `{_git_commit()}` "
            f"from `config/default_config.yaml` (propensity overridden per scenario), "
            f"split_seed={SPLIT_SEED}, test_size={TEST_SIZE}, "
            f"{args.n_bootstrap} bootstrap draws for Qini."
        )
    report = format_report(results, header)
    print(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
