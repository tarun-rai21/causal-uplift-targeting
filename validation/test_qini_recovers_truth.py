"""
Does Qini -- which uses only observed outcomes, never ground truth -- rank
targeting rules the same way the truth does? That is the property that
makes core/evaluation.py trustworthy on REAL pilots, where no oracle exists.

Fixed scoring rules (no model fitting, so the only noise is outcome
sampling) on a 250k-user simulated pilot, ordered by TRUE whole-curve area:
    E[tau|X] ceiling  >  historical_spend  >  random  >  -E[tau|X]

Why the pilot is so large -- a finding, not an implementation detail: on
the default 50k pilot's 10k-user test split, Qini intervals are about
+/-20 while real gaps between trained models are ~5-10, and across
simulator seeds 1-5 the Qini point estimate disagreed with the true
whole-curve ranking of baseline vs. T-learner in 3/5 runs. Outcome-based
evaluation needs far more data than oracle evaluation to separate close
models. Report the paired-difference intervals, not point rankings.

Marked slow (~40s: 200 paired bootstrap draws over four 250k-user curves).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config.schema import load_pilot_config
from core.evaluation import evaluate_scores
from data_sim.pilot_simulator import PilotSimulator

_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "default_config.yaml"


def _true_area(tau: np.ndarray, user_ids: np.ndarray, score: np.ndarray) -> float:
    """Whole-curve area above random using TRUE uplift (oracle analogue of AUUC)."""
    order = np.lexsort((user_ids, -score))
    gain = np.concatenate([[0.0], np.cumsum(tau[order])])
    return float((gain - np.arange(len(tau) + 1) / len(tau) * gain[-1]).mean())


@pytest.mark.slow
def test_qini_ranks_targeting_rules_like_the_truth():
    cfg = load_pilot_config(str(_DEFAULT_CONFIG)).model_copy(
        update={"n_users": 250_000, "seed": 9}
    )
    users, treatment, outcomes, gt = PilotSimulator(cfg).generate()
    ceiling = gt.true_cate_given_covariates(users).to_numpy()
    scores = {
        "ceiling": ceiling,
        "spend": users["historical_spend"].to_numpy(),
        "random": np.random.default_rng(0).random(len(users)),
        "anti_ceiling": -ceiling,
    }
    tau = users["segment"].map(gt.segment_effects).to_numpy()
    ids = users["user_id"].to_numpy()
    true_order = sorted(scores, key=lambda m: -_true_area(tau, ids, scores[m]))
    assert true_order == ["ceiling", "spend", "random", "anti_ceiling"]

    frame = pd.DataFrame({
        "user_id": ids,
        "treated": treatment["treated"].to_numpy(),
        "converted": outcomes["converted"].to_numpy(),
        **scores,
    })
    results, diffs = evaluate_scores(frame, list(scores), n_bootstrap=200, seed=1)
    for m in scores:
        q = results[m]["qini"]
        print(f"\n{m:<13} qini={q.estimate:8.1f} [{q.lower:7.1f}, {q.upper:7.1f}]", end="")

    assert sorted(scores, key=lambda m: -results[m]["qini"].estimate) == true_order
    assert results["ceiling"]["qini"].lower > 0
    assert results["anti_ceiling"]["qini"].upper < 0
    assert diffs[("ceiling", "spend")].lower > 0
    assert diffs[("random", "anti_ceiling")].lower > 0
