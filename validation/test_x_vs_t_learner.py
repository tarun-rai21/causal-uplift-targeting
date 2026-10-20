"""
Phase 4's claim, tested against the right target: the X-learner estimates
CATE more accurately than the T-learner, measured as RMSE against the true
E[tau | X] (data_sim/ground_truth.py) -- the best estimate any model could
make from the observable covariates.

What this does NOT claim: that the X-learner always TARGETS better. Across
simulator seeds 1-5 during development, X beat T on RMSE in 10/10 runs at
both e=0.5 and e=0.2, but on top-decile true incremental outcome X won 4/5
at e=0.2 and LOST 4/5 at e=0.5. Its stage-2 regression smooths the
estimate -- lower error overall, but a flatter top of the ranking -- so a
better estimator is not automatically a better top-k targeter. Phase 5's
Qini curves are where that trade-off gets measured across the whole curve
rather than at one cutoff.

Marked slow (six gradient-boosting fits per propensity).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from config.schema import load_pilot_config
from core.data_split import compute_data_split, subset_to_split
from core.t_learner import fit_t_learner
from core.x_learner import fit_x_learner
from data_sim.pilot_simulator import PilotSimulator

_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "default_config.yaml"


@pytest.mark.slow
@pytest.mark.parametrize("propensity", [0.5, 0.2])
def test_x_learner_estimates_cate_more_accurately_than_t_learner(propensity):
    cfg = load_pilot_config(str(_DEFAULT_CONFIG)).model_copy(
        update={"propensity": propensity}
    )
    users_df, treatment_df, outcomes_df, gt = PilotSimulator(cfg).generate()
    split_df = compute_data_split(
        treatment_df, outcomes_df, pilot_id="validation_run", test_size=0.2, split_seed=1
    )
    train = subset_to_split(split_df, "train", users_df, treatment_df, outcomes_df)
    (test_users,) = subset_to_split(split_df, "test", users_df)
    truth = gt.true_cate_given_covariates(test_users)

    def rmse(fit) -> float:
        cate = fit(*train, random_state=cfg.seed).predict_cate(test_users)
        return float(np.sqrt(((cate - truth) ** 2).mean()))

    rmse_t, rmse_x = rmse(fit_t_learner), rmse(fit_x_learner)
    print(f"\ne={propensity}: RMSE vs E[tau|X] -- T-learner {rmse_t:.4f}, X-learner {rmse_x:.4f}")
    assert rmse_x < rmse_t
