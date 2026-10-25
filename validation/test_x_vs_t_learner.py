"""
Phase 4's claim, tested against the right target: the X-learner estimates
CATE more accurately than the T-learner, measured as RMSE against the true
E[tau | X] (data_sim/ground_truth.py) -- the best estimate any model could
make from the observable covariates.

What this does NOT claim: that the X-learner always TARGETS better. The
validation harness (reports/validation_report.md) shows X beating T on
RMSE in every seed of both scenarios, while top-decile targeting is mixed
-- its stage-2 regression smooths the estimate, lowering error overall but
flattening the top of the ranking. A better estimator is not
automatically a better top-k targeter.

Marked slow (uses the session-scoped full-size harness runs).
"""

from __future__ import annotations

import pytest

from validation.harness import oracle_metrics


@pytest.mark.slow
@pytest.mark.parametrize("run_name", ["balanced_run", "imbalanced_run"])
def test_x_learner_estimates_cate_more_accurately_than_t_learner(run_name, request):
    run = request.getfixturevalue(run_name)
    m = oracle_metrics(run.test_users, run.ground_truth, run.scores)
    rmse_t, rmse_x = m.loc["t_learner", "rmse"], m.loc["x_learner", "rmse"]
    print(f"\n{run.scenario}: RMSE vs E[tau|X] -- T-learner {rmse_t:.4f}, X-learner {rmse_x:.4f}")
    assert rmse_x < rmse_t
