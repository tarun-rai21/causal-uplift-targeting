"""
Places the T-learner on the right scale: between random targeting and the
covariate-achievable ceiling (top decile by true E[tau | X]) -- not against
the segment oracle, which ranks by hidden segment membership no model can
observe (see data_sim/ground_truth.py).

Marked slow (fits two GradientBoostingClassifiers on a full-size pilot).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from config.schema import load_pilot_config
from core.data_split import compute_data_split, subset_to_split
from core.t_learner import fit_t_learner
from data_sim.pilot_simulator import PilotSimulator
from validation.oracle_evaluation import covariate_ceiling_target_ids, true_incremental_outcome

_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "default_config.yaml"


@pytest.mark.slow
def test_t_learner_beats_random_and_stays_under_covariate_ceiling():
    cfg = load_pilot_config(str(_DEFAULT_CONFIG))
    users_df, treatment_df, outcomes_df, gt = PilotSimulator(cfg).generate()
    split_df = compute_data_split(
        treatment_df, outcomes_df, pilot_id="validation_run", test_size=0.2, split_seed=1
    )
    train = subset_to_split(split_df, "train", users_df, treatment_df, outcomes_df)
    (test_users,) = subset_to_split(split_df, "test", users_df)

    cate = fit_t_learner(*train, random_state=cfg.seed).predict_cate(test_users)
    k = int(0.10 * len(test_users))
    t_learner = true_incremental_outcome(cate.nlargest(k).index, test_users, gt)
    ceiling = true_incremental_outcome(
        covariate_ceiling_target_ids(test_users, gt, k), test_users, gt
    )
    random_mean = k * test_users["segment"].map(gt.segment_effects).mean()
    segment_oracle = test_users["segment"].map(gt.segment_effects).nlargest(k).sum()

    print(
        f"\nTop-decile true incremental conversions -- random: {random_mean:.1f}, "
        f"T-learner: {t_learner:.1f}, X-ceiling: {ceiling:.1f}, "
        f"segment oracle (unattainable): {segment_oracle:.1f}\n"
        f"T-learner captures {(t_learner - random_mean) / (ceiling - random_mean):.0%} "
        "of the achievable lift over random"
    )

    assert t_learner > random_mean
    # The ceiling is the expected-value optimum; a model can exceed it on a
    # finite sample only by luck, so allow ~1 null std (~2 conversions).
    assert t_learner <= ceiling + 2.5
