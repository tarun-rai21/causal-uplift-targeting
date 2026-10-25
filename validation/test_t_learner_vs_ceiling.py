"""
Places the T-learner on the right scale: between random targeting and the
covariate-achievable ceiling (top decile by true E[tau | X]) -- not against
the segment oracle, which ranks by hidden segment membership no model can
observe (see data_sim/ground_truth.py).

Marked slow (uses the session-scoped full-size harness run).
"""

from __future__ import annotations

import pytest

from validation.harness import oracle_metrics


@pytest.mark.slow
def test_t_learner_beats_random_and_stays_under_covariate_ceiling(balanced_run):
    m = oracle_metrics(balanced_run.test_users, balanced_run.ground_truth, balanced_run.scores)
    t_learner = m.loc["t_learner", "top_10%"]
    ceiling = m.loc["x_ceiling", "top_10%"]
    random_mean = m.loc["random", "top_10%"]
    segment_oracle = m.loc["segment_oracle", "top_10%"]

    print(
        f"\nTop-decile true incremental conversions -- random: {random_mean:.1f}, "
        f"T-learner: {t_learner:.1f}, X-ceiling: {ceiling:.1f}, "
        f"segment oracle (unattainable): {segment_oracle:.1f}\n"
        f"T-learner captures {m.loc['t_learner', 'lift_share']:.0%} "
        "of the achievable lift over random"
    )

    assert t_learner > random_mean
    # The ceiling is the expected-value optimum; a model can exceed it on a
    # finite sample only by luck, so allow ~1 null std (~2 conversions).
    assert t_learner <= ceiling + 2.5
