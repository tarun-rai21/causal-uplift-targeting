import pandas as pd

from data_sim.ground_truth import GroundTruth
from validation.oracle_evaluation import permutation_test_against_random, true_incremental_outcome

GT = GroundTruth(
    baseline_rate=0.1,
    propensity=0.5,
    segment_mix={"persuadable": 0.5, "lost_cause": 0.5},
    segment_effects={"persuadable": 0.2, "lost_cause": 0.0},
)
USERS = pd.DataFrame(
    {
        "user_id": [f"u{i}" for i in range(100)],
        "segment": ["persuadable"] * 10 + ["lost_cause"] * 90,
    }
)


def test_true_incremental_outcome_sums_segment_uplift():
    assert true_incremental_outcome(["u0", "u1", "u50"], USERS, GT) == 0.4


def test_perfect_list_gets_small_but_nonzero_p_value():
    """The 10 persuadables are the best possible 10-user list; random draws
    essentially never match it. The (b+1)/(m+1) correction must still keep
    the reported p-value strictly positive."""
    result = permutation_test_against_random(
        USERS["user_id"][:10], USERS["user_id"], USERS, GT, n_resamples=200, seed=0
    )
    assert result["observed_incremental_outcome"] == 2.0
    assert 0 < result["p_beats_random"] <= 1 / 201 + 1e-12
