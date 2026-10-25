"""
Phase 2's central deliverable: prove -- not assert -- that naive top-k
propensity targeting does not OUTPERFORM random targeting on TRUE
incremental outcome.

Marked slow (uses the session-scoped full-size harness run, then a
1000-resample permutation test) -- excluded from default CI, run manually
via: pytest -m slow validation/test_naive_vs_random.py -v
"""

from __future__ import annotations

import pytest

from core.baseline import naive_top_k_targets
from validation.oracle_evaluation import permutation_test_against_random


def _naive_top_decile(run) -> list[str]:
    scores = run.scores["baseline"]
    return naive_top_k_targets(scores, int(0.10 * len(scores)))


@pytest.mark.slow
def test_naive_top_decile_does_not_beat_random_on_incremental_outcome(balanced_run):
    result = permutation_test_against_random(
        target_user_ids=_naive_top_decile(balanced_run),
        candidate_pool_ids=balanced_run.test_users["user_id"].tolist(),
        users_df=balanced_run.test_users,
        ground_truth=balanced_run.ground_truth,
        n_resamples=1000,
        seed=balanced_run.seed,
    )

    print(
        f"\nNaive top-decile true incremental outcome: "
        f"{result['observed_incremental_outcome']:.2f}\n"
        f"Random top-decile (mean of {result['n_resamples']} draws): "
        f"{result['null_mean']:.2f} (std={result['null_std']:.2f})\n"
        f"p (naive beats random, one-sided): {result['p_beats_random']:.3f}\n"
        f"p (two-sided difference): {result['p_value_two_sided']:.3f}"
    )

    # The claim being tested is one-sided: "naive does not OUTPERFORM
    # random." Naive scoring equal to OR WORSE than random both support
    # this; only naive scoring unusually well (small p_beats_random) would
    # refute it. A high p_beats_random (most random draws did at least as
    # well as naive) is exactly the expected, thesis-supporting outcome.
    #
    # Statistical caveat: p > 0.10 is a FAILURE TO REJECT "naive beats
    # random", not positive evidence of equivalence. What makes the claim
    # convincing is that the naive list lands at or BELOW the random mean
    # in every seed of the validation harness (reports/validation_report.md),
    # not merely that the p-value clears a threshold.
    assert result["p_beats_random"] > 0.10, (
        "Naive top-decile targeting appears to significantly OUTPERFORM "
        "random targeting on true incremental outcome -- this should NOT "
        "happen given the simulator's segment design (sure_thing has high "
        "control-arm conversion but near-zero true uplift; a naive model "
        "trained on pooled data should chase sure_things, not "
        "persuadables, and therefore should not beat random). If this "
        "assertion fails, check segment_baseline_offsets in the pilot "
        "config -- see config/schema.py rule 7's rationale."
    )


@pytest.mark.slow
def test_naive_model_actually_prefers_sure_thing_over_persuadable(balanced_run):
    """
    Direct mechanism check, not just the outcome-level proof above: the
    naive model's top-decile picks should be dominated by sure_thing
    (high baseline conversion, near-zero true uplift) rather than
    persuadable (the segment with real, high true uplift). This is WHY
    the outcome-level test above passes, not just evidence that it does.
    """
    picked_segments = balanced_run.test_users.set_index("user_id").loc[
        _naive_top_decile(balanced_run), "segment"
    ]
    segment_counts = picked_segments.value_counts(normalize=True)

    print(f"\nNaive top-decile segment composition:\n{segment_counts}")

    assert segment_counts.get("sure_thing", 0.0) > segment_counts.get("persuadable", 0.0), (
        "Expected the naive model to over-select sure_thing relative to "
        "persuadable -- if this fails, the naive-fails demonstration is "
        "not actually exercising its intended failure mode."
    )
