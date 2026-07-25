"""
Phase 2's central deliverable: prove -- not assert -- that naive top-k
propensity targeting does not OUTPERFORM random targeting on TRUE
incremental outcome.

Marked slow (trains a GradientBoostingClassifier on a full-size simulated
pilot, runs a 1000-resample permutation test) -- excluded from default CI,
run manually via: pytest -m slow validation/test_naive_vs_random.py -v
"""

from __future__ import annotations

import pytest

from config.schema import load_pilot_config
from core.baseline import fit_propensity_model, naive_top_k_targets, predict_propensity
from core.data_split import compute_data_split
from data_sim.pilot_simulator import PilotSimulator
from validation.oracle_evaluation import permutation_test_against_random


@pytest.mark.slow
def test_naive_top_decile_does_not_beat_random_on_incremental_outcome():
    cfg = load_pilot_config("config/default_config.yaml")
    users_df, treatment_df, outcomes_df, ground_truth = PilotSimulator(cfg).generate()

    split_df = compute_data_split(
        treatment_df, outcomes_df, pilot_id="validation_run", test_size=0.2, split_seed=1
    )
    train_ids = set(split_df.loc[split_df["split"] == "train", "user_id"])
    test_ids = set(split_df.loc[split_df["split"] == "test", "user_id"])

    train_users = users_df[users_df["user_id"].isin(train_ids)].reset_index(drop=True)
    train_outcomes = outcomes_df[outcomes_df["user_id"].isin(train_ids)].reset_index(drop=True)
    test_users = users_df[users_df["user_id"].isin(test_ids)].reset_index(drop=True)

    model = fit_propensity_model(train_users, train_outcomes, random_state=cfg.seed)
    test_scores = predict_propensity(model, test_users)

    k = int(0.10 * len(test_scores))  # top decile
    naive_target_ids = naive_top_k_targets(test_scores, k)

    result = permutation_test_against_random(
        target_user_ids=naive_target_ids,
        candidate_pool_ids=test_users["user_id"].tolist(),
        users_df=test_users,
        ground_truth=ground_truth,
        n_resamples=1000,
        seed=cfg.seed,
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
def test_naive_model_actually_prefers_sure_thing_over_persuadable():
    """
    Direct mechanism check, not just the outcome-level proof above: the
    naive model's top-decile picks should be dominated by sure_thing
    (high baseline conversion, near-zero true uplift) rather than
    persuadable (the segment with real, high true uplift). This is WHY
    the outcome-level test above passes, not just evidence that it does.
    """
    cfg = load_pilot_config("config/default_config.yaml")
    users_df, treatment_df, outcomes_df, _ = PilotSimulator(cfg).generate()

    split_df = compute_data_split(
        treatment_df, outcomes_df, pilot_id="validation_run", test_size=0.2, split_seed=1
    )
    train_ids = set(split_df.loc[split_df["split"] == "train", "user_id"])
    test_ids = set(split_df.loc[split_df["split"] == "test", "user_id"])

    train_users = users_df[users_df["user_id"].isin(train_ids)].reset_index(drop=True)
    train_outcomes = outcomes_df[outcomes_df["user_id"].isin(train_ids)].reset_index(drop=True)
    test_users = users_df[users_df["user_id"].isin(test_ids)].reset_index(drop=True)

    model = fit_propensity_model(train_users, train_outcomes, random_state=cfg.seed)
    test_scores = predict_propensity(model, test_users)
    k = int(0.10 * len(test_scores))
    naive_target_ids = naive_top_k_targets(test_scores, k)

    picked_segments = users_df.set_index("user_id").loc[naive_target_ids, "segment"]
    segment_counts = picked_segments.value_counts(normalize=True)

    print(f"\nNaive top-decile segment composition:\n{segment_counts}")

    assert segment_counts.get("sure_thing", 0.0) > segment_counts.get("persuadable", 0.0), (
        "Expected the naive model to over-select sure_thing relative to "
        "persuadable -- if this fails, the naive-fails demonstration is "
        "not actually exercising its intended failure mode."
    )
