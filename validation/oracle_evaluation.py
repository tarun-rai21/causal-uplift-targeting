"""
Ground-truth ("oracle") evaluation of a target list -- ONLY valid on
simulated data where each user's true segment (and therefore true uplift)
is known. This is what lets Phase 2's naive-vs-random proof, and Phase 6's
formal validation harness, compute a target list's TRUE incremental
outcome directly.

Design note (why a permutation test, not a plain before/after comparison):
comparing the naive list's true incremental outcome against a SINGLE
random list's true incremental outcome gives two deterministic numbers --
there's nothing stochastic to build a p-value from. The randomness that
makes "statistically indistinguishable from random" a checkable claim is
in WHICH k users happen to land in a random draw, not in re-simulating
outcomes (that would reintroduce the exact Bernoulli sampling noise this
approach exists to avoid). So: draw many random k-sized subsets, build an
empirical null distribution of their true incremental outcomes, and locate
the naive list's actual value within it.

NOT usable by Phase 5's Qini/AUUC evaluation module, which must work
generically on any model's predictions -- including, eventually, real,
non-simulated pilot data where no such oracle exists.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from data_sim.ground_truth import GroundTruth


def _build_uplift_lookup(users_df: pd.DataFrame, ground_truth: GroundTruth) -> pd.Series:
    """
    Precompute a user_id -> true_uplift Series once, so repeated calls
    (e.g. every draw inside a permutation test's resample loop) don't
    rebuild a segment->uplift mapping or re-call ground_truth.true_uplift
    per row each time.
    """
    segment_uplift = {s: ground_truth.true_uplift(s) for s in users_df["segment"].unique()}
    return users_df.set_index("user_id")["segment"].map(segment_uplift)


def true_incremental_outcome(
    user_ids: Iterable[str],
    users_df: pd.DataFrame,
    ground_truth: GroundTruth,
) -> float:
    """
    Sum of true per-user uplift over the given user_ids -- the TRUE
    incremental conversions this target list would cause if exactly these
    users were treated, per the simulator's known ground truth.
    Deterministic given the same user_ids (no sampling involved).
    """
    lookup = _build_uplift_lookup(users_df, ground_truth)
    return float(lookup.loc[list(user_ids)].sum())


def covariate_ceiling_target_ids(
    users_df: pd.DataFrame, ground_truth: GroundTruth, k: int
) -> list[str]:
    """
    The best k-user target list achievable from observable covariates: the
    top k by true E[tau | X]. Its true_incremental_outcome is the ceiling a
    CATE model should be judged against -- NOT the segment-oracle top k,
    which ranks by hidden segment membership no model can observe.
    """
    if not 0 < k <= len(users_df):
        raise ValueError(f"k must be in [1, {len(users_df)}], got {k}")
    cate_x = ground_truth.true_cate_given_covariates(users_df)
    return cate_x.sort_values(ascending=False, kind="stable").head(k).index.tolist()


def permutation_test_against_random(
    target_user_ids: Iterable[str],
    candidate_pool_ids: Iterable[str],
    users_df: pd.DataFrame,
    ground_truth: GroundTruth,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Empirical permutation test: is `target_user_ids`'s true incremental
    outcome distinguishable from a same-sized RANDOM draw from
    candidate_pool_ids?

    Returns a dict with the observed value, the null distribution's mean/
    std, a two-sided empirical p-value, and a one-sided p_beats_random
    (fraction of random draws that did at least as well as the target
    list -- the directionally-correct metric for "does naive outperform
    random").
    """
    target_user_ids = list(target_user_ids)
    k = len(target_user_ids)
    pool = np.array(list(candidate_pool_ids))

    if k > len(pool):
        raise ValueError(f"target list size {k} exceeds candidate pool size {len(pool)}")

    # Built ONCE, reused for the observed value and every resample -- avoids
    # rebuilding the segment->uplift map n_resamples times.
    uplift_lookup = _build_uplift_lookup(users_df, ground_truth)

    observed = float(uplift_lookup.loc[target_user_ids].sum())

    rng = np.random.default_rng(seed)
    null_distribution = np.empty(n_resamples)
    for i in range(n_resamples):
        random_draw = rng.choice(pool, size=k, replace=False)
        null_distribution[i] = uplift_lookup.loc[random_draw].sum()

    null_mean = float(null_distribution.mean())
    null_std = float(null_distribution.std())

    # Both p-values use the (b + 1) / (m + 1) Monte Carlo correction
    # (Phipson & Smyth, 2010): the observed list counts as one member of its
    # own null, so a p-value can never be reported as exactly 0 from a
    # finite number of resamples.
    observed_abs_dev = abs(observed - null_mean)
    null_abs_dev = np.abs(null_distribution - null_mean)
    p_value_two_sided = _mc_p_value(null_abs_dev >= observed_abs_dev)

    # One-sided: fraction of random draws that did AT LEAST AS WELL as the
    # target list. This is the actually-relevant claim for Phase 2's
    # thesis -- "naive targeting does not OUTPERFORM random" is a
    # one-sided claim. Naive scoring equal to or worse than random both
    # support the thesis; only a SMALL p_beats_random (naive unusually
    # good vs. the null) would refute it. A two-sided test could, in
    # principle, fail on a result (naive significantly WORSE than random)
    # that actually supports the thesis even more strongly -- that would
    # be a confusing, wrong test failure, so this one-sided value is what
    # the actual assertion should check, not p_value_two_sided.
    p_beats_random = _mc_p_value(null_distribution >= observed)

    return {
        "observed_incremental_outcome": observed,
        "null_mean": null_mean,
        "null_std": null_std,
        "p_value_two_sided": p_value_two_sided,
        "p_beats_random": p_beats_random,
        "n_resamples": n_resamples,
    }


def _mc_p_value(at_least_as_extreme: np.ndarray) -> float:
    return float((at_least_as_extreme.sum() + 1) / (len(at_least_as_extreme) + 1))
