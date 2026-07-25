"""
Baseline (naive) propensity/response model -- the industry-default,
INCORRECT approach this entire project exists to demonstrate the failure
of. Deliberately trains P(convert | X) on POOLED treated+control data,
ignoring treatment entirely.

Kept as a first-class, permanent module (spec Section 5.1) -- not deleted
after comparison. See validation/test_naive_vs_random.py for the actual
falsifiable proof that top-k targeting by this model does not beat random
targeting on TRUE incremental outcome.
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from core.features import build_feature_matrix


def fit_propensity_model(
    users_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    random_state: int = 42,
) -> GradientBoostingClassifier:
    """
    Fit P(convert | X), ignoring treatment entirely.

    users_df/outcomes_df MUST already be filtered to the TRAIN split only
    (Phase 1.5) -- this function does not filter for you, since it has no
    way to know which split a given DataFrame represents.
    """
    features = build_feature_matrix(users_df)
    labels = (
        users_df[["user_id"]]
        .merge(outcomes_df[["user_id", "converted"]], on="user_id", how="left")["converted"]
        .astype(int)
        .values
    )
    model = GradientBoostingClassifier(random_state=random_state)
    model.fit(features, labels)
    return model


def predict_propensity(model: GradientBoostingClassifier, users_df: pd.DataFrame) -> pd.Series:
    """Predicted P(convert|X) for each user, returned as a Series indexed by user_id."""
    features = build_feature_matrix(users_df)
    scores = model.predict_proba(features)[:, 1]
    return pd.Series(scores, index=users_df["user_id"].values, name="predicted_propensity")


def naive_top_k_targets(scores: pd.Series, k: int) -> list[str]:
    """Return the user_ids of the k highest-scoring users."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if k > len(scores):
        raise ValueError(f"k={k} exceeds available population size {len(scores)}")
    return scores.sort_values(ascending=False).head(k).index.tolist()
