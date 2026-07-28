"""
T-learner (spec Section 7.2): fit mu1(x) = E[Y|X, T=1] on treated-only rows
and mu0(x) = E[Y|X, T=0] on control-only rows, independently; CATE estimate
is tau_hat(x) = mu1(x) - mu0(x).

Kept as a separate file from x_learner.py (spec Section 5.1), even though
they share structure: the X-learner's propensity-weighted combination is
conceptually distinct, and burying both in one branching class would
obscure exactly the mechanism that matters.

Known failure mode (spec 7.2, and why Phase 4 exists): mu1 and mu0 are
estimated with independent errors. Under an imbalanced propensity (e.g.
0.2), the small-arm model is high-variance and its noise passes straight
into tau_hat with nothing forcing the two models to be consistent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from core.features import build_feature_matrix


def _default_base_learner_factory(random_state: int) -> GradientBoostingClassifier:
    # Identical class + params as core/baseline.py's model, deliberately:
    # any Qini difference vs. baseline must come from the estimand, not
    # from model capacity.
    return GradientBoostingClassifier(random_state=random_state)


@dataclass
class TLearner:
    """
    Holds the two fitted arm models. Construct via fit_t_learner(); the
    dataclass itself is a dumb container so tests can also construct it
    directly with deterministic stand-in models.
    """

    mu1: object  # model fit on treated rows, exposes predict_proba
    mu0: object  # model fit on control rows, exposes predict_proba

    def predict_cate(self, users_df: pd.DataFrame) -> pd.Series:
        """tau_hat(x) = P(Y=1|X, T=1) - P(Y=1|X, T=0), user_id-indexed."""
        features = build_feature_matrix(users_df)
        p1 = self.mu1.predict_proba(features)[:, 1]
        p0 = self.mu0.predict_proba(features)[:, 1]
        return pd.Series(
            p1 - p0, index=users_df["user_id"].values, name="predicted_uplift"
        )


def fit_t_learner(
    users_df: pd.DataFrame,
    treatment_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    random_state: int = 42,
    base_learner_factory: Callable[[int], object] | None = None,
) -> TLearner:
    """
    Fit mu1 on treated-only and mu0 on control-only rows.

    All three DataFrames MUST already be filtered to the TRAIN split
    (same contract as core/baseline.py: this function cannot know which
    split it was handed, so it does not filter for you).
    """
    factory = base_learner_factory or _default_base_learner_factory

    merged = (
        users_df.merge(treatment_df[["user_id", "treated"]], on="user_id")
        .merge(outcomes_df[["user_id", "converted"]], on="user_id")
    )
    if len(merged) != len(users_df):
        raise ValueError(
            "users/treatment/outcomes did not merge 1:1 -- check that all "
            "three DataFrames cover the same user set for this pilot"
        )

    arms = {}
    for arm_name, treated_flag in (("mu1", True), ("mu0", False)):
        arm = merged[merged["treated"] == treated_flag]
        if arm.empty:
            raise ValueError(f"no rows in the {'treated' if treated_flag else 'control'} arm")
        labels = arm["converted"].astype(int)
        if labels.nunique() < 2:
            raise ValueError(
                f"{arm_name} training data contains a single class "
                f"(all converted={labels.iloc[0]}); CATE estimation is "
                "meaningless on this arm -- increase pilot size or check the "
                "simulator config rather than silently degenerating"
            )
        model = factory(random_state)
        model.fit(build_feature_matrix(arm), labels.values)
        arms[arm_name] = model

    return TLearner(mu1=arms["mu1"], mu0=arms["mu0"])
