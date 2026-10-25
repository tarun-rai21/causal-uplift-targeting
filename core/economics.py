"""
Per-user economics estimates the policy layer prices targeting with
(Phase 7). Kept separate from the uplift models on purpose: every policy --
naive, T-learner, X-learner -- is costed with the SAME estimates, so policy
comparisons differ only in who each model ranks first, never in how a
given user's cost was estimated.

Under a cashback-on-conversion incentive at rate r, treating user i:
    costs                 r * p1(x_i) * m(x_i)     cashback on every treated conversion
    earns (incrementally) tau(x_i) * m(x_i)        uplift-model territory, not here
where
    p1(x) = P(convert | treated, x)   -- fit on treated TRAIN rows only
    m(x)  = E[revenue | converted, x] -- fit on converted TRAIN rows, both arms

m pools both arms on the assumption that the incentive changes WHETHER a
user converts, not their basket size. That holds in the simulator by
construction; on real data it is checkable (compare treated vs. control
revenue among converters) and, if it fails, m should be fit on treated
converters only.

Note p1 is exactly the T-learner's mu1 when fit with the same data and
seed; it is refit here so the economics do not depend on which uplift
model happens to be in use.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.features import build_feature_matrix
from core.learners import (
    BaseLearnerFactory,
    BaseRegressorFactory,
    default_base_learner,
    default_base_regressor,
)
from core.t_learner import join_training_frame


@dataclass
class EconomicsModel:
    p_treated: object  # classifier: P(convert | treated, x)
    revenue: object  # regressor: E[revenue | converted, x]

    def predict(self, users_df: pd.DataFrame) -> pd.DataFrame:
        """user_id, p_convert_if_treated, revenue_per_conversion."""
        features = build_feature_matrix(users_df)
        return pd.DataFrame({
            "user_id": users_df["user_id"].to_numpy(),
            "p_convert_if_treated": self.p_treated.predict_proba(features)[:, 1],
            # A regressor can extrapolate below 0; a basket cannot.
            "revenue_per_conversion": self.revenue.predict(features).clip(min=0.0),
        })


def fit_economics(
    users_df: pd.DataFrame,
    treatment_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    random_state: int = 42,
    base_learner_factory: BaseLearnerFactory | None = None,
    revenue_learner_factory: BaseRegressorFactory | None = None,
) -> EconomicsModel:
    """Fit p1 and m. Same TRAIN-split-only contract as the uplift models."""
    merged = join_training_frame(users_df, treatment_df, outcomes_df).merge(
        outcomes_df[["user_id", "revenue"]], on="user_id"
    )
    treated = merged[merged["treated"]]
    converted = merged[merged["converted"]]
    if treated["converted"].nunique() < 2:
        raise ValueError("treated arm has a single outcome class; cannot fit p1")
    if converted.empty:
        raise ValueError("no converted users; cannot fit revenue per conversion")

    p_treated = (base_learner_factory or default_base_learner)(random_state)
    p_treated.fit(build_feature_matrix(treated), treated["converted"].astype(int).to_numpy())
    revenue = (revenue_learner_factory or default_base_regressor)(random_state)
    revenue.fit(build_feature_matrix(converted), converted["revenue"].to_numpy())
    return EconomicsModel(p_treated=p_treated, revenue=revenue)
