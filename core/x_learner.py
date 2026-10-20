"""
X-learner (spec Section 7.3; Kunzel, Sekhon, Bickel & Yu, PNAS 2019).

Stage 1 -- per-arm outcome models, identical to the T-learner:
    mu1(x) = P(Y=1 | X, T=1),   mu0(x) = P(Y=1 | X, T=0)

Stage 2 -- impute each user's individual effect using the OTHER arm's model,
then regress those imputed effects on X:
    treated i:  D1_i = Y_i - mu0(X_i)   -> tau1(x) fit on treated rows
    control i:  D0_i = mu1(X_i) - Y_i   -> tau0(x) fit on control rows

Stage 3 -- combine, weighting by the propensity e:
    tau_hat(x) = e * tau0(x) + (1 - e) * tau1(x)

Why this targets the T-learner's failure mode (core/t_learner.py): under an
imbalanced pilot (say e = 0.2) mu1 is fit on few rows and is noisy, while
mu0 is fit on the large control arm and is reliable. Look at what each
effect model is built from:
    tau1: treated rows' REAL outcomes minus mu0 (reliable)  -> trustworthy
    tau0: mu1 (noisy) minus control rows' real outcomes     -> inherits mu1's noise
The weights e * tau0 + (1 - e) * tau1 put 1 - e = 0.8 on tau1 and only
e = 0.2 on tau0, so the estimate leans on the effect model whose imputed
targets came from the well-estimated arm. The T-learner, by contrast,
subtracts the noisy mu1 directly at full weight. At e = 0.5 the two are
weighted equally and the X-learner behaves much like the T-learner.

e is the KNOWN randomization probability (treatment_assignment.propensity_e),
not an estimate: in a randomized pilot it is a design constant, and
estimating it would only add noise. Hence config rule 3 (0 < propensity < 1):
at e = 0 or 1 one arm is empty and the combination degenerates.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.features import build_feature_matrix
from core.learners import (
    BaseLearnerFactory,
    BaseRegressorFactory,
    default_base_regressor,
)
from core.t_learner import fit_t_learner, join_training_frame, split_arms


@dataclass
class XLearner:
    """
    The two fitted effect models plus the propensity they are combined with.
    Construct via fit_x_learner(); like TLearner, a dumb container so tests
    can build it from deterministic stand-in models.
    """

    tau1: object  # regressor fit on treated rows' imputed effects, exposes predict
    tau0: object  # regressor fit on control rows' imputed effects, exposes predict
    propensity: float

    def predict_cate(self, users_df: pd.DataFrame) -> pd.Series:
        """tau_hat(x) = e * tau0(x) + (1 - e) * tau1(x), user_id-indexed."""
        features = build_feature_matrix(users_df)
        e = self.propensity
        cate = e * self.tau0.predict(features) + (1 - e) * self.tau1.predict(features)
        return pd.Series(cate, index=users_df["user_id"].values, name="predicted_uplift")


def fit_x_learner(
    users_df: pd.DataFrame,
    treatment_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    random_state: int = 42,
    base_learner_factory: BaseLearnerFactory | None = None,
    effect_learner_factory: BaseRegressorFactory | None = None,
) -> XLearner:
    """
    Fit all three stages. Same TRAIN-split-only contract as fit_t_learner.

    treatment_df must carry propensity_e (as persisted in
    treatment_assignment) with a single value across users: the
    combination weight is the pilot's design propensity, and a pilot with
    varying e(x) would need a per-user weight this implementation does not
    support -- better to refuse than to silently average it away.
    """
    propensity = _design_propensity(treatment_df)
    effect_factory = effect_learner_factory or default_base_regressor

    stage1 = fit_t_learner(
        users_df, treatment_df, outcomes_df,
        random_state=random_state, base_learner_factory=base_learner_factory,
    )
    other_arm = {"mu1": stage1.mu0, "mu0": stage1.mu1}

    effect_models = {}
    for arm_name, arm in split_arms(join_training_frame(users_df, treatment_df, outcomes_df)):
        features = build_feature_matrix(arm)
        y = arm["converted"].astype(float).to_numpy()
        counterfactual = other_arm[arm_name].predict_proba(features)[:, 1]
        imputed = y - counterfactual if arm_name == "mu1" else counterfactual - y
        model = effect_factory(random_state)
        model.fit(features, imputed)
        effect_models[arm_name] = model

    return XLearner(tau1=effect_models["mu1"], tau0=effect_models["mu0"], propensity=propensity)


def _design_propensity(treatment_df: pd.DataFrame) -> float:
    if "propensity_e" not in treatment_df.columns:
        raise ValueError(
            "treatment_df has no propensity_e column; the X-learner needs the "
            "pilot's known randomization probability"
        )
    values = treatment_df["propensity_e"].unique()
    if len(values) != 1:
        raise ValueError(
            f"propensity_e must be constant across the pilot, got {sorted(values)[:5]}"
        )
    e = float(values[0])
    if not 0.0 < e < 1.0:
        raise ValueError(f"propensity_e must be in (0, 1), got {e}")
    return e
