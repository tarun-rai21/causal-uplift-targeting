"""
Single definition of the default base learner shared by every model
(baseline, T-learner, and later X-learner).

Any Qini/AUUC difference between models must come from the estimand, not
from model capacity. That guarantee previously lived in a comment that
asked two modules to keep identical constructor calls in sync by hand;
defining the learner once here makes it structural instead.
"""

from __future__ import annotations

from typing import Callable

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

# Maps random_state -> an unfitted classifier exposing fit/predict_proba.
BaseLearnerFactory = Callable[[int], object]
# Maps random_state -> an unfitted regressor exposing fit/predict.
BaseRegressorFactory = Callable[[int], object]


def default_base_learner(random_state: int) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(random_state=random_state)


def default_base_regressor(random_state: int) -> GradientBoostingRegressor:
    """The X-learner's second-stage effect models regress continuous imputed
    effects, so they need a regressor -- the same gradient-boosting family
    with the same default capacity (100 depth-3 trees, learning rate 0.1) as
    the classifier above, keeping the capacity-parity argument intact."""
    return GradientBoostingRegressor(random_state=random_state)
