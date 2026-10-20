"""
Closed-form and behavioral tests for the X-learner. As with the T-learner,
exactness lives in tests with deterministic stand-in models; fitted-model
tests only check direction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.features import build_feature_matrix
from core.x_learner import XLearner, fit_x_learner
from tests.test_t_learner import _ConstantProba, _toy_users


class _ConstantPredict:
    def __init__(self, value: float):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value)


class _RecordingRegressor:
    """Captures the imputed targets stage 2 fits on, then predicts their mean."""

    fitted: list = []

    def fit(self, X, y):
        _RecordingRegressor.fitted.append(np.asarray(y).copy())
        self.mean = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean)


def _pilot(n=40, propensity=0.5):
    users = _toy_users(n=n)
    treated = np.array([True, False] * (n // 2))
    converted = np.array([True, True, False, False] * (n // 4))
    treatment = pd.DataFrame(
        {"user_id": users["user_id"], "treated": treated, "propensity_e": propensity}
    )
    outcomes = pd.DataFrame({"user_id": users["user_id"], "converted": converted})
    return users, treatment, outcomes


def test_closed_form_combination_weights_tau0_by_e():
    users = _toy_users()
    x = XLearner(tau1=_ConstantPredict(0.30), tau0=_ConstantPredict(0.10), propensity=0.2)
    cate = x.predict_cate(users)
    assert np.allclose(cate.values, 0.2 * 0.10 + 0.8 * 0.30)
    assert list(cate.index) == list(users["user_id"])


def test_stage2_imputes_effects_from_the_other_arm():
    """With constant mu1 = 0.3 and mu0 = 0.1: treated targets must be
    Y - 0.1 and control targets 0.3 - Y, exactly."""
    users, treatment, outcomes = _pilot(propensity=0.25)
    stand_ins = iter([_ConstantProba(0.30), _ConstantProba(0.10)])  # mu1, then mu0
    _RecordingRegressor.fitted = []

    model = fit_x_learner(
        users, treatment, outcomes,
        base_learner_factory=lambda _rs: _FitNoop(next(stand_ins)),
        effect_learner_factory=lambda _rs: _RecordingRegressor(),
    )

    y_treated = outcomes["converted"][treatment["treated"]].astype(float).to_numpy()
    y_control = outcomes["converted"][~treatment["treated"]].astype(float).to_numpy()
    d1, d0 = _RecordingRegressor.fitted
    assert np.allclose(d1, y_treated - 0.10)
    assert np.allclose(d0, 0.30 - y_control)
    assert model.propensity == 0.25
    expected = 0.25 * d0.mean() + 0.75 * d1.mean()
    assert np.allclose(model.predict_cate(users).values, expected)


class _FitNoop:
    """Wraps a stand-in so fit_t_learner can 'fit' it without changing it."""

    def __init__(self, inner):
        self.inner = inner

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return self.inner.predict_proba(X)


@pytest.mark.parametrize(
    "treatment_update, match",
    [
        ({"propensity_e": None}, "no propensity_e"),
        ({"propensity_e": "vary"}, "constant"),
        ({"propensity_e": 1.0}, r"\(0, 1\)"),
    ],
)
def test_rejects_unusable_propensity(treatment_update, match):
    users, treatment, outcomes = _pilot()
    value = treatment_update["propensity_e"]
    if value is None:
        treatment = treatment.drop(columns="propensity_e")
    elif value == "vary":
        treatment["propensity_e"] = np.linspace(0.2, 0.8, len(treatment))
    else:
        treatment["propensity_e"] = value
    with pytest.raises(ValueError, match=match):
        fit_x_learner(users, treatment, outcomes)


def test_fitted_learner_recovers_directional_effect():
    n = 4000
    users = _toy_users(n=n, seed=7)
    rng = np.random.default_rng(1007)  # independent of spend; see the T-learner test
    treated = rng.random(n) < 0.3
    high_spend = build_feature_matrix(users)["historical_spend"].values > 50
    assert (treated & high_spend).sum() > 300  # the effect cell is populated
    converted = rng.random(n) < 0.10 + np.where(treated & high_spend, 0.25, 0.0)
    treatment = pd.DataFrame(
        {"user_id": users["user_id"], "treated": treated, "propensity_e": 0.3}
    )
    outcomes = pd.DataFrame({"user_id": users["user_id"], "converted": converted})

    cate = fit_x_learner(users, treatment, outcomes).predict_cate(users)
    assert cate.values[high_spend].mean() > cate.values[~high_spend].mean()
