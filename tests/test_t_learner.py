"""
Closed-form and behavioral tests for the T-learner.

The closed-form test is the spec's Phase 3 requirement: with known,
deterministic mu1/mu0, predict_cate must equal their difference EXACTLY --
no model fitting involved, so any deviation is a bug in the combination
logic itself, not learner noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.features import build_feature_matrix
from core.t_learner import TLearner, fit_t_learner


def _toy_users(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "user_id": [f"u{i}" for i in range(n)],
            "signup_date": pd.Timestamp("2023-06-01"),
            "historical_spend": rng.uniform(0, 100, n),
            "region": rng.choice(["north", "south", "east", "west"], n),
            "device_type": rng.choice(["mobile", "desktop", "tablet"], n),
        }
    )


class _ConstantProba:
    """Deterministic stand-in model: predict_proba returns a fixed p."""

    def __init__(self, p: float):
        self.p = p

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 1 - self.p), np.full(n, self.p)])


class _SpendThresholdProba:
    """p = high if historical_spend > threshold else low -- deterministic,
    feature-dependent, so the test exercises real per-row differencing."""

    def __init__(self, threshold: float, low: float, high: float):
        self.threshold, self.low, self.high = threshold, low, high

    def predict_proba(self, X):
        p = np.where(X["historical_spend"].values > self.threshold, self.high, self.low)
        return np.column_stack([1 - p, p])


def test_closed_form_constant_models():
    users = _toy_users()
    t = TLearner(mu1=_ConstantProba(0.30), mu0=_ConstantProba(0.10))
    cate = t.predict_cate(users)
    assert np.allclose(cate.values, 0.20)
    assert list(cate.index) == list(users["user_id"])


def test_closed_form_feature_dependent_models():
    users = _toy_users()
    mu1 = _SpendThresholdProba(threshold=50, low=0.10, high=0.40)
    mu0 = _SpendThresholdProba(threshold=50, low=0.10, high=0.10)
    cate = TLearner(mu1, mu0).predict_cate(users)

    features = build_feature_matrix(users)
    expected = np.where(features["historical_spend"].values > 50, 0.30, 0.0)
    assert np.allclose(cate.values, expected)


def test_fit_raises_on_single_class_arm():
    users = _toy_users(n=40)
    treatment = pd.DataFrame({"user_id": users["user_id"], "treated": [True, False] * 20})
    # every treated user converted -> mu1 arm is single-class
    outcomes = pd.DataFrame(
        {"user_id": users["user_id"], "converted": treatment["treated"].values}
    )
    with pytest.raises(ValueError, match="single class"):
        fit_t_learner(users, treatment, outcomes)


def test_fit_raises_on_mismatched_user_sets():
    users = _toy_users(n=10)
    treatment = pd.DataFrame({"user_id": users["user_id"][:5], "treated": [True] * 5})
    outcomes = pd.DataFrame({"user_id": users["user_id"], "converted": [False] * 10})
    with pytest.raises(ValueError, match="1:1"):
        fit_t_learner(users, treatment, outcomes)


def test_fitted_learner_recovers_directional_effect():
    """Behavioral (not closed-form): a real GBC fit on data with a genuine
    spend-dependent effect should predict higher uplift for high-spend users.
    Coarse assertion on group means only -- learner noise makes exact
    values untestable here; exactness lives in the closed-form tests."""
    rng = np.random.default_rng(7)
    n = 4000
    users = _toy_users(n=n, seed=7)
    treated = rng.random(n) < 0.5
    high_spend = users["historical_spend"].values > 50
    p = np.full(n, 0.10)
    p = p + np.where(treated & high_spend, 0.25, 0.0)
    converted = rng.random(n) < p

    treatment = pd.DataFrame({"user_id": users["user_id"], "treated": treated})
    outcomes = pd.DataFrame({"user_id": users["user_id"], "converted": converted})

    t = fit_t_learner(users, treatment, outcomes)
    cate = t.predict_cate(users)
    # Ordering only, not magnitude: the T-learner recovers the DIRECTION of
    # heterogeneity (high-spend users have higher uplift) but its absolute
    # level is biased -- mu1 and mu0 carry independent, non-cancelling errors
    # (spec 7.2). Pinning a magnitude here would test learner calibration, not
    # the T-learner logic; exact magnitude lives in the closed-form tests.
    assert cate.values[high_spend].mean() > cate.values[~high_spend].mean()
