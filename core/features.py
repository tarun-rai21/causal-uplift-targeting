"""
Shared feature-matrix construction -- used identically by baseline,
T-learner, and X-learner (Phases 2-4), so every model sees the SAME
numeric representation of user covariates. If each model engineered its
own features, a difference in Qini/AUUC between models could be a
feature-engineering artifact rather than a real difference in causal-
estimation method, which would undermine the whole point of comparing
them.

Deliberately does NOT include `segment` -- that column is retained only
for ground-truth validation/diagnostics (Phase 1, Phase 6), never a real-
world observable feature a deployed model would have access to.
"""

from __future__ import annotations

import pandas as pd

# Fixed, known vocabularies rather than deriving categories from whatever
# happens to be present in a given DataFrame slice. This matters: if train
# and test were one-hot encoded independently via a naive pd.get_dummies
# call, a category missing by chance from one slice (more likely in a small
# test split) would produce a DIFFERENT set of columns between train and
# test -- silently misaligning features at predict() time, or crashing.
# Defined here (the feature contract) and imported by data_sim, so the
# simulator cannot drift out of sync with what the models can encode.
KNOWN_REGIONS = ["north", "south", "east", "west"]
KNOWN_DEVICES = ["mobile", "desktop", "tablet"]

_REFERENCE_DATE = pd.Timestamp("2023-01-01")


def build_feature_matrix(users_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a numeric feature matrix from raw user covariates.

    Columns: historical_spend, signup_days_since_start, region_<r> (one per
    known region), device_<d> (one per known device). Index matches
    users_df's row order (NOT necessarily user_id -- caller should track
    user_id separately if needed for later joins).

    Raises ValueError on missing values or out-of-vocabulary categories.
    pd.Categorical maps both to NaN, which get_dummies would encode as an
    all-zero row -- indistinguishable from a legitimate reference level, so
    a "NORTH" or a null region would silently become a different user.
    """
    _validate_inputs(users_df)
    signup_days = (pd.to_datetime(users_df["signup_date"]) - _REFERENCE_DATE).dt.days

    region_dummies = pd.get_dummies(
        pd.Categorical(users_df["region"], categories=KNOWN_REGIONS),
        prefix="region",
    )
    device_dummies = pd.get_dummies(
        pd.Categorical(users_df["device_type"], categories=KNOWN_DEVICES),
        prefix="device",
    )

    features = pd.DataFrame(
        {
            "historical_spend": users_df["historical_spend"].astype(float).values,
            "signup_days_since_start": signup_days.values,
        },
        index=users_df.index,
    )
    region_dummies = region_dummies.set_axis(users_df.index)
    device_dummies = device_dummies.set_axis(users_df.index)
    features = pd.concat([features, region_dummies, device_dummies], axis=1)
    return features


def _validate_inputs(users_df: pd.DataFrame) -> None:
    for col in ("historical_spend", "signup_date", "region", "device_type"):
        n_missing = int(users_df[col].isna().sum())
        if n_missing:
            raise ValueError(f"{col!r} has {n_missing} missing value(s)")
    for col, vocab in (("region", KNOWN_REGIONS), ("device_type", KNOWN_DEVICES)):
        unknown = set(users_df[col].unique()) - set(vocab)
        if unknown:
            raise ValueError(f"{col!r} has value(s) outside {vocab}: {sorted(unknown)}")
