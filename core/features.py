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

# Fixed, known vocabularies (matching data_sim/pilot_simulator.py's
# _REGIONS/_DEVICES) rather than deriving categories from whatever happens
# to be present in a given DataFrame slice. This matters: if train and
# test were one-hot encoded independently via a naive pd.get_dummies call,
# a category missing by chance from one slice (more likely in a small
# test split) would produce a DIFFERENT set of columns between train and
# test -- silently misaligning features at predict() time, or crashing.
_KNOWN_REGIONS = ["north", "south", "east", "west"]
_KNOWN_DEVICES = ["mobile", "desktop", "tablet"]

_REFERENCE_DATE = pd.Timestamp("2023-01-01")


def build_feature_matrix(users_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a numeric feature matrix from raw user covariates.

    Columns: historical_spend, signup_days_since_start, region_<r> (one per
    known region), device_<d> (one per known device). Index matches
    users_df's row order (NOT necessarily user_id -- caller should track
    user_id separately if needed for later joins).
    """
    signup_days = (pd.to_datetime(users_df["signup_date"]) - _REFERENCE_DATE).dt.days

    region_dummies = pd.get_dummies(
        pd.Categorical(users_df["region"], categories=_KNOWN_REGIONS),
        prefix="region",
    )
    device_dummies = pd.get_dummies(
        pd.Categorical(users_df["device_type"], categories=_KNOWN_DEVICES),
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
