"""
Train/test split assignment (Phase 1.5 -- not in the original spec, added
because every downstream phase (baseline, T-learner, X-learner, evaluation,
validation harness) must evaluate against the SAME held-out split, or
Qini/CI numbers become incomparable across models and evaluating on
training data risks an overfit model showing spurious Qini separation.

Design: stratify on the combination of (treated, converted), not a plain
random split. A pilot is a randomized experiment (treatment assigned
independently of segment), so a plain random split preserves the treatment
ratio only in expectation -- any single actual split can, by chance,
under-represent treated (or converted) users in the test set, especially
with an imbalanced propensity or a smaller pilot. Since Phase 5's Qini
evaluation directly compares treated-vs-control conversion rates within
the test set, an unlucky split would inject noise into that comparison
that has nothing to do with model quality. Stratifying on the joint
(treated, converted) label guarantees both splits keep a representative
mix of all four treated/converted combinations, which T-learner and
X-learner also need (they fit separate sub-models per treatment arm).
"""

from __future__ import annotations

import warnings

import pandas as pd
from sklearn.model_selection import train_test_split


def compute_data_split(
    treatment_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    pilot_id: str,
    test_size: float = 0.2,
    split_seed: int = 42,
) -> pd.DataFrame:
    """
    Assign each user in a pilot to 'train' or 'test', stratified on the
    joint (treated, converted) label.

    Returns a DataFrame matching the data_splits table schema:
    user_id, pilot_id, split, split_seed.
    """
    if not (0.0 < test_size < 1.0):
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")

    merged = treatment_df[["user_id", "treated"]].merge(
        outcomes_df[["user_id", "converted"]], on="user_id"
    )
    strat_key = merged["treated"].astype(str) + "_" + merged["converted"].astype(str)

    try:
        train_ids, test_ids = train_test_split(
            merged["user_id"],
            test_size=test_size,
            random_state=split_seed,
            stratify=strat_key,
        )
    except ValueError as e:
        # train_test_split raises if any stratify class has fewer members
        # than needed for the requested test_size (e.g. a tiny pilot with
        # a rare treated+converted combination). Fall back to an
        # unstratified split rather than hard-failing -- a warning here is
        # the right severity since a small deliberate test pilot is a
        # legitimate use case (same philosophy as PilotConfig's thin-
        # segment warning in Phase 0).
        warnings.warn(
            f"Stratified split failed ({e}); falling back to unstratified "
            "random split. This is expected for small pilots where some "
            "(treated, converted) combination is too rare to stratify on.",
            stacklevel=2,
        )
        train_ids, test_ids = train_test_split(
            merged["user_id"], test_size=test_size, random_state=split_seed
        )

    split_df = pd.DataFrame({
        "user_id": pd.concat([train_ids, test_ids], ignore_index=True),
        "pilot_id": pilot_id,
        "split": ["train"] * len(train_ids) + ["test"] * len(test_ids),
        "split_seed": split_seed,
    })
    return split_df


def persist_split(engine, split_df: pd.DataFrame) -> None:
    """Write a computed split to the data_splits table."""
    with engine.begin() as conn:
        split_df.to_sql("data_splits", conn, if_exists="append", index=False)


def load_split_user_ids(engine, pilot_id: str, split: str) -> list[str]:
    """
    Fetch the user_ids assigned to a given split ('train' or 'test') for a
    pilot. This is the read path every later phase (baseline training,
    T-learner training, evaluation) uses to filter down to the correct
    subset -- centralizing it here means there's exactly one place that
    can get the train/test boundary wrong, not one per consuming phase.
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    from sqlalchemy import text

    with engine.connect() as conn:
        result = conn.execute(
            text(
                "SELECT user_id FROM data_splits "
                "WHERE pilot_id = :pilot_id AND split = :split"
            ),
            {"pilot_id": pilot_id, "split": split},
        )
        return [row[0] for row in result]
