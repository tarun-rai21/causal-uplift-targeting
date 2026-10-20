import pandas as pd
import pytest

from config.schema import PilotConfig
from core.data_split import (
    compute_data_split,
    load_split_user_ids,
    persist_split,
    subset_to_split,
)
from data_sim.pilot_simulator import PilotSimulator
from tests.conftest import insert_pilot

BASE_KWARGS = dict(
    n_users=10_000,
    baseline_rate=0.10,
    propensity=0.5,
    seed=42,
    segment_effects={"persuadable": 0.15, "sure_thing": 0.01, "lost_cause": 0.0},
    segment_mix={"persuadable": 0.34, "sure_thing": 0.33, "lost_cause": 0.33},
    segment_baseline_offsets={"persuadable": -0.02, "sure_thing": 0.30, "lost_cause": -0.08},
)


def _generate_pilot(**overrides):
    kwargs = dict(BASE_KWARGS)
    kwargs.update(overrides)
    sim = PilotSimulator(PilotConfig(**kwargs))
    return sim.generate()


def test_split_covers_every_user_exactly_once():
    users_df, treatment_df, outcomes_df, _ = _generate_pilot()
    split_df = compute_data_split(
        treatment_df, outcomes_df, pilot_id="p1", test_size=0.2, split_seed=1
    )
    assert set(split_df["user_id"]) == set(users_df["user_id"])
    assert split_df["user_id"].is_unique
    assert set(split_df["split"]) == {"train", "test"}


def test_split_proportions_match_test_size():
    _, treatment_df, outcomes_df, _ = _generate_pilot()
    split_df = compute_data_split(
        treatment_df, outcomes_df, pilot_id="p1", test_size=0.2, split_seed=1
    )
    test_frac = (split_df["split"] == "test").mean()
    assert abs(test_frac - 0.2) < 0.01


def test_split_is_reproducible_given_same_seed():
    _, treatment_df, outcomes_df, _ = _generate_pilot()
    s1 = compute_data_split(treatment_df, outcomes_df, "p1", 0.2, split_seed=99)
    s2 = compute_data_split(treatment_df, outcomes_df, "p1", 0.2, split_seed=99)
    pd.testing.assert_frame_equal(
        s1.sort_values("user_id").reset_index(drop=True),
        s2.sort_values("user_id").reset_index(drop=True),
    )


def test_stratification_preserves_treatment_ratio_in_both_splits():
    """
    The whole point of stratifying instead of plain-random splitting:
    both train and test must closely match the overall treated ratio.
    """
    _, treatment_df, outcomes_df, _ = _generate_pilot(n_users=10_000, propensity=0.3, seed=11)
    split_df = compute_data_split(treatment_df, outcomes_df, "p1", test_size=0.2, split_seed=1)
    merged = split_df.merge(treatment_df, on="user_id")

    overall_rate = merged["treated"].mean()
    train_rate = merged.loc[merged["split"] == "train", "treated"].mean()
    test_rate = merged.loc[merged["split"] == "test", "treated"].mean()

    assert abs(train_rate - overall_rate) < 0.02
    assert abs(test_rate - overall_rate) < 0.02


def test_stratification_preserves_conversion_ratio_in_both_splits():
    _, treatment_df, outcomes_df, _ = _generate_pilot(n_users=10_000, seed=12)
    split_df = compute_data_split(treatment_df, outcomes_df, "p1", test_size=0.2, split_seed=1)
    merged = split_df.merge(outcomes_df, on="user_id")

    overall_rate = merged["converted"].mean()
    train_rate = merged.loc[merged["split"] == "train", "converted"].mean()
    test_rate = merged.loc[merged["split"] == "test", "converted"].mean()

    assert abs(train_rate - overall_rate) < 0.02
    assert abs(test_rate - overall_rate) < 0.02


def test_invalid_test_size_is_rejected():
    _, treatment_df, outcomes_df, _ = _generate_pilot()
    with pytest.raises(ValueError, match="test_size must be in"):
        compute_data_split(treatment_df, outcomes_df, "p1", test_size=1.5, split_seed=1)


def test_persist_and_load_round_trip(engine):
    _, treatment_df, outcomes_df, _ = _generate_pilot(n_users=500, seed=2)
    insert_pilot(engine, "p_test", treatment_df["user_id"])
    split_df = compute_data_split(treatment_df, outcomes_df, "p_test", test_size=0.25, split_seed=1)
    persist_split(engine, split_df)

    train_ids = load_split_user_ids(engine, "p_test", "train")
    test_ids = load_split_user_ids(engine, "p_test", "test")

    assert len(train_ids) + len(test_ids) == 500
    assert set(train_ids).isdisjoint(set(test_ids))


def test_load_split_rejects_invalid_split_name(engine):
    with pytest.raises(ValueError, match="split must be"):
        load_split_user_ids(engine, "any_pilot", "validation")


def test_subset_to_split_partitions_every_frame_consistently():
    users_df, treatment_df, outcomes_df, _ = _generate_pilot()
    split_df = compute_data_split(treatment_df, outcomes_df, "p1", test_size=0.2, split_seed=1)
    train_u, train_o = subset_to_split(split_df, "train", users_df, outcomes_df)
    (test_u,) = subset_to_split(split_df, "test", users_df)

    assert list(train_u["user_id"]) == list(train_o["user_id"])  # row-aligned
    assert set(train_u["user_id"]).isdisjoint(test_u["user_id"])
    assert len(train_u) + len(test_u) == len(users_df)
    assert list(test_u.index) == list(range(len(test_u)))
    with pytest.raises(ValueError, match="split must be"):
        subset_to_split(split_df, "validation", users_df)
