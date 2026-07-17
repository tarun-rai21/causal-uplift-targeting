import pandas as pd
import pytest
from pydantic import ValidationError

from config.schema import PilotConfig
from data_sim.pilot_simulator import PilotSimulator

BASE_KWARGS = dict(
    n_users=5000,
    baseline_rate=0.10,
    propensity=0.5,
    seed=42,
    segment_effects={"persuadable": 0.15, "sure_thing": 0.01, "lost_cause": 0.0},
    segment_mix={"persuadable": 0.34, "sure_thing": 0.33, "lost_cause": 0.33},
    segment_baseline_offsets={
        "persuadable": -0.02,
        "sure_thing": 0.30,
        "lost_cause": -0.08,
    },
)


def _make_simulator(**overrides) -> PilotSimulator:
    kwargs = dict(BASE_KWARGS)
    kwargs.update(overrides)
    return PilotSimulator(PilotConfig(**kwargs))


def test_generate_returns_expected_shapes_and_matching_user_ids():
    sim = _make_simulator()
    users_df, treatment_df, outcomes_df, _ = sim.generate()
    assert len(users_df) == len(treatment_df) == len(outcomes_df) == 5000
    assert set(users_df["user_id"]) == set(treatment_df["user_id"]) == set(outcomes_df["user_id"])


def test_fixed_seed_is_reproducible():
    u1, t1, o1, _ = _make_simulator(seed=7).generate()
    u2, t2, o2, _ = _make_simulator(seed=7).generate()
    pd.testing.assert_frame_equal(u1, u2)
    pd.testing.assert_frame_equal(t1, t2)
    pd.testing.assert_frame_equal(o1, o2)


def test_different_seeds_produce_different_data():
    u1, _, _, _ = _make_simulator(seed=1).generate()
    u2, _, _, _ = _make_simulator(seed=2).generate()
    assert not u1["historical_spend"].equals(u2["historical_spend"])


def test_treatment_propensity_matches_config_within_sampling_noise():
    sim = _make_simulator(n_users=20_000, propensity=0.3, seed=3)
    _, treatment_df, _, _ = sim.generate()
    observed_rate = treatment_df["treated"].mean()
    assert abs(observed_rate - 0.3) < 0.02


def test_outcomes_are_well_formed():
    sim = _make_simulator(n_users=20_000, seed=4)
    _, _, outcomes_df, _ = sim.generate()
    assert outcomes_df["converted"].dtype == bool
    assert (outcomes_df["revenue"] >= 0).all()
    assert (outcomes_df.loc[~outcomes_df["converted"], "revenue"] == 0).all()


def test_segment_names_a_do_not_leak_a_deterministic_covariate():
    """
    Segments must overlap on observable covariates, not be perfectly
    separable -- otherwise CATE estimation in later phases would be
    trivial and not representative of real difficulty (spec Section 7.1).
    """
    sim = _make_simulator(n_users=20_000, seed=6)
    users_df, _, _, _ = sim.generate()
    persuadable_spend = users_df.loc[users_df["segment"] == "persuadable", "historical_spend"]
    sure_thing_spend = users_df.loc[users_df["segment"] == "sure_thing", "historical_spend"]
    # means should differ (segment carries signal)...
    assert sure_thing_spend.mean() > persuadable_spend.mean()
    # ...but distributions must overlap substantially (not perfectly separable)
    overlap = (persuadable_spend.max() > sure_thing_spend.min())
    assert overlap


def test_observed_uplift_direction_matches_ground_truth():
    """
    Directional smoke test ahead of the formal Phase 6 validation harness:
    sure_thing should show a near-zero observed treated-vs-control
    conversion gap despite a high baseline rate; persuadable should show
    a real, large gap. This confirms baseline_rate, segment_baseline_
    offsets, and segment_effects are wired together correctly.
    """
    sim = _make_simulator(n_users=100_000, seed=5)
    users_df, treatment_df, outcomes_df, _ = sim.generate()
    merged = users_df.merge(treatment_df, on="user_id").merge(outcomes_df, on="user_id")

    def observed_uplift(segment: str) -> float:
        seg_df = merged[merged["segment"] == segment]
        treated_rate = seg_df.loc[seg_df["treated"], "converted"].mean()
        control_rate = seg_df.loc[~seg_df["treated"], "converted"].mean()
        return treated_rate - control_rate

    assert observed_uplift("persuadable") > 0.10   # true effect 0.15
    assert abs(observed_uplift("sure_thing")) < 0.03  # true effect 0.01


def test_invalid_config_is_rejected_before_a_simulator_can_be_built():
    """
    Regression guard: PilotConfig's own validation must catch an invalid
    probability combination before a PilotSimulator is ever constructed --
    not surface three calls deep inside generate() as a numpy error.
    """
    bad_kwargs = dict(BASE_KWARGS)
    bad_kwargs["segment_baseline_offsets"] = {
        "persuadable": 0.0, "sure_thing": 0.95, "lost_cause": 0.0,
    }
    with pytest.raises(ValidationError, match=r"out of \[0, 1\] bounds"):
        PilotConfig(**bad_kwargs)
