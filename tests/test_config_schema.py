# tests/test_config_schema.py

import pytest
from pydantic import ValidationError

from config.schema import PilotConfig, load_pilot_config

VALID_KWARGS = dict(
    n_users=50_000,
    baseline_rate=0.10,
    propensity=0.5,
    seed=42,
    segment_effects={
        "persuadable": 0.15,
        "sure_thing": 0.01,
        "lost_cause": 0.0,
    },
    segment_mix={
        "persuadable": 0.34,
        "sure_thing": 0.33,
        "lost_cause": 0.33,
    },
)


def test_valid_config_passes():
    cfg = PilotConfig(**VALID_KWARGS)
    assert cfg.n_users == 50_000


def test_segment_mix_must_sum_to_one():
    bad = dict(VALID_KWARGS)
    bad["segment_mix"] = {"persuadable": 0.5, "sure_thing": 0.3, "lost_cause": 0.3}  # sums to 1.1
    with pytest.raises(ValidationError, match="must sum to 1.0"):
        PilotConfig(**bad)


def test_segment_keys_must_match_between_mix_and_effects():
    bad = dict(VALID_KWARGS)
    bad["segment_effects"] = {
        "persuadable": 0.15,
        "sure_thing": 0.01,
        "sleeping_dog": -0.05,  # not in segment_mix
    }
    with pytest.raises(ValidationError, match="identical segment keys"):
        PilotConfig(**bad)


def test_required_segments_must_be_present():
    bad = dict(VALID_KWARGS)
    bad["segment_mix"] = {"persuadable": 0.5, "sure_thing": 0.5}  # missing lost_cause
    bad["segment_effects"] = {"persuadable": 0.15, "sure_thing": 0.01}
    with pytest.raises(ValidationError, match="Missing required segment"):
        PilotConfig(**bad)


@pytest.mark.parametrize("bad_propensity", [0.0, 1.0, -0.1, 1.5])
def test_propensity_must_be_strictly_between_zero_and_one(bad_propensity):
    bad = dict(VALID_KWARGS)
    bad["propensity"] = bad_propensity
    with pytest.raises(ValidationError):
        PilotConfig(**bad)


@pytest.mark.parametrize("bad_rate", [0.0, 1.0, -0.05, 1.2])
def test_baseline_rate_must_be_strictly_between_zero_and_one(bad_rate):
    bad = dict(VALID_KWARGS)
    bad["baseline_rate"] = bad_rate
    with pytest.raises(ValidationError):
        PilotConfig(**bad)


def test_n_users_must_be_positive():
    bad = dict(VALID_KWARGS)
    bad["n_users"] = 0
    with pytest.raises(ValidationError):
        PilotConfig(**bad)


def test_thin_segment_warns_but_does_not_raise():
    thin = dict(VALID_KWARGS)
    thin["n_users"] = 100  # -> ~33 users per segment, below the 200 heuristic floor
    with pytest.warns(UserWarning, match="fewer than"):
        cfg = PilotConfig(**thin)
    assert cfg.n_users == 100


def test_load_pilot_config_from_default_yaml():
    cfg = load_pilot_config("config/default_config.yaml")
    assert cfg.n_users > 0
    assert set(cfg.segment_mix.keys()) == set(cfg.segment_effects.keys())
