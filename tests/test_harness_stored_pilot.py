"""--pilot-id mode: validate STORED model runs against STORED ground truth."""

import pytest

from validation.harness import CLAIMS, MODELS, format_report, validate_stored_pilot


def test_validates_a_stored_pilot_end_to_end(engine, small_config_path):
    from core.pipeline import run_pipeline
    from db.seed import seed_demo_pilot

    pilot_id = seed_demo_pilot(small_config_path)
    run_pipeline(engine, pilot_id, split_seed=1)

    result = validate_stored_pilot(engine, pilot_id, n_bootstrap=20)
    assert result.n_test == 400  # the persisted test split, not the whole pilot
    assert set(MODELS) <= set(result.oracle.index)
    assert set(result.claims) == set(CLAIMS)
    assert "Validation report" in format_report([result], header="stored")


def test_stored_pilot_without_model_runs_is_an_error(engine, small_config_path):
    from db.seed import seed_demo_pilot

    pilot_id = seed_demo_pilot(small_config_path)
    with pytest.raises(ValueError, match="no model runs"):
        validate_stored_pilot(engine, pilot_id)
