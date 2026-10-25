import json

import numpy as np
import pandas as pd
import pytest

from config.schema import load_pilot_config
from data_sim.ground_truth import GroundTruth
from data_sim.pilot_simulator import PilotSimulator
from validation.oracle_evaluation import covariate_ceiling_target_ids, true_incremental_outcome


@pytest.fixture(scope="module")
def big_pilot():
    cfg = load_pilot_config("config/default_config.yaml").model_copy(
        update={"n_users": 200_000, "seed": 3}
    )
    users_df, _, _, gt = PilotSimulator(cfg).generate()
    return users_df, gt


def test_posterior_rows_are_valid_distributions_including_spend_zero(big_pilot):
    _, gt = big_pilot
    post = gt.segment_posterior(np.array([0.0, 50.0, 150.0, 1_000.0]))
    assert list(post.columns) == list(gt.segment_mix)
    assert not post.isna().any().any()
    assert np.allclose(post.sum(axis=1), 1.0)
    assert post.iloc[-1].idxmax() == "sure_thing"  # highest spend mean
    assert post.iloc[0].idxmax() == "lost_cause"  # lowest mean, largest mass at 0


def test_cate_given_x_is_calibrated_against_realized_segment_uplift(big_pilot):
    """The real correctness check for the Bayes formula (clip point mass
    included): within bins of predicted E[tau|X], the average TRUE segment
    uplift of the users in that bin must match the predicted average."""
    users_df, gt = big_pilot
    predicted = gt.true_cate_given_covariates(users_df).to_numpy()
    realized = users_df["segment"].map(gt.segment_effects).to_numpy()
    bins = pd.qcut(predicted, 10, labels=False, duplicates="drop")
    by_bin = pd.DataFrame({"bin": bins, "pred": predicted, "real": realized}).groupby("bin")
    gap = (by_bin["pred"].mean() - by_bin["real"].mean()).abs()
    assert gap.max() < 0.005, gap


def test_covariate_ceiling_beats_random_but_not_segment_oracle(big_pilot):
    users_df, gt = big_pilot
    k = len(users_df) // 10
    ceiling = true_incremental_outcome(covariate_ceiling_target_ids(users_df, gt, k), users_df, gt)
    random_mean = k * users_df["segment"].map(gt.segment_effects).mean()
    segment_oracle = users_df["segment"].map(gt.segment_effects).nlargest(k).sum()
    assert random_mean < ceiling < segment_oracle


def test_round_trips_through_json(big_pilot):
    _, gt = big_pilot
    restored = GroundTruth.from_dict(json.loads(json.dumps(gt.to_dict())))
    assert restored == gt
    assert restored.true_uplift("persuadable") == gt.true_uplift("persuadable")


def test_posterior_requires_covariate_process_parameters():
    gt = GroundTruth(
        baseline_rate=0.1, propensity=0.5,
        segment_mix={"a": 1.0}, segment_effects={"a": 0.1},
    )
    with pytest.raises(ValueError, match="spend_means"):
        gt.segment_posterior(np.array([1.0]))


def test_revenue_economics_from_segment_means(big_pilot):
    _, gt = big_pilot
    assert gt.true_revenue_uplift("persuadable") == pytest.approx(0.15 * 40.0)
    assert gt.true_revenue_uplift("sleeping_dog") == pytest.approx(-0.05 * 35.0)
    # sure_thing: tiny incremental revenue, large revenue an incentive pays out on
    assert gt.expected_treated_revenue("sure_thing") == pytest.approx(0.41 * 60.0)
    assert gt.expected_treated_revenue("sure_thing") > 40 * gt.true_revenue_uplift("sure_thing")


def test_revenue_methods_refuse_a_truth_without_revenue_means():
    gt = GroundTruth(
        baseline_rate=0.1, propensity=0.5,
        segment_mix={"a": 1.0}, segment_effects={"a": 0.1},
    )
    with pytest.raises(ValueError, match="predates segment-dependent revenue"):
        gt.true_revenue_uplift("a")
