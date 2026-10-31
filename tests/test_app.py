"""
Streamlit app: data layer and figure builders unit-tested directly; the
page itself run end to end with Streamlit's AppTest against a freshly
seeded, trained and evaluated temp DB (no browser needed).
"""

from __future__ import annotations

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from app.components import charts, data

APP = "app/streamlit_app.py"


@pytest.fixture
def trained_pilot(engine, small_config_path):
    from core.evaluation import evaluate_model_runs, latest_runs_per_model_type, persist_evaluation
    from core.pipeline import run_pipeline
    from db.seed import seed_demo_pilot

    pilot_id = seed_demo_pilot(small_config_path)
    run_pipeline(engine, pilot_id, split_seed=1)
    evaluations, _ = evaluate_model_runs(
        engine, latest_runs_per_model_type(engine, pilot_id), n_bootstrap=20
    )
    for ev in evaluations:
        persist_evaluation(engine, ev, n_bootstrap=20, ci_level=0.95, seed=42)
    return engine, pilot_id


def test_policy_views_carry_truth_and_select_prefixes(trained_pilot):
    engine, pilot_id = trained_pilot
    views = data.policy_views(engine, pilot_id, cashback_rate=0.1, min_roi=None, n_bootstrap=20)
    assert [v.model_type for v in views] == list(data.MODEL_ORDER)
    for v in views:
        assert "true_net" in v.curve and v.curve["true_net"].iloc[0] == 0.0
        small, large = v.selected(0.1), v.selected(0.5)
        assert large[: len(small)] == small  # budget growth only appends users

    table = data.headline(views, 0.1)
    assert list(table["model"]) == [data.MODEL_LABELS[m] for m in data.MODEL_ORDER]
    assert {"true net", "A/B net", "model-claimed net"} <= set(table.columns)
    assert np.isnan(table.loc[0, "model-claimed net"])  # naive makes no uplift claim

    mix = data.segment_mix(engine, pilot_id, views[1].selected(0.1))
    assert list(mix.index) == list(data.SEGMENT_ORDER)
    assert mix.sum() == pytest.approx(1.0)

    budgets = views[0].curve["budget"]
    random_net = data.random_true_net(engine, pilot_id, 0.1, budgets)
    assert random_net[0] == 0.0 and len(random_net) == len(budgets)


def test_figures_build_with_expected_traces(trained_pilot):
    engine, pilot_id = trained_pilot
    views = data.policy_views(engine, pilot_id, cashback_rate=0.1, min_roi=None, n_bootstrap=20)
    random_net = data.random_true_net(engine, pilot_id, 0.1, views[0].curve["budget"])
    for mode in ("light", "dark"):
        net = charts.policy_net_chart(views, random_net, mode=mode, budget_pct=0.1)
        assert [t.name for t in net.data] == [
            "Naive (propensity)", "T-learner", "X-learner", "Random targeting"
        ]  # simulated -> true-net lines only, no CI bands
        valuation = charts.valuation_chart(data.headline(views, 0.1), mode=mode)
        assert {t.name for t in valuation.data} == {
            "A/B estimate (95% CI)", "Model's own claim", "True net"
        }
        # Rows must stay model categories (a numeric axis silently drops them)
        assert valuation.layout.yaxis.type == "category"
        plotted = [t for t in valuation.data if t.showlegend is False]
        assert all(list(t.y) == list(data.MODEL_LABELS.values()) for t in plotted)
        mixes = {v.model_type: data.segment_mix(engine, pilot_id, v.selected(0.1)) for v in views}
        assert len(charts.segment_mix_chart(mixes, mode=mode).data) == len(data.SEGMENT_ORDER)
        _, curves = data.qini_view(engine, pilot_id)
        qini = charts.qini_chart(curves, mode=mode)
        assert [t.name for t in qini.data if t.showlegend is not False][-1] == "Random targeting"
    # A model keeps its color across charts and modes differ only in step.
    assert charts.model_color("t_learner", "light") != charts.model_color("t_learner", "dark")


def test_real_pilot_chart_falls_back_to_estimates_with_ci_bands(trained_pilot):
    engine, pilot_id = trained_pilot
    views = data.policy_views(engine, pilot_id, cashback_rate=0.1, min_roi=None, n_bootstrap=20)
    for v in views:
        v.curve = v.curve.drop(columns="true_net")  # as for a pilot with no ground truth
    fig = charts.policy_net_chart(views, None)
    assert sum(t.fill == "toself" for t in fig.data) == 3  # one CI band per model
    assert fig.layout.yaxis.title.text.startswith("Estimated net revenue")


def test_app_renders_every_tab_without_errors(trained_pilot):
    _, pilot_id = trained_pilot
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception, at.exception
    assert at.title[0].value == "Who should get the cashback?"
    assert [t.label for t in at.tabs] == [
        "Budget policy", "Model evaluation (Qini)", "Ground-truth check", "How it works"
    ]
    assert len(at.metric) == 4  # simulated pilot -> true-net headline tiles
    assert pilot_id in at.selectbox[0].options[0]

    at.slider[1].set_value(25).run()  # budget
    at.toggle[0].set_value(True).run()  # break-even ROI floor
    assert not at.exception, at.exception


def test_app_explains_what_to_run_when_db_is_empty(engine):
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, at.exception
    assert "python -m core.pipeline" in at.info[0].value
