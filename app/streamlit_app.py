"""
Streamlit policy simulator (Phase 9).

    streamlit run app/streamlit_app.py

Reads whatever the pipeline has stored ($DATABASE_URL or ./causal_uplift.db):
    python -m db.seed && python -m core.pipeline && python -m core.evaluation

Presentation only: every number comes from app/components/data.py, which
calls the project's own modules.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

# `streamlit run app/streamlit_app.py` puts app/ (not the repo root) on
# sys.path; the project's packages live at the root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.components import charts, data  # noqa: E402
from db.connection import get_database_url, get_engine, init_schema  # noqa: E402

st.set_page_config(page_title="Uplift Targeting", page_icon="🎯", layout="wide")


@st.cache_resource
def _engine(db_url: str):
    engine = get_engine()
    init_schema(engine)
    return engine


@st.cache_resource
def _build_lock() -> threading.Lock:
    """One lock per server process: two visitors arriving at an empty
    deployment at once must not both build a demo pilot."""
    return threading.Lock()


@st.cache_data(show_spinner="Building policy curves (bootstrap per budget point)...")
def _policy_views(db_url: str, pilot_id: str, rate: float, floor: float | None):
    return data.policy_views(_engine(db_url), pilot_id, rate, floor)


@st.cache_data
def _random_net(db_url: str, pilot_id: str, rate: float, budgets: tuple):
    return data.random_true_net(_engine(db_url), pilot_id, rate, list(budgets))


@st.cache_data
def _qini(db_url: str, pilot_id: str):
    return data.qini_view(_engine(db_url), pilot_id)


@st.cache_data(show_spinner="Scoring stored models against the simulator's truth...")
def _truth_check(db_url: str, pilot_id: str):
    from validation.harness import validate_stored_pilot

    return validate_stored_pilot(_engine(db_url), pilot_id, n_bootstrap=200)


def _money(x) -> str:
    return "n/a" if pd.isna(x) else f"{x:,.0f}"


db_url = get_database_url()
engine = _engine(db_url)
mode = "dark" if getattr(st.context.theme, "type", None) == "dark" else "light"

ready = data.ready_pilots(engine)

st.title("Who should get the cashback?")
st.caption(
    "Budget-constrained incentive targeting by **causal uplift** -- who converts *because* "
    "of the cashback -- versus the industry-default **propensity** model -- who converts at all."
)

if ready.empty:
    # A fresh deployment starts with an empty DB. UPLIFT_AUTO_DEMO=1 (e.g. a
    # Streamlit Cloud secret) builds the demo pilot on first visit; otherwise
    # the visitor can trigger it -- nothing is built behind anyone's back.
    st.info(
        "No pilot with trained models yet. From the repo root run:\n\n"
        "```\npython -m db.seed\npython -m core.pipeline\npython -m core.evaluation\n```\n\n"
        "or build the demo pilot right here."
    )
    auto = os.environ.get("UPLIFT_AUTO_DEMO") == "1"
    if auto or st.button("Build the demo pilot (simulate 50,000 users, train, evaluate)"):
        with _build_lock(), st.spinner("Building the demo pilot -- about 1-2 minutes, once..."):
            if data.ready_pilots(engine).empty:  # another session may have built it
                data.build_demo_pilot(engine)
        st.cache_data.clear()
        st.rerun()
    st.stop()

with st.sidebar:
    st.header("Scenario")
    labels = {
        r.pilot_id: f"{r.pilot_id} ({'sim' if r.simulated else 'real'}, {r.n_users / 1000:.0f}k)"
        for r in ready.itertuples()
    }
    pilot_id = st.selectbox("Pilot", list(labels), format_func=labels.get)
    rate_pct = st.slider("Cashback rate (% of revenue, paid per treated conversion)", 1, 30, 10)
    budget_pct = st.slider("Budget (% of the cost of treating everyone)", 5, 100, 10, step=5)
    floor = 1.0 if st.toggle("Only treat users expected to break even (ROI ≥ 1)") else None
    # File name only: a full local path leaks the machine's directory layout
    # into anything screenshotted or shared.
    source = Path(db_url.split("///")[-1]).name if db_url.startswith("sqlite") else "DATABASE_URL"
    st.caption(f"Data: `{source}`")

rate = rate_pct / 100
views = _policy_views(db_url, pilot_id, rate, floor)
has_truth = all("true_net" in v.curve for v in views)
budget = budget_pct / 100
table = data.headline(views, budget)

policy_tab, qini_tab, truth_tab, about_tab = st.tabs(
    ["Budget policy", "Model evaluation (Qini)", "Ground-truth check", "How it works"]
)

with policy_tab:
    if has_truth:
        by_model = table.set_index("model")
        best = by_model.drop(index=data.MODEL_LABELS["baseline"])["true net"].idxmax()
        random_now = _random_net(
            db_url, pilot_id, rate, (budget * float(views[0].inputs["expected_cost"].sum()),)
        )[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"True net · {best}", _money(by_model.loc[best, "true net"]), border=True)
        c2.metric(
            "True net · Naive", _money(by_model.loc[data.MODEL_LABELS["baseline"], "true net"]),
            border=True,
        )
        c3.metric("True net · Random targeting", _money(random_now), border=True)
        c4.metric(
            f"{best} claims", _money(by_model.loc[best, "model-claimed net"]), border=True,
            help="The model's own estimate over the users it picked -- inflated by that selection.",
        )

    st.subheader("Net revenue by budget")
    random_curve = (
        _random_net(db_url, pilot_id, rate, tuple(views[0].curve["budget"])) if has_truth else None
    )
    st.plotly_chart(
        charts.policy_net_chart(views, random_curve, mode=mode, budget_pct=budget), theme=None
    )
    st.caption(
        "Simulated pilot: lines are TRUE net revenue (incremental revenue minus cashback paid). "
        "Uplift models rank users by estimated ROI; the naive model by P(convert)."
        if has_truth else
        "Real pilot: lines are outcome-based estimates from the randomized holdout, "
        "with 95% CI bands."
    )

    left, right = st.columns(2)
    with left:
        st.subheader(f"What the {budget_pct}% budget is worth")
        st.plotly_chart(charts.valuation_chart(table, mode=mode), theme=None)
        st.caption(
            "Hollow: the model's own claim. Filled with whisker: estimate from randomized "
            "outcomes (unbiased; 95% CI). Diamond: the truth, when simulated."
        )
    with right:
        if has_truth:
            st.subheader("Who gets targeted")
            mixes = {
                v.model_type: data.segment_mix(engine, pilot_id, v.selected(budget))
                for v in views
            }
            st.plotly_chart(charts.segment_mix_chart(mixes, mode=mode), theme=None)
            st.caption(
                "True (hidden) segments of the targeted users. Sure things convert anyway and "
                "collect cashback; persuadables are who the incentive is for."
            )
    with st.expander("Table view"):
        money = [c for c in table.columns if c not in ("model", "users targeted")]
        st.dataframe(
            table.assign(**{c: table[c].round(0) for c in money}), hide_index=True,
            column_config={c: st.column_config.NumberColumn(format="localized") for c in money},
        )

with qini_tab:
    scores, curves = _qini(db_url, pilot_id)
    if scores is None:
        st.info(f"No stored evaluation for this pilot. Run:\n\n"
                f"```\npython -m core.evaluation --pilot-id {pilot_id}\n```")
    else:
        bands = st.toggle("Show 95% bands")
        st.plotly_chart(charts.qini_chart(curves, mode=mode, bands=bands), theme=None)
        st.dataframe(data.qini_table(scores), hide_index=True)
        st.caption(
            "Observed-outcome metrics -- no ground truth used, so they apply to real pilots. "
            "On a ~10k-user holdout the intervals are wider than the gaps between models: "
            "compare intervals, not point estimates."
        )

with truth_tab:
    if not has_truth:
        st.info("Ground truth exists only for simulated pilots (seeded via `python -m db.seed`).")
    else:
        result = _truth_check(db_url, pilot_id)
        from validation.harness import CLAIMS

        claims = pd.DataFrame(
            [{"claim": desc,
              "holds": {True: "✓ yes", False: "✗ no", None: "n/a"}[result.claims[name]]}
             for name, (desc, _) in CLAIMS.items()]
        )
        st.subheader("README claims, checked on this pilot")
        # Sized to fit: an inner scroll hid the last claims (screenshot review).
        st.dataframe(claims, hide_index=True, height=35 * (len(claims) + 1) + 3)
        st.caption(
            "One pilot is one draw. `python -m validation.harness` checks every claim "
            "across seeds and scenarios (reports/validation_report.md)."
        )
        st.subheader("Ranking quality against the truth")
        st.dataframe(data.ranking_table(result.oracle))
        st.caption(
            "Top-k columns: true incremental conversions from each model's top k% of test "
            "users. No RMSE for naive: its score is a probability, not an uplift estimate."
        )

with about_tab:
    st.markdown(Path(__file__).with_name("ABOUT.md").read_text(encoding="utf-8"))
