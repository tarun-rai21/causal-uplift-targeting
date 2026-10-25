"""
DB-backed training pipeline: the first code path that wires the individually
tested pieces together end to end.

    pilot in DB (db/seed.py)
      -> load_pilot_frames        users / treatment / outcomes for one pilot
      -> ensure_split             persisted train/test split (computed once)
      -> fit on TRAIN rows        baseline + T-learner + X-learner
      -> predict on TEST rows     persisted via core/model_store.py
      -> economics on TEST rows   P(convert | treated), revenue per conversion,
                                  persisted for the policy layer (core/policy.py)

Predictions are written for TEST users only. Phase 5's Qini/AUUC evaluation
reads a model_run's predictions and compares treated vs. control outcomes
among them, so persisting train-set scores would invite exactly the
evaluate-on-training-data mistake the split exists to prevent; rank_pct is
likewise a percentile within the held-out set.

Run directly: `python -m core.pipeline [--pilot-id PILOT_ID]`
(defaults to the most recently started pilot).
"""

from __future__ import annotations

import argparse

import pandas as pd
from sqlalchemy import text

from core.baseline import fit_propensity_model, predict_propensity
from core.data_split import compute_data_split, load_split_user_ids, persist_split
from core.economics import fit_economics
from core.model_store import persist_model_run
from core.policy import persist_economics
from core.t_learner import fit_t_learner
from core.x_learner import fit_x_learner

BASE_LEARNER = "gradient_boosting"  # core/learners.default_base_learner


def load_pilot_frames(
    engine, pilot_id: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Read one pilot's users, treatment assignment, and outcomes, ordered by
    user_id. Booleans are coerced back to bool: SQLite returns BOOLEAN
    columns as 0/1 integers, while Postgres returns real booleans.
    """
    params = {"pilot_id": pilot_id}
    with engine.connect() as conn:
        users_df = pd.read_sql(
            text(
                "SELECT u.user_id, u.signup_date, u.segment, u.historical_spend, "
                "u.region, u.device_type "
                "FROM users u JOIN treatment_assignment t ON t.user_id = u.user_id "
                "WHERE t.pilot_id = :pilot_id ORDER BY u.user_id"
            ),
            conn,
            params=params,
        )
        treatment_df = pd.read_sql(
            text(
                "SELECT user_id, treated, propensity_e FROM treatment_assignment "
                "WHERE pilot_id = :pilot_id ORDER BY user_id"
            ),
            conn,
            params=params,
        )
        outcomes_df = pd.read_sql(
            text(
                "SELECT user_id, converted, revenue FROM outcomes "
                "WHERE pilot_id = :pilot_id ORDER BY user_id"
            ),
            conn,
            params=params,
        )

    if users_df.empty:
        raise ValueError(f"no users found for pilot_id={pilot_id!r}")
    if not (
        users_df["user_id"].tolist()
        == treatment_df["user_id"].tolist()
        == outcomes_df["user_id"].tolist()
    ):
        raise ValueError(
            f"pilot {pilot_id!r}: users / treatment_assignment / outcomes do not "
            "cover the same user set"
        )
    treatment_df["treated"] = treatment_df["treated"].astype(bool)
    outcomes_df["converted"] = outcomes_df["converted"].astype(bool)
    return users_df, treatment_df, outcomes_df


def ensure_split(
    engine,
    pilot_id: str,
    treatment_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    test_size: float,
    split_seed: int,
) -> None:
    """
    Persist a split for the pilot unless one already exists. The persisted
    split is the source of truth: every model trained on this pilot must see
    the same held-out set, so an existing split is reused rather than
    recomputed -- and a request for a DIFFERENT split_seed is an error, not
    a silent reuse that would leave the caller believing their seed applied.
    """
    with engine.connect() as conn:
        existing = [
            row[0]
            for row in conn.execute(
                text("SELECT DISTINCT split_seed FROM data_splits WHERE pilot_id = :p"),
                {"p": pilot_id},
            )
        ]
    if existing:
        if existing != [split_seed]:
            raise ValueError(
                f"pilot {pilot_id!r} already has a split with split_seed={existing}; "
                f"refusing to train against a different one (requested {split_seed})"
            )
        return
    persist_split(
        engine,
        compute_data_split(
            treatment_df, outcomes_df, pilot_id, test_size=test_size, split_seed=split_seed
        ),
    )


def run_pipeline(
    engine,
    pilot_id: str,
    test_size: float = 0.2,
    split_seed: int = 42,
    random_state: int = 42,
) -> dict[str, str]:
    """Train baseline, T- and X-learner for a pilot; return {model_type: model_run_id}."""
    users_df, treatment_df, outcomes_df = load_pilot_frames(engine, pilot_id)
    ensure_split(engine, pilot_id, treatment_df, outcomes_df, test_size, split_seed)

    train_ids = set(load_split_user_ids(engine, pilot_id, "train"))
    test_ids = set(load_split_user_ids(engine, pilot_id, "test"))

    def subset(df: pd.DataFrame, ids: set[str]) -> pd.DataFrame:
        return df[df["user_id"].isin(ids)].reset_index(drop=True)

    train_users = subset(users_df, train_ids)
    train_treatment = subset(treatment_df, train_ids)
    train_outcomes = subset(outcomes_df, train_ids)
    test_users = subset(users_df, test_ids)

    baseline = fit_propensity_model(train_users, train_outcomes, random_state=random_state)
    t_learner = fit_t_learner(
        train_users, train_treatment, train_outcomes, random_state=random_state
    )
    x_learner = fit_x_learner(
        train_users, train_treatment, train_outcomes, random_state=random_state
    )
    economics = fit_economics(
        train_users, train_treatment, train_outcomes, random_state=random_state
    )
    persist_economics(engine, pilot_id, economics.predict(test_users))

    return {
        "baseline": persist_model_run(
            engine, pilot_id, "baseline", BASE_LEARNER,
            predict_propensity(baseline, test_users),
        ),
        "t_learner": persist_model_run(
            engine, pilot_id, "t_learner", BASE_LEARNER,
            t_learner.predict_cate(test_users),
        ),
        "x_learner": persist_model_run(
            engine, pilot_id, "x_learner", BASE_LEARNER,
            x_learner.predict_cate(test_users),
        ),
    }


def latest_pilot_id(engine) -> str:
    """The pilot with the latest start_date. start_date is a DATE, so two
    pilots seeded the same day tie -- that is an error, not an arbitrary
    pick, since training the wrong pilot would be silent."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT pilot_id FROM pilots "
                "WHERE start_date = (SELECT MAX(start_date) FROM pilots)"
            )
        ).fetchall()
    if not rows:
        raise ValueError("no pilots in the database -- run `python -m db.seed` first")
    if len(rows) > 1:
        raise ValueError(
            f"several pilots share the latest start_date: {sorted(r[0] for r in rows)}; "
            "pass --pilot-id"
        )
    return rows[0][0]


def main() -> None:
    from db.connection import get_engine, init_schema

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pilot-id", help="defaults to the most recently started pilot")
    args = parser.parse_args()

    engine = get_engine()
    init_schema(engine)
    pilot_id = args.pilot_id or latest_pilot_id(engine)
    for model_type, run_id in run_pipeline(engine, pilot_id).items():
        print(f"{pilot_id}  {model_type:<10} model_run_id={run_id}")


if __name__ == "__main__":
    main()
