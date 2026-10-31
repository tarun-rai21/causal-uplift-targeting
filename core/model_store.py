"""
Shared persistence for model runs and their per-user uplift predictions.

Every model type (baseline, t_learner, x_learner) writes through this one
module, so model_runs / uplift_predictions rows are structurally identical
regardless of which model produced them -- which is what lets Phase 5's
evaluation module read any model_run_id uniformly, and what makes the
side-by-side comparison a filtered query rather than per-model glue code.

rank_pct is precomputed here (percentile rank of predicted_uplift within
the model_run, descending -- rank_pct=0.01 means "top 1%") because the
Qini SQL in spec Section 4.3 orders by it; computing it once at write time
is cheaper and less error-prone than recomputing per evaluation query.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import text

from core.learners import KNOWN_BASE_LEARNERS


def persist_model_run(
    engine,
    pilot_id: str,
    model_type: str,
    base_learner: str,
    predictions: pd.Series,
) -> str:
    """
    Write one model_runs row and its uplift_predictions rows.

    predictions: Series indexed by user_id, values = predicted uplift
    (or, for model_type='baseline', the raw propensity score used as a
    deliberately-naive proxy -- the schema comment and app labeling make
    clear it is not a causal estimate).

    Returns the generated model_run_id.
    """
    allowed = {"baseline", "t_learner", "x_learner"}
    if model_type not in allowed:
        raise ValueError(f"model_type must be one of {allowed}, got {model_type!r}")
    if base_learner not in KNOWN_BASE_LEARNERS:
        raise ValueError(
            f"base_learner must be one of {KNOWN_BASE_LEARNERS}, got {base_learner!r}"
        )
    if predictions.empty:
        raise ValueError("predictions is empty; refusing to persist an empty model run")
    if predictions.index.has_duplicates:
        raise ValueError("predictions index (user_id) contains duplicates")

    model_run_id = str(uuid.uuid4())

    # Descending percentile rank: highest predicted uplift -> smallest rank_pct.
    rank_pct = predictions.rank(ascending=False, method="first") / len(predictions)

    preds_df = pd.DataFrame(
        {
            "model_run_id": model_run_id,
            "user_id": predictions.index,
            "predicted_uplift": predictions.values,
            "rank_pct": rank_pct.values,
        }
    )

    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO model_runs "
                "(model_run_id, pilot_id, model_type, base_learner, trained_at) "
                "VALUES (:model_run_id, :pilot_id, :model_type, :base_learner, :trained_at)"
            ),
            {
                "model_run_id": model_run_id,
                "pilot_id": pilot_id,
                "model_type": model_type,
                "base_learner": base_learner,
                "trained_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        preds_df.to_sql("uplift_predictions", conn, if_exists="append", index=False)

    return model_run_id


def load_predictions(engine, model_run_id: str) -> pd.Series:
    """Read back predictions for a model_run as a user_id-indexed Series."""
    with engine.connect() as conn:
        result = conn.execute(
            text(
                "SELECT user_id, predicted_uplift FROM uplift_predictions "
                "WHERE model_run_id = :model_run_id"
            ),
            {"model_run_id": model_run_id},
        )
        rows = result.fetchall()
    if not rows:
        raise ValueError(f"no predictions found for model_run_id={model_run_id!r}")
    return pd.Series(
        [r[1] for r in rows], index=[r[0] for r in rows], name="predicted_uplift"
    )
