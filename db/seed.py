"""
Generate a demo pilot from config/default_config.yaml and persist it to the
local dev DB (Phase 1 deliverable, spec Section 6 Phase 1).

Run directly: `python -m db.seed`
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pandas as pd

from config.schema import load_pilot_config
from data_sim.pilot_simulator import PilotSimulator
from db.connection import get_engine, init_schema


def seed_demo_pilot(config_path: str = "config/default_config.yaml") -> str:
    cfg = load_pilot_config(config_path)
    simulator = PilotSimulator(cfg)
    users_df, treatment_df, outcomes_df, ground_truth = simulator.generate()

    engine = get_engine()
    init_schema(engine)

    pilot_id = f"pilot_{uuid.uuid4().hex[:8]}"
    now = datetime.now(timezone.utc)
    pilots_df = pd.DataFrame([{
        "pilot_id": pilot_id,
        "name": "Demo pilot (simulated)",
        "start_date": now.date(),
        "budget_cap": None,
    }])

    # generate() deliberately leaves assigned_at/observed_at unset (NaT) to
    # stay a pure, seed-reproducible function -- this is where the real
    # "when was this actually persisted" timestamp belongs.
    treatment_df = treatment_df.assign(pilot_id=pilot_id, assigned_at=now)
    outcomes_df = outcomes_df.assign(pilot_id=pilot_id, observed_at=now)

    with engine.begin() as conn:
        users_df.to_sql("users", conn, if_exists="append", index=False)
        pilots_df.to_sql("pilots", conn, if_exists="append", index=False)
        treatment_df.to_sql("treatment_assignment", conn, if_exists="append", index=False)
        outcomes_df.to_sql("outcomes", conn, if_exists="append", index=False)

    print(f"Seeded pilot {pilot_id!r} with {len(users_df)} users.\n")
    print("True per-segment parameters (GroundTruth):")
    for seg in sorted(cfg.segment_mix):
        p0 = ground_truth.true_control_probability(seg)
        p1 = ground_truth.true_treated_probability(seg)
        print(
            f"  {seg:<12} control_p={p0:.4f}  treated_p={p1:.4f}  "
            f"true_uplift={p1 - p0:+.4f}"
        )

    return pilot_id


if __name__ == "__main__":
    seed_demo_pilot()
