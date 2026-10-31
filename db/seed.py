"""
Generate a demo pilot from a pilot config (default: config/default_config.yaml)
and persist it to the DB (Phase 1 deliverable, spec Section 6 Phase 1).

Run directly: `python -m db.seed [--config PATH]`
"""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config.schema import DEFAULT_CONFIG_PATH, load_pilot_config
from data_sim.pilot_simulator import PilotSimulator
from db.connection import get_engine, init_schema


def seed_demo_pilot(config_path: str | Path = DEFAULT_CONFIG_PATH) -> str:
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
        # Persisted with the pilot so validation reads back the truth that
        # was ACTUALLY used, instead of re-deriving it from a YAML file that
        # may have changed since (see data_sim/ground_truth.py).
        "ground_truth_json": json.dumps(ground_truth.to_dict(), sort_keys=True),
    }])

    # The simulator creates a brand-new synthetic population per call and
    # numbers it u_00000000.. every time, but users.user_id is a global
    # primary key (real users can join many pilots). Without namespacing, a
    # second seed collides on the PK -- or, worse, would silently attach a
    # new pilot's outcomes to a previous pilot's covariates.
    users_df, treatment_df, outcomes_df = (
        df.assign(user_id=pilot_id + ":" + df["user_id"])
        for df in (users_df, treatment_df, outcomes_df)
    )

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate a pilot and store it")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                        help="pilot config YAML (default: config/default_config.yaml)")
    seed_demo_pilot(parser.parse_args().config)


if __name__ == "__main__":
    main()
