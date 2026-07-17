"""
PilotSimulator: generates synthetic randomized-pilot data with known,
segment-varying heterogeneous treatment effects (spec Section 7.1).

Generation logic, in order:
1. Assign each user a segment by sampling from segment_mix. Segment
   determines true control/treated conversion probability, but is NOT
   itself a model feature -- only observable covariates are, matching a
   real deployment where segment membership is unobservable.
2. Generate observable covariates (historical_spend, region, device_type,
   signup_date) correlated with, but overlapping across, segment. Overlap
   is deliberate: if segments were perfectly separable from covariates,
   CATE estimation would be trivially easy and not representative of real
   difficulty.
3. Assign treatment via Bernoulli(propensity), independent of segment
   (true randomization).
4. Control-arm conversion probability = baseline_rate + segment_baseline_
   offsets[segment]. Treated-arm = control + segment_effects[segment].
   PilotConfig's own validation already guarantees these stay in [0, 1]
   for every segment before a PilotSimulator can even be constructed.
5. Draw the binary outcome from Bernoulli(conversion probability); if
   converted, draw revenue from a log-normal distribution.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from config.schema import PilotConfig
from data_sim.ground_truth import GroundTruth

_REGIONS = ["north", "south", "east", "west"]
_DEVICES = ["mobile", "desktop", "tablet"]

# Per-segment mean historical_spend, used to give covariates a realistic,
# overlapping (not perfectly separable) correlation with segment.
# Unrecognized segment names fall back to a neutral mean.
_SEGMENT_SPEND_MEAN = {
    "persuadable": 120.0,
    "sure_thing": 200.0,
    "lost_cause": 60.0,
    "sleeping_dog": 90.0,
}
_SPEND_NOISE_STD = 80.0  # deliberately large relative to the mean gaps above


class PilotSimulator:
    def __init__(self, config: PilotConfig):
        self.config = config
        self._rng = np.random.default_rng(config.seed)

    def generate(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, GroundTruth]:
        cfg = self.config
        rng = self._rng
        n = cfg.n_users

        segments = np.array(list(cfg.segment_mix.keys()))
        mix_probs = np.array([cfg.segment_mix[s] for s in segments])
        segment_labels = rng.choice(segments, size=n, p=mix_probs)

        user_ids = np.array([f"u_{i:08d}" for i in range(n)])

        users_df = pd.DataFrame({
            "user_id": user_ids,
            "signup_date": self._generate_signup_dates(n, rng),
            "segment": segment_labels,
            "historical_spend": self._generate_historical_spend(segment_labels, rng),
            "region": rng.choice(_REGIONS, size=n),
            "device_type": rng.choice(_DEVICES, size=n),
        })

        treated = rng.random(n) < cfg.propensity
        # NOTE: assigned_at is intentionally NOT set here. generate() must be
        # a pure function of `config` (fully reproducible under a fixed
        # seed, per NFR2) -- a wall-clock call here would make two calls
        # with the identical seed produce different output, which is
        # exactly the bug this comment replaced (caught by
        # test_fixed_seed_is_reproducible). The real "when was this pilot
        # ingested" timestamp is stamped later, at actual persistence time,
        # in db/seed.py.
        treatment_df = pd.DataFrame({
            "user_id": user_ids,
            "treated": treated,
            "propensity_e": cfg.propensity,
            "assigned_at": pd.NaT,
        })

        control_p = np.array([
            cfg.baseline_rate + cfg.segment_baseline_offsets.get(s, 0.0)
            for s in segment_labels
        ])
        effect = np.array([cfg.segment_effects[s] for s in segment_labels])
        conversion_prob = np.where(treated, control_p + effect, control_p)
        # PilotConfig validation already guarantees this stays in [0, 1] for
        # every segment; clip anyway as a defense-in-depth safety net against
        # floating-point edge cases, not as a substitute for that validation.
        conversion_prob = np.clip(conversion_prob, 0.0, 1.0)

        converted = rng.random(n) < conversion_prob
        revenue = np.where(
            converted,
            rng.lognormal(mean=3.5, sigma=0.6, size=n),
            0.0,
        )
        outcomes_df = pd.DataFrame({
            "user_id": user_ids,
            "converted": converted,
            "revenue": revenue,
            "observed_at": pd.NaT,
        })

        ground_truth = GroundTruth(
            baseline_rate=cfg.baseline_rate,
            propensity=cfg.propensity,
            segment_mix=dict(cfg.segment_mix),
            segment_effects=dict(cfg.segment_effects),
            segment_baseline_offsets=dict(cfg.segment_baseline_offsets),
            seed=cfg.seed,
        )

        return users_df, treatment_df, outcomes_df, ground_truth

    def _generate_historical_spend(
        self, segment_labels: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        means = np.array(
            [_SEGMENT_SPEND_MEAN.get(s, 100.0) for s in segment_labels]
        )
        noise = rng.normal(loc=0.0, scale=_SPEND_NOISE_STD, size=len(segment_labels))
        return np.clip(means + noise, 0.0, None)

    def _generate_signup_dates(self, n: int, rng: np.random.Generator) -> list[date]:
        start = date(2023, 1, 1)
        offsets = rng.integers(0, 700, size=n)
        return [start + timedelta(days=int(o)) for o in offsets]
