"""
GroundTruth: an immutable record of every true parameter used to generate a
pilot. Exists so Phase 6's validation harness has an exact target to check
recovered CATE estimates against -- "did the model recover the true
per-segment uplift" is only checkable if the true values are captured
somewhere, not re-derived from config on demand (config could drift; this
is the frozen record of what was ACTUALLY used for a specific generate()
call). It is persisted alongside the pilot (pilots.ground_truth_json, see
db/seed.py) so a pilot loaded back from the DB carries its own truth.

The dict fields are wrapped in read-only MappingProxyType views on
construction: `frozen=True` alone only blocks attribute reassignment, so
`gt.segment_effects["persuadable"] = 99` would otherwise silently rewrite
the "frozen" truth.

Two different "true uplift" targets live here, and Phase 6 must use the
right one for each question:

- true_uplift(segment): the per-SEGMENT effect. Summed over a target list,
  it is that list's true incremental outcome (what targeting actually
  buys). Segment is hidden from models, though, so no model can be expected
  to rank by it.
- true_cate_given_covariates(users_df): E[tau | X], the best ranking ANY
  model can achieve from the observable covariates. This is the estimand a
  CATE model is actually trying to recover, and the ceiling it should be
  judged against -- on the default config the X-achievable top-decile
  outcome is roughly a third of the segment-oracle one, so judging models
  against segment truth would report a "failure" that is really just the
  simulator's covariate overlap.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
import pandas as pd
from scipy.stats import norm
from sqlalchemy import text

_MAPPING_FIELDS = (
    "segment_mix", "segment_effects", "segment_baseline_offsets", "spend_means", "revenue_means",
)


@dataclass(frozen=True)
class GroundTruth:
    baseline_rate: float
    propensity: float
    segment_mix: Mapping[str, float]
    segment_effects: Mapping[str, float]
    segment_baseline_offsets: Mapping[str, float] = field(default_factory=dict)
    seed: int = 42
    # Covariate-generating process: historical_spend ~ N(spend_means[s],
    # spend_noise_std), clipped at 0. Needed for E[tau | X]; left empty /
    # None by callers that only need per-segment truth.
    spend_means: Mapping[str, float] = field(default_factory=dict)
    spend_noise_std: float | None = None
    # Revenue per conversion ~ lognormal with mean revenue_means[s] and
    # shape revenue_sigma, independent of treatment. Empty / None on
    # GroundTruths that predate segment-dependent revenue.
    revenue_means: Mapping[str, float] = field(default_factory=dict)
    revenue_sigma: float | None = None

    def __post_init__(self) -> None:
        for name in _MAPPING_FIELDS:
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))

    def true_uplift(self, segment: str) -> float:
        """The true CATE for this segment -- what Phase 6 checks recovered
        predictions against."""
        return self.segment_effects[segment]

    def true_control_probability(self, segment: str) -> float:
        return self.baseline_rate + self.segment_baseline_offsets.get(segment, 0.0)

    def true_treated_probability(self, segment: str) -> float:
        return self.true_control_probability(segment) + self.true_uplift(segment)

    def true_revenue_uplift(self, segment: str) -> float:
        """Expected incremental revenue per user from treating this segment:
        effect * revenue per conversion (basket size does not depend on
        treatment, so the whole uplift comes through conversion)."""
        return self.true_uplift(segment) * self._revenue_mean(segment)

    def expected_treated_revenue(self, segment: str) -> float:
        """Expected revenue per TREATED user -- what a conversion-triggered
        incentive (cashback, discount) is paid against."""
        return self.true_treated_probability(segment) * self._revenue_mean(segment)

    def _revenue_mean(self, segment: str) -> float:
        if not self.revenue_means:
            raise ValueError(
                "GroundTruth has no revenue_means; it predates segment-dependent "
                "revenue -- regenerate the pilot"
            )
        return self.revenue_means[segment]

    def segment_posterior(self, historical_spend: np.ndarray) -> pd.DataFrame:
        """
        P(segment | X) for each user, one column per segment, rows sum to 1.

        Only historical_spend enters: the simulator draws region, device and
        signup_date independently of segment, so they cancel out of Bayes'
        rule and P(segment | X) = P(segment | historical_spend).

        The clip at 0 turns spend into a mixed distribution: a continuous
        density for spend > 0 plus a point mass Phi(-mu/sigma) at exactly 0.
        Each user's likelihood is taken from whichever part they fall in;
        within a single user every segment is compared on the same part,
        so the posterior is well defined either way.
        """
        if self.spend_noise_std is None or not self.spend_means:
            raise ValueError(
                "GroundTruth has no covariate-process parameters (spend_means / "
                "spend_noise_std); construct it via PilotSimulator.generate()"
            )
        x = np.asarray(historical_spend, dtype=float)
        segments = list(self.segment_mix)
        at_zero = x <= 0.0
        likelihood = np.column_stack([
            np.where(
                at_zero,
                norm.cdf(0.0, self.spend_means[s], self.spend_noise_std),
                norm.pdf(x, self.spend_means[s], self.spend_noise_std),
            ) * self.segment_mix[s]
            for s in segments
        ])
        return pd.DataFrame(
            likelihood / likelihood.sum(axis=1, keepdims=True), columns=segments
        )

    def true_cate_given_covariates(self, users_df: pd.DataFrame) -> pd.Series:
        """E[tau | X] = sum_s P(s | X) * tau_s, user_id-indexed."""
        posterior = self.segment_posterior(users_df["historical_spend"].to_numpy())
        effects = np.array([self.segment_effects[s] for s in posterior.columns])
        return pd.Series(
            posterior.to_numpy() @ effects,
            index=users_df["user_id"].values,
            name="true_cate_given_x",
        )

    def economics_given_covariates(self, users_df: pd.DataFrame) -> pd.DataFrame:
        """
        Covariate-level truth for the policy layer, user_id-indexed:
            incremental_revenue = E[tau * m | X] = sum_s P(s|X) tau_s m_s
            treated_revenue     = E[p1 * m | X]  = sum_s P(s|X) p1_s m_s
        A cashback policy at rate r costs r * treated_revenue per treated
        user, so ranking by incremental_revenue / treated_revenue is the
        best ROI ordering any model could achieve from covariates.
        """
        posterior = self.segment_posterior(users_df["historical_spend"].to_numpy())
        segments = list(posterior.columns)
        incr = np.array([self.true_revenue_uplift(s) for s in segments])
        treated = np.array([self.expected_treated_revenue(s) for s in segments])
        return pd.DataFrame(
            {
                "incremental_revenue": posterior.to_numpy() @ incr,
                "treated_revenue": posterior.to_numpy() @ treated,
            },
            index=users_df["user_id"].values,
        )

    def to_dict(self) -> dict:
        """JSON-serializable form (dataclasses.asdict cannot deep-copy the
        MappingProxyType fields)."""
        out = {
            "baseline_rate": self.baseline_rate,
            "propensity": self.propensity,
            "seed": self.seed,
            "spend_noise_std": self.spend_noise_std,
            "revenue_sigma": self.revenue_sigma,
        }
        out.update({name: dict(getattr(self, name)) for name in _MAPPING_FIELDS})
        return out

    @classmethod
    def from_dict(cls, data: Mapping) -> GroundTruth:
        return cls(**data)


def load_ground_truth(engine, pilot_id: str) -> GroundTruth:
    """Read back the GroundTruth persisted with a simulated pilot."""
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT ground_truth_json FROM pilots WHERE pilot_id = :pilot_id"),
            {"pilot_id": pilot_id},
        ).fetchone()
    if row is None:
        raise ValueError(f"no pilot found with pilot_id={pilot_id!r}")
    if row[0] is None:
        raise ValueError(
            f"pilot {pilot_id!r} has no ground truth -- only simulated pilots "
            "(seeded via db/seed.py) carry one; real pilots must be evaluated "
            "with Qini/AUUC, not the oracle"
        )
    return GroundTruth.from_dict(json.loads(row[0]))
