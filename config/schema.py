"""
Validated configuration for the pilot simulator.

This module exists because the original spec never specifies validation for
simulator config, and every downstream phase (simulator, baseline, T-learner,
X-learner, evaluation, validation harness) implicitly assumes the config it's
handed is well-formed. Left unchecked, malformed config produces plausible-
looking but wrong simulated data — the exact failure mode this entire project
exists to detect in propensity modeling. It would undercut the project's own
thesis to let it happen silently in our own config layer.

Validation rules and why each one exists:

1. segment_mix values must sum to 1.0 (within floating-point tolerance).
   Without this, the simulator would silently under- or over-sample the
   population (e.g., mix summing to 0.9 quietly drops 10% of the population
   into an undefined segment, or numpy's np.random.choice raises a cryptic
   error whose root cause is nowhere near this line).

2. segment_effects and segment_mix must have exactly the same set of keys.
   A segment present in one dict but not the other means either (a) a
   segment gets a treatment effect but is never generated, or (b) a segment
   gets generated but its treatment effect silently defaults to something
   undefined -- both are silent-data-corruption bugs, not crashes.

3. propensity must be strictly inside (0, 1) -- NOT inclusive of 0 or 1.
   The X-learner's combination formula (Section 7.3) is
   tau(x) = e(x)*g0(x) + (1-e(x))*g1(x). At e(x)=0 or e(x)=1, one entire arm
   (treated or control) has zero observations, mu1 or mu0 cannot be fit at
   all, and the "propensity-weighted combination" degenerates into an
   undefined/zero-signal estimate for the entire model, not a boundary case
   worth handling gracefully -- so we reject it at config time instead of
   producing an inscrutable failure three phases downstream.

4. baseline_rate must be inside (0, 1) -- it's a probability.

5. n_users must be a positive integer, and should be large enough that the
   rarest configured segment still has a meaningfully estimable sample size;
   we warn (not error) below a heuristic threshold rather than hard-fail,
   since a small deliberate test pilot is a legitimate use case.

6. The three segments the spec calls "at minimum" (persuadable, sure_thing,
   lost_cause) must be present. sleeping_dog is optional per spec Section 1.3.

7. For every segment, baseline_rate + segment_effects[segment] must be
   within [0, 1] -- unlike the propensity check (rule 3), this is a CLOSED
   interval: a Bernoulli conversion probability of exactly 0 or 1 is
   mathematically valid (a deterministic outcome for that user), it does
   not break any downstream formula the way propensity=0 breaks the
   X-learner's weighting. This rule exists because the simulator (Phase 1)
   computes each treated user's conversion probability as
   p0 + segment_effects[segment], and without this check a large positive
   effect (or baseline_rate) combination would only surface as a cryptic
   numpy error deep inside a Bernoulli draw, far from the config that
   actually caused it.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, Field, field_validator, model_validator

_REQUIRED_SEGMENTS = {"persuadable", "sure_thing", "lost_cause"}
_MIN_RECOMMENDED_SEGMENT_COUNT = 200  # heuristic floor for a segment to be estimable


class PilotConfig(BaseModel):
    n_users: int = Field(gt=0)
    baseline_rate: float = Field(gt=0.0, lt=1.0)
    propensity: float = Field(gt=0.0, lt=1.0)
    seed: int = 42

    segment_effects: dict[str, float]
    segment_mix: dict[str, float]

    @field_validator("segment_mix")
    @classmethod
    def _mix_sums_to_one(cls, v: dict[str, float]) -> dict[str, float]:
        total = sum(v.values())
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(
                f"segment_mix values must sum to 1.0, got {total!r}. "
                f"Values were: {v!r}"
            )
        return v

    @model_validator(mode="after")
    def _segments_match_and_present(self) -> "PilotConfig":
        mix_keys = set(self.segment_mix.keys())
        effect_keys = set(self.segment_effects.keys())

        if mix_keys != effect_keys:
            raise ValueError(
                "segment_mix and segment_effects must have identical segment "
                f"keys. segment_mix has {mix_keys}, segment_effects has "
                f"{effect_keys}. Symmetric difference: "
                f"{mix_keys.symmetric_difference(effect_keys)}"
            )

        missing_required = _REQUIRED_SEGMENTS - mix_keys
        if missing_required:
            raise ValueError(
                f"Missing required segment(s): {missing_required}. "
                f"The spec requires at minimum {_REQUIRED_SEGMENTS} "
                "(sleeping_dog is optional)."
            )

        return self

    @model_validator(mode="after")
    def _warn_on_thin_segments(self) -> "PilotConfig":
        thin_segments = [
            seg for seg, frac in self.segment_mix.items()
            if frac * self.n_users < _MIN_RECOMMENDED_SEGMENT_COUNT
        ]
        if thin_segments:
            import warnings

            warnings.warn(
                f"Segment(s) {thin_segments} will have fewer than "
                f"{_MIN_RECOMMENDED_SEGMENT_COUNT} expected users at "
                f"n_users={self.n_users}. Treatment-effect recovery for thin "
                "segments will be noisy; this is a warning, not an error, "
                "since a small deliberate test pilot is a legitimate use case.",
                stacklevel=2,
            )
        return self

    @model_validator(mode="after")
    def _treated_probability_stays_in_bounds(self) -> "PilotConfig":
        """
        For every segment, the treated conversion probability
        (baseline_rate + segment_effects[segment]) must land in [0, 1].
        Checked per-segment, not just against the largest/smallest effect,
        so the error message names exactly which segment is at fault.
        """
        out_of_bounds = {}
        for segment, effect in self.segment_effects.items():
            treated_p = self.baseline_rate + effect
            if not (0.0 <= treated_p <= 1.0):
                out_of_bounds[segment] = treated_p

        if out_of_bounds:
            details = ", ".join(
                f"{seg!r}: baseline_rate({self.baseline_rate}) + "
                f"effect({self.segment_effects[seg]}) = {p:.4f}"
                for seg, p in out_of_bounds.items()
            )
            raise ValueError(
                "Treated conversion probability out of [0, 1] bounds for "
                f"segment(s): {details}. Reduce baseline_rate or the "
                "offending segment_effects value(s)."
            )
        return self


def load_pilot_config(path: str) -> PilotConfig:
    """Load and validate a YAML pilot config file."""
    import yaml

    with open(path) as f:
        raw = yaml.safe_load(f)
    return PilotConfig(**raw)
