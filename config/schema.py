"""
Validated configuration for the pilot simulator.

This module exists because the original spec never specifies validation for
simulator config, and every downstream phase (simulator, baseline, T-learner,
X-learner, evaluation, validation harness) implicitly assumes the config it's
handed is well-formed. Left unchecked, malformed config produces plausible-
looking but wrong simulated data -- the exact failure mode this entire
project exists to detect in propensity modeling.

Validation rules and why each one exists:

1. segment_mix values must sum to 1.0 (within floating-point tolerance).

2. segment_effects and segment_mix must have exactly the same set of keys.

3. propensity must be strictly inside (0, 1) -- not inclusive. The
   X-learner's combination formula tau(x) = e(x)*g0(x) + (1-e(x))*g1(x)
   degenerates at e(x)=0 or 1 (one entire arm has zero observations).

4. baseline_rate must be inside (0, 1) -- it's a probability.

5. n_users must be positive; segments below a heuristic estimable-size
   floor produce a warning (not an error).

6. The three "at minimum" segments (persuadable, sure_thing, lost_cause)
   must be present. sleeping_dog is optional.

7. segment_baseline_offsets (optional; unlisted segments default to an
   offset of 0.0) is added to baseline_rate to get each segment's
   CONTROL-arm conversion probability. This exists because a single shared
   baseline_rate across all segments would make the Phase 2 naive-model
   failure demo a strawman: covariates would only correlate with
   conversion via the treatment effect itself, diluted across the ~50%
   control group, so the naive model would accidentally rank persuadables
   highly instead of demonstrating its real failure (chasing "sure
   things"). Any key present in segment_baseline_offsets must be a real
   segment (subset of segment_mix's keys).

8. For every segment: control probability (baseline_rate + offset) and
   treated probability (control + segment_effects[segment]) must both be
   within the CLOSED interval [0, 1]. Checked per-segment so the error
   names exactly which segment and which arm (control vs. treated) is at
   fault.
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
    segment_baseline_offsets: dict[str, float] = Field(default_factory=dict)

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
    def _baseline_offset_keys_are_known_segments(self) -> "PilotConfig":
        mix_keys = set(self.segment_mix.keys())
        offset_keys = set(self.segment_baseline_offsets.keys())
        unknown = offset_keys - mix_keys
        if unknown:
            raise ValueError(
                f"segment_baseline_offsets has segment(s) not present in "
                f"segment_mix: {unknown}. segment_baseline_offsets is "
                "optional per-segment (missing entries default to an "
                "offset of 0.0), but any key present must be a real segment."
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
    def _segment_probabilities_stay_in_bounds(self) -> "PilotConfig":
        problems = []
        for segment in self.segment_mix:
            offset = self.segment_baseline_offsets.get(segment, 0.0)
            control_p = self.baseline_rate + offset
            treated_p = control_p + self.segment_effects[segment]

            if not (0.0 <= control_p <= 1.0):
                problems.append(
                    f"{segment!r} control probability = "
                    f"baseline_rate({self.baseline_rate}) + offset({offset}) "
                    f"= {control_p:.4f}"
                )
            if not (0.0 <= treated_p <= 1.0):
                problems.append(
                    f"{segment!r} treated probability = "
                    f"control({control_p:.4f}) + "
                    f"effect({self.segment_effects[segment]}) = {treated_p:.4f}"
                )

        if problems:
            raise ValueError(
                "Segment conversion probability out of [0, 1] bounds: "
                + "; ".join(problems)
                + ". Adjust baseline_rate, segment_baseline_offsets, or "
                "segment_effects."
            )
        return self


def load_pilot_config(path: str) -> PilotConfig:
    """Load and validate a YAML pilot config file."""
    import yaml

    with open(path) as f:
        raw = yaml.safe_load(f)
    return PilotConfig(**raw)
