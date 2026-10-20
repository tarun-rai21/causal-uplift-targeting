"""
GroundTruth: an immutable record of every true parameter used to generate a
pilot. Exists so Phase 6's validation harness has an exact target to check
recovered CATE estimates against -- "did the model recover the true
per-segment uplift" is only checkable if the true values are captured
somewhere, not re-derived from config on demand (config could drift; this
is the frozen record of what was ACTUALLY used for a specific generate()
call).

The dict fields are wrapped in read-only MappingProxyType views on
construction: `frozen=True` alone only blocks attribute reassignment, so
`gt.segment_effects["persuadable"] = 99` would otherwise silently rewrite
the "frozen" truth.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


@dataclass(frozen=True)
class GroundTruth:
    baseline_rate: float
    propensity: float
    segment_mix: Mapping[str, float]
    segment_effects: Mapping[str, float]
    segment_baseline_offsets: Mapping[str, float] = field(default_factory=dict)
    seed: int = 42

    def __post_init__(self) -> None:
        for name in ("segment_mix", "segment_effects", "segment_baseline_offsets"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))

    def true_uplift(self, segment: str) -> float:
        """The true CATE for this segment -- what Phase 6 checks recovered
        predictions against."""
        return self.segment_effects[segment]

    def true_control_probability(self, segment: str) -> float:
        return self.baseline_rate + self.segment_baseline_offsets.get(segment, 0.0)

    def true_treated_probability(self, segment: str) -> float:
        return self.true_control_probability(segment) + self.true_uplift(segment)
