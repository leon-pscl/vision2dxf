from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class MeasurementProfile:
    profile_id: str
    chest: float
    waist: float
    hip: float
    shoulder: float
    torso_length: float
    armhole_depth: float
    sleeve_length: float
    ease: float


@dataclass
class PatternResult:
    design_id: str
    points: dict[str, tuple[float, float]]
    segments: list[dict]
    valid: bool
    validation_errors: list[str]
    metadata: dict = field(default_factory=dict)


class PatternGenerator(Protocol):
    design_id: str

    def generate(self, measurements: MeasurementProfile) -> PatternResult: ...
