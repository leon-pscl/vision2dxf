"""Ease assignment. Ease is applied to girths only; linear dimensions keep their
measured value unless an explicit override is supplied in the settings."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..config import drafting_config, seam_allowance


@dataclass
class Settings:
    garment_type: str = "basic_bodice"
    # Which base block to transform; empty means the catalog default.
    base_block_id: str | None = None
    ease_preset: str = "fitted"
    bust_ease_mm: float | None = None
    waist_ease_mm: float | None = None
    hip_ease_mm: float | None = None
    seam_allowances: dict[str, float] = field(default_factory=dict)
    cutting_join: str | None = None
    paper_size: str | None = None
    paper_orientation: str | None = None
    # Operator confirmation that a target outside the tested transformation envelope may
    # still be transformed. A `stop` envelope status is refused regardless.
    confirm_envelope: bool = False
    unit: str = "cm"

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class Ease:
    bust_mm: float
    waist_mm: float
    hip_mm: float
    preset: str


def resolve_ease(settings: Settings) -> Ease:
    presets = drafting_config()["ease_presets"]
    name = settings.ease_preset if settings.ease_preset in presets else drafting_config()["default_ease_preset"]
    preset = presets[name]
    return Ease(
        bust_mm=settings.bust_ease_mm if settings.bust_ease_mm is not None else preset["bust_mm"],
        waist_mm=settings.waist_ease_mm if settings.waist_ease_mm is not None else preset["waist_mm"],
        hip_mm=settings.hip_ease_mm if settings.hip_ease_mm is not None else preset["hip_mm"],
        preset=name,
    )


def resolve_join(settings: Settings) -> str:
    default = drafting_config()["cutting_join"]
    return settings.cutting_join or default


def allowance_for(role: str, override: dict[str, float] | None = None) -> float:
    if override and role in override:
        return float(override[role])
    return seam_allowance(role)
