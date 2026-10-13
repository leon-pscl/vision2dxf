"""Generation settings and the two helpers the cutting-line stage needs.

The ease model itself lives in garment_surface/ease.py; this module only carries the
request-level settings object, which the whole pipeline is parameterised by.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..config import drafting_config, reconstruction_config, seam_allowance


@dataclass
class Settings:
    garment_type: str = "basic_torso"

    # ---- reconstruction ---------------------------------------------------
    section_model: str = "superellipse"       # ellipse | superellipse
    section_exponent: float | None = None     # overrides the configured exponent
    cross_section_resolution: int | None = None
    longitudinal_segments: int | None = None

    # ---- garment surface --------------------------------------------------
    ease_preset: str = "fitted"
    bust_ease_mm: float | None = None
    waist_ease_mm: float | None = None
    hip_ease_mm: float | None = None
    armhole_ease_mm: float | None = None

    # ---- seam network -----------------------------------------------------
    seam_preset: str = "basic_torso_v1"
    front_neck_drop_mm: float | None = None
    back_neck_drop_mm: float | None = None

    # ---- flattening -------------------------------------------------------
    flattening_method: str = "lscm_arap"
    w_distortion: float | None = None
    w_boundary: float | None = None
    w_seam: float | None = None
    w_grain: float | None = None
    w_flip: float | None = None

    # ---- output -----------------------------------------------------------
    seam_allowances: dict[str, float] = field(default_factory=dict)
    cutting_join: str | None = None
    paper_size: str | None = None
    paper_orientation: str | None = None

    # The operator confirms that the file's values are centimetres before anything is
    # converted. Nothing is generated until this is true.
    unit: str = "cm"
    confirm_units: bool = False

    def as_dict(self) -> dict:
        return dict(self.__dict__)

    def exponent(self) -> float | None:
        return self.section_exponent

    def resolution(self) -> int:
        cfg = reconstruction_config()["mesh"]
        return int(self.cross_section_resolution or cfg["cross_section_resolution"])

    def segments(self) -> int:
        cfg = reconstruction_config()["mesh"]
        return int(self.longitudinal_segments or cfg["longitudinal_segments"])

    def weights(self) -> dict[str, float]:
        base = dict(reconstruction_config()["flattening"]["objective"])
        for key in ("w_distortion", "w_boundary", "w_seam", "w_grain", "w_flip"):
            value = getattr(self, key)
            if value is not None:
                base[key] = float(value)
        return base


def resolve_join(settings: Settings) -> str:
    return settings.cutting_join or drafting_config()["cutting_join"]


def allowance_for(role: str, override: dict[str, float] | None = None) -> float:
    if override and role in override:
        return float(override[role])
    return seam_allowance(role)
