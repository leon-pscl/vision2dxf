"""Request/response schemas for the HTTP API."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class UploadRequest(BaseModel):
    filename: str
    content: str


class SettingsPayload(BaseModel):
    """Everything the operator configures. Mirrors drafting.ease.Settings exactly."""

    model_config = ConfigDict(extra="forbid")
    garment_type: str = "basic_torso"

    section_model: str = "superellipse"
    section_exponent: float | None = Field(default=None, ge=2.0, le=6.0)
    cross_section_resolution: int | None = Field(default=None, ge=16, le=120)
    longitudinal_segments: int | None = Field(default=None, ge=1, le=12)

    ease_preset: str = "fitted"
    bust_ease_mm: float | None = None
    waist_ease_mm: float | None = None
    hip_ease_mm: float | None = None
    armhole_ease_mm: float | None = None

    seam_preset: str = "basic_torso_v1"
    front_neck_drop_mm: float | None = Field(default=None, ge=0.0, le=400.0)
    back_neck_drop_mm: float | None = Field(default=None, ge=0.0, le=400.0)

    flattening_method: str = "lscm_arap"
    w_distortion: float | None = Field(default=None, ge=0.0)
    w_boundary: float | None = Field(default=None, ge=0.0)
    w_seam: float | None = Field(default=None, ge=0.0)
    w_grain: float | None = Field(default=None, ge=0.0)
    w_flip: float | None = Field(default=None, ge=0.0)

    seam_allowances: dict[str, float] = Field(default_factory=dict)
    cutting_join: str | None = None
    paper_size: str | None = None
    paper_orientation: str | None = None

    unit: str = "cm"
    confirm_units: bool = False


class SemanticsRequest(BaseModel):
    """A generation request up to, but not including, the geometry.

    `confirmations` selects one of the interpretations the schema registry offers for a
    field; it cannot invent one. `corrections` replaces a subject's numeric value only.
    """

    model_config = ConfigDict(extra="forbid")
    yamlText: str
    filename: str = "body_measurements.yaml"
    confirmations: dict[str, str] = Field(default_factory=dict)
    corrections: dict[str, float] = Field(default_factory=dict)


class GenerateRequest(SemanticsRequest):
    """A full generation request: the profile, its confirmations and the settings."""

    settings: SettingsPayload = Field(default_factory=SettingsPayload)
    dark: bool = False
    # The dense flattened mesh is large; the interface asks for it only when it needs it.
    include_mesh: bool = False
    # Intermediate surface meshes are opt-in because they are much larger than the pattern.
    include_surfaces: bool = False


ExportRequest = GenerateRequest


class PatternRequest(BaseModel):
    """A supplied GarmageSet document, for the schema validator."""

    pattern: dict[str, Any]
    dark: bool = False
