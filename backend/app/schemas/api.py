"""Request/response schemas for the HTTP API."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class UploadRequest(BaseModel):
    filename: str
    content: str


class SettingsPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    garment_type: str = "basic_bodice"
    ease_preset: str = "fitted"
    bust_ease_mm: float | None = None
    waist_ease_mm: float | None = None
    hip_ease_mm: float | None = None
    seam_allowances: dict[str, float] = Field(default_factory=dict)
    cutting_join: str | None = None
    paper_size: str | None = None
    paper_orientation: str | None = None
    shoulder_slope_deg: float | None = None
    armscye_depth_mm: float | None = None
    bust_apex_y_mm: float | None = None
    dart_intake_mm: float | None = None
    unit: str = "cm"


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
    settings: SettingsPayload = Field(default_factory=SettingsPayload)


ExportRequest = GenerateRequest


class RenderRequest(BaseModel):
    pattern: dict[str, Any]
    dark: bool = False
