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
    # A catalog block id, or `template:<file>.json` to load any template in the library.
    base_block_id: str | None = None
    ease_preset: str = "fitted"
    bust_ease_mm: float | None = None
    waist_ease_mm: float | None = None
    hip_ease_mm: float | None = None
    seam_allowances: dict[str, float] = Field(default_factory=dict)
    cutting_join: str | None = None
    paper_size: str | None = None
    paper_orientation: str | None = None
    # Transformation-strength limits: the envelope is advisory only once confirmed, and
    # still refuses a target more than stop_factor past the tested range.
    confirm_envelope: bool = False
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


class TransformRequest(SemanticsRequest):
    """A transformation request: the target profile, its confirmations and the settings."""

    settings: SettingsPayload = Field(default_factory=SettingsPayload)
    dark: bool = False


ExportRequest = TransformRequest


class RenderRequest(BaseModel):
    pattern: dict[str, Any]
    dark: bool = False
