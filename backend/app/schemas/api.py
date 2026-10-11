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


class GenerateRequest(BaseModel):
    yamlText: str
    filename: str = "body_measurements.yaml"
    settings: SettingsPayload = Field(default_factory=SettingsPayload)


class RenderRequest(BaseModel):
    pattern: dict[str, Any]
    layers: list[str] | None = None
    dark: bool = False
    showGrid: bool = True


class ExportRequest(BaseModel):
    yamlText: str
    filename: str = "body_measurements.yaml"
    settings: SettingsPayload = Field(default_factory=SettingsPayload)


class MeasurementReviewResponse(BaseModel):
    ok: bool
    filename: str
    unit: str
    rows: list[dict]
    warnings: list[dict]
    dictionary: dict
