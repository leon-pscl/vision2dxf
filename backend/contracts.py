"""Frozen contract layer.

Every adapter, mock or real, returns one of these models. Do not edit a field
to accommodate a model: if a model cannot satisfy a contract, report the
mismatch instead.

Conventions:
  * all lengths are centimetres, all angles degrees, all areas implicit
  * images travel as base64 strings (data URL or bare base64)
  * every result carries ``is_mock`` and ``warnings``
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# --------------------------------------------------------------------------
# shared
# --------------------------------------------------------------------------


class Result(BaseModel):
    """Base for every step result."""

    is_mock: bool = True
    warnings: list[str] = Field(default_factory=list)


class GarmentType(str, Enum):
    POLO_SHIRT = "polo_shirt"
    LONG_SLEEVE_POLO = "long_sleeve_polo"
    SHORT_SLEEVE_POLO = "short_sleeve_polo"
    SLACKS = "slacks"


class CalibMethod(str, Enum):
    HEIGHT = "height"
    REFERENCE_OBJECT = "reference_object"


class LandmarksSource(str, Enum):
    MODEL = "model"
    HEURISTIC = "heuristic"
    USER = "user"


class DraftRoute(str, Enum):
    BLOCK = "block"
    GARMENTCODE = "garmentcode"


class MeasurementBackend(str, Enum):
    REGRESSION = "regression"
    BODY_MODEL = "body_model"


# --------------------------------------------------------------------------
# step 1 - garment type
# --------------------------------------------------------------------------


class GarmentSpec(Result):
    """What the chosen garment needs in order to be drafted."""

    garment_type: GarmentType
    required_measurements: list[str]
    ease_cm: dict[str, float]
    design_rules: dict = Field(default_factory=dict)


# --------------------------------------------------------------------------
# step 2 - capture
# --------------------------------------------------------------------------


class CaptureInput(BaseModel):
    front_image: str  # base64
    side_image: str  # base64
    height_cm: float = Field(gt=0)
    weight_kg: float | None = Field(default=None, gt=0)


class CaptureResult(Result):
    distance_ok: bool
    pose_ok: bool
    full_body_visible: bool
    clothing_ok: bool
    guidance: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# step 3 - segmentation
# --------------------------------------------------------------------------

Point2D = tuple[float, float]
Box2D = tuple[float, float, float, float]  # x1, y1, x2, y2


class SegmentationInput(BaseModel):
    image: str  # base64
    points: list[Point2D] = Field(default_factory=list)
    box: Box2D | None = None
    text: str | None = None


class SegmentationResult(Result):
    mask_png: str  # base64 PNG
    boundary_quality: float = Field(ge=0.0, le=1.0)
    model_name: str


# --------------------------------------------------------------------------
# step 4 - calibration
# --------------------------------------------------------------------------


class CalibrationResult(Result):
    method: CalibMethod
    # keyed by view name, e.g. {"front": 27.4, "side": 26.9}
    px_per_cm: dict[str, float]
    relative_uncertainty: float = Field(ge=0.0)


# --------------------------------------------------------------------------
# step 5 - landmarks
# --------------------------------------------------------------------------


class LandmarkPoint(BaseModel):
    x: float
    y: float
    confidence: float = Field(ge=0.0, le=1.0)
    source: LandmarksSource

    @field_validator("source", mode="before")
    @classmethod
    def _coerce_source(cls, v: object) -> object:
        # the wire format is a plain string; accept it as the enum
        if isinstance(v, str):
            return LandmarksSource(v)
        return v


class LandmarkResult(Result):
    points: dict[str, LandmarkPoint]
    confirmed_by_user: bool = False


class LandmarkUpdate(BaseModel):
    """User drag on the step 5 canvas."""

    name: str
    x: float
    y: float


# --------------------------------------------------------------------------
# step 6 - measurement (6a regression, 6b body model, identical result type)
# --------------------------------------------------------------------------


class Measurement(BaseModel):
    value_cm: float
    # None when the backend is uncalibrated and no interval exists
    lower_cm: float | None = None
    upper_cm: float | None = None
    method: str


class MeasurementResult(Result):
    backend: MeasurementBackend | None = None
    measurements: dict[str, Measurement] = Field(default_factory=dict)


# --------------------------------------------------------------------------
# step 7 - validation and YAML
# --------------------------------------------------------------------------

Severity = Literal["info", "warning", "error"]


class ValidationFlag(BaseModel):
    measurement: str
    rule: str
    severity: Severity
    detail: str = ""


class ValidationResult(Result):
    flags: list[ValidationFlag] = Field(default_factory=list)
    yaml: str


# --------------------------------------------------------------------------
# steps 8 and 9 - drafting and production
# --------------------------------------------------------------------------


class PatternPiece(BaseModel):
    name: str
    svg: str
    # step 9 only, optional so step 8 stays a plain block/garmentcode draft
    seam_allowance_cm: float | None = None
    notches: list[dict] = Field(default_factory=list)


class DraftResult(Result):
    route: DraftRoute
    pieces: list[PatternPiece] = Field(default_factory=list)


class ProductionResult(Result):
    pieces: list[PatternPiece] = Field(default_factory=list)
    export_formats: list[str] = Field(default_factory=lambda: ["svg"])


# --------------------------------------------------------------------------
# step 10 - drape and toile
# --------------------------------------------------------------------------


class DrapeResult(Result):
    placeholder_image: str  # base64 PNG
    toile_checklist: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# session
# --------------------------------------------------------------------------


class StepStatus(BaseModel):
    status: Literal["pending", "done", "warning"] = "pending"
    is_mock: bool = False


class SessionState(BaseModel):
    session_id: str
    created_at: str
    # step number -> serialised result. Step 6 is stored under its backend too.
    results: dict[str, dict] = Field(default_factory=dict)
    status: dict[str, StepStatus] = Field(default_factory=dict)
    # any result in the session is a mock. Drives the UI banner.
    any_mock: bool = False
    # step 6 is gated on landmark confirmation
    landmarks_confirmed: bool = False


class StepRequest(BaseModel):
    """Body of POST /api/session/{id}/step/{n}."""

    backend: MeasurementBackend | None = None
    payload: dict = Field(default_factory=dict)


class ModelsResponse(BaseModel):
    adapters: dict[str, str]
    device: str
