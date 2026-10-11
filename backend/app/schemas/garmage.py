"""Typed models for the GarmageSet-like fixture schema (GarmageSet.json)."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EdgeRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    clothPieceId: str
    edgeId: str
    param: float = Field(ge=0.0, le=1.0)


class StitchRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start: EdgeRef
    end: EdgeRef
    isCounterClockWise: bool


class StitchPair(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: list[StitchRecord] = Field(min_length=2, max_length=2)

    @model_validator(mode="before")
    @classmethod
    def accept_array_form(cls, value):
        """The fixture stores a seam as a bare 2-element array of records."""
        if isinstance(value, list):
            return {"records": value}
        return value

    def __iter__(self):  # keeps dict(StitchPair(...)) shape-compatible
        return iter(self.records)


class PanelEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    label: str
    controlPoints: list[list[float]]
    bezierPoints: list[list[float]]


class SeqEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: int
    circleType: int
    vertices: list[list[float]] = Field(min_length=3)
    edges: list[PanelEdge] = Field(min_length=3)


class Panel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    label: str
    center: list[float]
    collisionLayer: int
    particleDistance: float
    seqEdges: list[SeqEdge]


class GarmageSet(BaseModel):
    model_config = ConfigDict(extra="forbid")
    panels: list[Panel]
    stitches: list[StitchPair]
