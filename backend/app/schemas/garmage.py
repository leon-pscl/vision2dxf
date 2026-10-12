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


# Contour kinds observed across the template library: 3 is the closed outer boundary of a
# panel, 4 an internal contour - a reference line, a guide, or a closed inner loop.
CLOSED_BOUNDARY_TYPE = 3
INTERNAL_TYPE = 4


class SeqEdge(BaseModel):
    """One contour of a panel.

    The closed outer boundary needs at least three vertices to enclose an area. An
    internal contour is often a single two-point line - a bust, waist or chest reference
    line - so the minimum there is two. Requiring three everywhere rejects 28 of every 60
    files in the template library.
    """

    model_config = ConfigDict(extra="forbid")
    type: int
    circleType: int
    vertices: list[list[float]] = Field(min_length=2)
    edges: list[PanelEdge] = Field(min_length=1)

    @model_validator(mode="after")
    def closed_boundary_encloses_area(self):
        if self.type == CLOSED_BOUNDARY_TYPE:
            if len(self.vertices) < 3 or len(self.edges) < 3:
                raise ValueError(
                    "a closed boundary contour needs at least three vertices and edges"
                )
            if len(self.vertices) != len(self.edges):
                raise ValueError(
                    "a closed boundary contour needs one edge per vertex"
                )
        return self

    @property
    def is_closed_boundary(self) -> bool:
        return self.type == CLOSED_BOUNDARY_TYPE


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
