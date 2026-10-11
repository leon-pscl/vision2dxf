"""Semantic geometry model, independent of any output schema."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from math import ceil, hypot

# Fixed namespace so UUIDs are reproducible: identical input + settings must give
# byte-identical output (design doc, "Deterministic generation").
NAMESPACE = uuid.UUID("6f1a1f36-6b1e-5f4a-9a3e-0c1d2b3a4d5e")


def stable_id(*parts: str) -> str:
    return str(uuid.uuid5(NAMESPACE, "|".join(parts)))


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float

    def dist(self, other: "Point2D") -> float:
        return hypot(self.x - other.x, self.y - other.y)

    def lerp(self, other: "Point2D", t: float) -> "Point2D":
        return Point2D(self.x + (other.x - self.x) * t, self.y + (other.y - self.y) * t)

    def as_triple(self) -> list[float]:
        return [self.x, self.y, 0.0]


@dataclass(frozen=True)
class CubicBezier:
    """Cubic segment in millimetres."""

    p0: Point2D
    c1: Point2D
    c2: Point2D
    p3: Point2D

    def point_at(self, t: float) -> Point2D:
        u = 1.0 - t
        a, b, c, d = u * u * u, 3 * u * u * t, 3 * u * t * t, t * t * t
        return Point2D(
            a * self.p0.x + b * self.c1.x + c * self.c2.x + d * self.p3.x,
            a * self.p0.y + b * self.c1.y + c * self.c2.y + d * self.p3.y,
        )

    @property
    def chord(self) -> float:
        return self.p0.dist(self.p3)

    def polygon_length(self) -> float:
        return self.p0.dist(self.c1) + self.c1.dist(self.c2) + self.c2.dist(self.p3)

    def sample(self, spacing_mm: float) -> list[Point2D]:
        """Deterministic near-uniform sampling, matching the fixture's polyline edges."""
        n = max(2, ceil(self.polygon_length() / spacing_mm))
        return [self.point_at(i / n) for i in range(n + 1)]

    def handle_length(self) -> float:
        return max(self.p0.dist(self.c1), self.c2.dist(self.p3))


@dataclass(frozen=True)
class Notch:
    """Cut mark on an edge; preserved through cutting-line generation."""

    at: float  # normalised position along the edge
    kind: str  # single | double | triple
    role: str


@dataclass
class PatternEdge:
    role: str
    start: Point2D
    end: Point2D
    curve: CubicBezier | None = None
    seam_allowance_mm: float = 0.0
    paired_edge_id: str | None = None
    notches: list[Notch] = field(default_factory=list)
    label: str = ""
    rule_id: str = ""
    key: str = ""
    id: str = ""

    def ensure_id(self) -> str:
        if not self.id:
            self.id = stable_id("edge", self.key or self.role)
        return self.id

    @property
    def length_mm(self) -> float:
        if self.curve is not None:
            return sum(a.dist(b) for a, b in zip(self.polyline(), self.polyline()[1:]))
        return self.start.dist(self.end)

    def polyline(self, spacing_mm: float = 2.0) -> list[Point2D]:
        return self.curve.sample(spacing_mm) if self.curve else [self.start, self.end]

    def point_at_param(self, param: float) -> Point2D:
        """Normalised parameter along the edge, 0 = start, 1 = end."""
        if self.curve is None:
            return self.start.lerp(self.end, param)
        return self.curve.point_at(param)


@dataclass
class Dart:
    id: str
    apex: Point2D
    legs: list[tuple[Point2D, Point2D, Point2D]]  # (apex, leg_a, leg_b) per level
    intake_mm: float
    rule_id: str = ""

    @property
    def contours(self) -> list[list[Point2D]]:
        return [[apex, a, b] for apex, a, b in self.legs]


@dataclass
class PatternPanel:
    key: str
    label: str
    edges: list[PatternEdge]
    darts: list[Dart] = field(default_factory=list)
    construction: list[Point2D] = field(default_factory=list)
    landmarks: dict[str, Point2D] = field(default_factory=dict)
    center: Point2D | None = None
    id: str = ""

    def ensure_id(self) -> str:
        if not self.id:
            self.id = stable_id("panel", self.key)
        return self.id

    def outer_contour(self) -> list[Point2D]:
        """Closed outer boundary: every edge start, in edge order."""
        return [e.start for e in self.edges]

    def edge_by_role(self, role: str) -> PatternEdge | None:
        return next((e for e in self.edges if e.role == role), None)

    def bbox(self) -> tuple[float, float, float, float]:
        pts = self.outer_contour() + [e.end for e in self.edges]
        xs = [p.x for p in pts]
        ys = [p.y for p in pts]
        return min(xs), min(ys), max(xs), max(ys)


@dataclass
class StitchRecord:
    panel_id: str
    edge_id: str
    param_start: float
    param_end: float
    is_counter_clockwise: bool = False

    @property
    def length_param(self) -> float:
        return abs(self.param_end - self.param_start)


@dataclass
class StitchPair:
    records: list[StitchRecord]

    def edge_ids(self) -> list[str]:
        return [r.edge_id for r in self.records]


@dataclass
class PatternDocument:
    panels: list[PatternPanel]
    stitches: list[StitchPair]

    def panel_by_id(self, panel_id: str) -> PatternPanel | None:
        return next((p for p in self.panels if p.ensure_id() == panel_id), None)

    def edge_by_id(self, edge_id: str) -> tuple[PatternPanel, PatternEdge] | None:
        for panel in self.panels:
            for edge in panel.edges:
                if edge.ensure_id() == edge_id:
                    return panel, edge
        return None
