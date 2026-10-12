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
    """One semantic boundary edge.

    Geometry is held in exactly one of two forms:

    * `points` - the sampled control polyline the GarmageSet schema stores, which is what
      a base block loaded from a template carries. Deforming a template means moving
      these points, so keeping them is lossless: an unchanged target reproduces the
      source edge exactly.
    * `curve` - a single cubic, for edges that were constructed analytically.

    With neither, the edge is the straight segment from `start` to `end`.
    """

    role: str
    start: Point2D
    end: Point2D
    curve: CubicBezier | None = None
    points: list[Point2D] = field(default_factory=list)
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
    def is_curved(self) -> bool:
        return self.curve is not None or len(self.points) > 2

    @property
    def length_mm(self) -> float:
        pts = self.polyline()
        return sum(a.dist(b) for a, b in zip(pts, pts[1:]))

    def polyline(self, spacing_mm: float = 2.0) -> list[Point2D]:
        if len(self.points) >= 2:
            return list(self.points)
        return self.curve.sample(spacing_mm) if self.curve else [self.start, self.end]

    def point_at_param(self, param: float) -> Point2D:
        """Normalised parameter along the edge, 0 = start, 1 = end.

        A control polyline is parameterised by arc length, so a stitch or a notch at 0.25
        sits a quarter of the way along the seam the way a sewer would walk it. A cubic
        keeps its own parameterisation.
        """
        if len(self.points) >= 2:
            return _point_at_arc_length(self.points, param)
        if self.curve is None:
            return self.start.lerp(self.end, param)
        return self.curve.point_at(param)


def _point_at_arc_length(points: list[Point2D], param: float) -> Point2D:
    steps = [a.dist(b) for a, b in zip(points, points[1:])]
    total = sum(steps)
    if total <= 0.0:
        return points[0]
    target = max(0.0, min(1.0, param)) * total
    walked = 0.0
    for (a, b), step in zip(zip(points, points[1:]), steps):
        if walked + step >= target:
            return a.lerp(b, (target - walked) / step if step > 0 else 0.0)
        walked += step
    return points[-1]


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
    # Grain direction, as the two endpoints of the grainline arrow. The transformation
    # must preserve its direction (topology contract), so it travels with the panel.
    grainline: tuple[Point2D, Point2D] | None = None

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

    def edge_index(self, edge_id: str) -> int | None:
        return next((i for i, e in enumerate(self.edges) if e.ensure_id() == edge_id), None)

    def seam_length_mm(self, record: "StitchRecord") -> float | None:
        """Length of boundary one stitch record covers, walking the shorter way round.

        A record that starts and ends on the same edge is just a slice of it. A record
        that spans a corner covers the tail of one edge, whole edges in between, and the
        head of the last one.
        """
        first = self.edge_index(record.start_edge_id)
        last = self.edge_index(record.end_edge_id)
        if first is None or last is None:
            return None
        if first == last:
            return self.edges[first].length_mm * abs(record.end_param - record.start_param)

        def walk(a: int, pa: float, b: int, pb: float) -> float:
            total = self.edges[a].length_mm * (1.0 - pa)
            index = (a + 1) % len(self.edges)
            while index != b:
                total += self.edges[index].length_mm
                index = (index + 1) % len(self.edges)
            return total + self.edges[b].length_mm * pb

        return min(
            walk(first, record.start_param, last, record.end_param),
            walk(last, record.end_param, first, record.start_param),
        )


@dataclass
class StitchRecord:
    """One side of a seam: from a point on an edge to a point on an edge.

    The two ends may sit on different edges, so a seam can run across a boundary corner.
    Collapsing them into a single edge loses that, which is why both are kept.
    """

    panel_id: str
    start_edge_id: str
    start_param: float
    end_edge_id: str
    end_param: float
    is_counter_clockwise: bool = False

    @property
    def edge_ids(self) -> list[str]:
        return list(dict.fromkeys([self.start_edge_id, self.end_edge_id]))

    @property
    def spans_one_edge(self) -> bool:
        return self.start_edge_id == self.end_edge_id


@dataclass
class StitchPair:
    records: list[StitchRecord]

    def edge_ids(self) -> list[str]:
        return list(dict.fromkeys(i for r in self.records for i in r.edge_ids))


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


def world_point(panel: PatternPanel, point: Point2D) -> Point2D:
    """A panel-local point in workspace coordinates.

    Panel geometry is stored local to the panel and placed by its `center`, which is what
    the GarmageSet schema does, so anything drawing or exporting several panels together
    has to add the placement back.
    """
    center = panel.center or Point2D(0.0, 0.0)
    return Point2D(point.x + center.x, point.y + center.y)
