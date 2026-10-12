"""Serializer from the internal geometry model to the verified GarmageSet schema.

Fixture-compatible defaults that the sample file cannot explain are read from
config/drafting.yaml (fixture_defaults) and documented in README.md:
  particleDistance = 6, collisionLayer = 0, seqEdge.type = 3, circleType = 0,
  edge.bezierPoints = [[0,0,0],[0,0,0]] (constant in every sampled edge).
`controlPoints` holds the flattened polyline of the edge, which is what the fixture
stores: two endpoints for straight edges, near-uniform samples for curves.
"""
from __future__ import annotations

from ..config import drafting_config
from ..drafting.geometry import PatternDocument, PatternEdge, Point2D
from ..schemas.garmage import GarmageSet, Panel, PanelEdge, SeqEdge, EdgeRef, StitchPair, StitchRecord

CONTINUITY_EPS = 1e-6


def edge_control_points(edge: PatternEdge, spacing_mm: float) -> list[list[float]]:
    """The points the schema stores for one edge.

    An edge that already holds a control polyline - which is what a block loaded from a
    template carries - is written out as it stands, point for point. Writing only its
    two ends would throw the curve away. A cubic is sampled; a straight segment is its
    two endpoints.
    """
    if len(edge.points) >= 2:
        return [p.as_triple() for p in edge.points]
    if edge.curve is None:
        return [edge.start.as_triple(), edge.end.as_triple()]
    return [p.as_triple() for p in edge.curve.sample(spacing_mm)]


def _contour_edges(edges: list[PatternEdge], spacing: float, defaults: dict) -> SeqEdge:
    return SeqEdge(
        type=defaults["seq_edge_type"],
        circleType=defaults["seq_edge_circle_type"],
        vertices=[e.start.as_triple() for e in edges],
        edges=[
            PanelEdge(
                id=e.ensure_id(),
                label=e.label,
                controlPoints=edge_control_points(e, spacing),
                bezierPoints=[list(p) for p in defaults["edge_bezier_points"]],
            )
            for e in edges
        ],
    )


def serialize(doc: PatternDocument) -> GarmageSet:
    cfg = drafting_config()
    defaults = cfg["fixture_defaults"]
    spacing = cfg["sample_spacing_mm"]

    panels: list[Panel] = []
    for panel in doc.panels:
        contours = [_contour_edges(panel.edges, spacing, defaults)]
        for dart in panel.darts:
            leg_edges = []
            for i, (apex, a, b) in enumerate(dart.legs):
                for j, (s, e) in enumerate(((apex, a), (a, b), (b, apex))):
                    leg_edges.append(
                        PatternEdge(
                            role="dart", start=s, end=e,
                            label=f"{panel.label} dart {i + 1}.{j + 1}",
                            key=f"{dart.id}.{i}.{j}", rule_id=dart.rule_id,
                        )
                    )
            for e in leg_edges:
                e.ensure_id()
            contours.append(_contour_edges(leg_edges, spacing, defaults))
        panels.append(
            Panel(
                id=panel.ensure_id(),
                label=panel.label,
                center=(panel.center or Point2D(0, 0)).as_triple(),
                collisionLayer=defaults["panel_collision_layer"],
                particleDistance=defaults["panel_particle_distance"],
                seqEdges=contours,
            )
        )

    stitches = [
        StitchPair(
            records=[
                StitchRecord(
                    start=EdgeRef(clothPieceId=r.panel_id, edgeId=r.start_edge_id,
                                  param=round(r.start_param, 8)),
                    end=EdgeRef(clothPieceId=r.panel_id, edgeId=r.end_edge_id,
                                param=round(r.end_param, 8)),
                    isCounterClockWise=r.is_counter_clockwise,
                )
                for r in pair.records
            ]
        )
        for pair in doc.stitches
    ]
    return GarmageSet(panels=panels, stitches=stitches)


def to_fixture_dict(doc: PatternDocument) -> dict:
    """Shape the stitch pairs as arrays, matching the fixture file structure."""
    data = serialize(doc).model_dump()
    data["stitches"] = [[record.model_dump() for record in pair.records] for pair in serialize(doc).stitches]
    return data
