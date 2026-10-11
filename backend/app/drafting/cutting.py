"""Cutting lines.

Panel boundaries are sewing lines and stay untouched. Cutting lines are generated
afterwards by offsetting the flattened boundary outward, one distance per edge so
fold edges (allowance 0) stay on the fold line.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..config import drafting_config
from .curves import offset_polylines
from .ease import Settings, resolve_join
from .geometry import PatternDocument, PatternPanel, Point2D


@dataclass
class CuttingLine:
    panel_key: str
    contour: list[Point2D]
    dart_contours: list[list[Point2D]] = field(default_factory=list)
    join: str = "miter"
    closed: bool = True


def flatten_boundary(panel: PatternPanel, spacing_mm: float | None = None) -> list[list[Point2D]]:
    """Sewing-line boundary as one closed polyline per edge, in edge order."""
    spacing = spacing_mm or drafting_config()["sample_spacing_mm"]
    return [edge.polyline(spacing) for edge in panel.edges]


def build_cutting_lines(doc: PatternDocument, settings: Settings) -> dict[str, CuttingLine]:
    spacing = drafting_config()["sample_spacing_mm"]
    join = resolve_join(settings)
    dart_allowance = settings.seam_allowances.get(
        "dart", drafting_config()["seam_allowance_mm"]["dart"]
    )
    out: dict[str, CuttingLine] = {}
    for panel in doc.panels:
        polylines = flatten_boundary(panel, spacing)
        distances = [edge.seam_allowance_mm for edge in panel.edges]
        contour = offset_polylines(polylines, distances, join=join)
        dart_contours = [
            offset_polylines(
                [[leg[0], leg[1]], [leg[1], leg[2]], [leg[2], leg[0]]],
                [dart_allowance] * 3,
                join=join,
            )
            for dart in panel.darts
            for leg in dart.contours
        ]
        out[panel.key] = CuttingLine(
            panel_key=panel.key, contour=contour, dart_contours=dart_contours, join=join
        )
    return out
