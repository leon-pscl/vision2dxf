"""Full-scale DXF export. 1 drawing unit = 1 mm, one layer per geometry class."""
from __future__ import annotations

import io
import ezdxf

from ..config import drafting_config
from ..drafting.cutting import CuttingLine
from ..drafting.geometry import PatternDocument, Point2D, world_point


LAYERS = (
    ("SEWING", 7),      # white
    ("CUTTING", 1),     # red
    ("CONSTRUCTION", 5),
    ("DARTS", 6),       # magenta
    ("NOTCHES", 4),
    ("LABELS", 3),
)


def export_dxf(doc: PatternDocument, cutting: dict[str, CuttingLine]) -> bytes:
    spacing = drafting_config()["sample_spacing_mm"]
    dxf = ezdxf.new("R2010", setup=True)
    dxf.header["$INSUNITS"] = 4  # millimetres
    dxf.header["$MEASUREMENT"] = 1  # metric
    for name, color in LAYERS:
        if name not in dxf.layers:
            dxf.layers.add(name, color=color)
    msp = dxf.modelspace()

    for panel in doc.panels:
        for edge in panel.edges:
            world = [world_point(panel, p) for p in edge.polyline(spacing)]
            msp.add_lwpolyline([(p.x, p.y) for p in world], dxfattribs={"layer": "SEWING"})
        line = cutting.get(panel.key)
        if line:
            for contour in [line.contour] + line.dart_contours:
                world = [world_point(panel, p) for p in contour]
                pts = [(p.x, p.y) for p in world]
                msp.add_lwpolyline(pts + [pts[0]], dxfattribs={"layer": "CUTTING"})
        for a, b in panel.construction:
            p, q = world_point(panel, a), world_point(panel, b)
            msp.add_line((p.x, p.y), (q.x, q.y), dxfattribs={"layer": "CONSTRUCTION"})
        for dart in panel.darts:
            for leg in dart.contours:
                world = [world_point(panel, p) for p in leg]
                pts = [(p.x, p.y) for p in world]
                msp.add_lwpolyline(pts + [pts[0]], dxfattribs={"layer": "DARTS"})
        for edge in panel.edges:
            for notch in edge.notches:
                p = world_point(panel, edge.point_at_param(notch.at))
                s = 4.0
                msp.add_line((p.x - s, p.y), (p.x + s, p.y), dxfattribs={"layer": "NOTCHES"})
            p = world_point(panel, edge.point_at_param(0.5))
            msp.add_text(f"{edge.label} {edge.length_mm:.1f}", height=6,
                         dxfattribs={"layer": "LABELS"}).set_placement((p.x + 6, p.y + 6))
        p = world_point(panel, Point2D(0.0, 0.0))
        msp.add_text(panel.label, height=14, dxfattribs={"layer": "LABELS"}).set_placement((p.x, p.y))

    buf = io.StringIO()
    dxf.write(buf)
    return buf.getvalue().encode("utf-8")
