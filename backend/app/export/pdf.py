"""Full-scale PDF export. 1 drawing unit = 1 mm, panels tiled across paper sheets."""
from __future__ import annotations

import io

from reportlab.lib.colors import Color, black, white
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas as rl_canvas

from ..config import drafting_config
from ..drafting.cutting import CuttingLine
from ..drafting.geometry import PatternDocument
from ..drafting.panels import world_point

DASH_CUTTING = (12, 5)
DASH_CONSTRUCTION = (8, 3, 1.5, 3)


def _set_line(c: rl_canvas.Canvas, width: float, color=black, dash=None) -> None:
    c.setLineWidth(width)
    c.setStrokeColor(color)
    c.setDash(dash or [], 0)


def export_pdf(
    doc: PatternDocument,
    cutting: dict[str, CuttingLine],
    paper_size: str | None = None,
    orientation: str | None = None,
) -> bytes:
    cfg = drafting_config()["paper"]
    size_name = paper_size or cfg["default_size"]
    orient = orientation or cfg["default_orientation"]
    page_w, page_h = cfg["sizes_mm"][size_name]
    if orient == "landscape":
        page_w, page_h = max(page_w, page_h), min(page_w, page_h)
    else:
        page_w, page_h = min(page_w, page_h), max(page_w, page_h)
    margin = cfg["margin_mm"]

    buf = io.BytesIO()
    c = rl_canvas.Canvas(buf, pagesize=(page_w * mm, page_h * mm))
    c.setTitle("Pattern - full scale")
    c.setAuthor("formula-drafting")

    placed = _place(doc, page_w, page_h, margin)
    for index, (sx, sy, panel) in enumerate(placed):
        c.showPage()
        _draw_sheet(c, index, sx, sy, doc, cutting, size_name, page_w, page_h)
    c.showPage()
    c.save()
    return buf.getvalue()


def _place(doc: PatternDocument, page_w: float, page_h: float, margin: float):
    placed = []
    x, y = margin, margin
    row_top = page_h - margin
    for panel in doc.panels:
        x0, y0, x1, y1 = panel.bbox()
        w, h = x1 - x0, y1 - y0
        if x + w + margin > page_w:
            x = margin
            y = row_top - h - margin
            row_top = y
        placed.append((x - x0, page_h - y - h, panel))
        x += w + margin * 2
    return placed


def _draw_sheet(c, index, sx, sy, doc, cutting, size_name, page_w, page_h) -> None:
    spacing = drafting_config()["sample_spacing_mm"]
    def X(p): return (p.x + sx) * mm
    def Y(p): return page_h * mm - (p.y + sy) * mm

    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin_c(page_w) * 1, page_h * mm - margin_c(page_h) * 1 - 14,
                  f"Pattern sheet {index + 1} - {size_name} - 1:1")

    # construction
    _set_line(c, 0.4, Color(0.2, 0.4, 0.9), DASH_CONSTRUCTION)
    for panel in doc.panels:
        for a, b in panel.construction:
            p, q = world_point(panel, a), world_point(panel, b)
            path = c.beginPath()
            path.moveTo(X(p), Y(p))
            path.lineTo(X(q), Y(q))
            c.drawPath(path)

    # cutting lines
    _set_line(c, 0.8, Color(0.75, 0.1, 0.1), DASH_CUTTING)
    for panel in doc.panels:
        line = cutting.get(panel.key)
        if not line:
            continue
        for contour in [line.contour] + line.dart_contours:
            path = c.beginPath()
            world = [world_point(panel, p) for p in contour]
            path.moveTo(X(world[0]), Y(world[0]))
            for p in world[1:]:
                path.lineTo(X(p), Y(p))
            path.close()
            c.drawPath(path)

    # darts
    _set_line(c, 0.6, Color(0.7, 0.2, 0.2), [4, 3])
    for panel in doc.panels:
        for dart in panel.darts:
            for leg in dart.contours:
                world = [world_point(panel, p) for p in leg]
                path = c.beginPath()
                path.moveTo(X(world[0]), Y(world[0]))
                for p in world[1:]:
                    path.lineTo(X(p), Y(p))
                path.close()
                c.drawPath(path)

    # sewing lines
    _set_line(c, 1.2, black)
    for panel in doc.panels:
        for edge in panel.edges:
            world = [world_point(panel, p) for p in edge.polyline(spacing)]
            path = c.beginPath()
            path.moveTo(X(world[0]), Y(world[0]))
            for p in world[1:]:
                path.lineTo(X(p), Y(p))
            c.drawPath(path)

    c.setFillColor(black)
    c.setStrokeColor(white)
    c.setFont("Helvetica", 7)
    for panel in doc.panels:
        for edge in panel.edges:
            p = world_point(panel, edge.point_at_param(0.5))
            c.drawString(X(p) + 6, Y(p) - 3, f"{edge.label} {edge.length_mm:.1f} mm")


def margin_c(v: float) -> float:
    return drafting_config()["paper"]["margin_mm"] * mm
