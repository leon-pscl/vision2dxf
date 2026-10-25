"""Full-scale PDF export. 1 drawing unit = 1 mm, one sheet per panel."""
from __future__ import annotations

import io

from reportlab.lib.colors import Color, black, white
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas as rl_canvas

from ..config import drafting_config
from ..drafting.cutting import CuttingLine
from ..drafting.geometry import PatternDocument, PatternPanel, world_point


DASH_CUTTING = (12, 5)
DASH_CONSTRUCTION = (8, 3, 1.5, 3)


def _set_line(c: rl_canvas.Canvas, width: float, color=black, dash=None) -> None:
    c.setLineWidth(width)
    c.setStrokeColor(color)
    c.setDash(dash or [], 0)


def _page_size(cfg: dict, size_name: str, orient: str) -> tuple[float, float]:
    w, h = cfg["sizes_mm"][size_name]
    return (max(w, h), min(w, h)) if orient == "landscape" else (min(w, h), max(w, h))


def export_pdf(
    doc: PatternDocument,
    cutting: dict[str, CuttingLine],
    paper_size: str | None = None,
    orientation: str | None = None,
) -> bytes:
    cfg = drafting_config()["paper"]
    size_name = paper_size or cfg["default_size"]
    orient = orientation or cfg["default_orientation"]

    sheets = _place(doc, cfg, size_name, orient)

    buf = io.BytesIO()
    base_w, base_h = _page_size(cfg, size_name, orient)
    c = rl_canvas.Canvas(buf, pagesize=(base_w * mm, base_h * mm))
    c.setTitle("Pattern - full scale")
    c.setAuthor("formula-drafting")

    for index, (panel, sx, sy, page_w, page_h) in enumerate(sheets):
        c.setPageSize((page_w * mm, page_h * mm))
        _draw_sheet(c, index, panel, sx, sy, cutting, size_name, page_w, page_h, cfg)
        c.showPage()
    c.save()
    return buf.getvalue()


def _place(doc: PatternDocument, cfg: dict, size_name: str, orient: str):
    """One sheet per panel, positioned in world coordinates.

    Offsets come from the panel's *world* bbox — local geometry plus the
    `center` it is placed by — because that is the frame every renderer draws
    in; a local bbox here is what used to throw panels about a metre off the
    sheet. The bbox is grown by the seam allowance so the cut line lands on the
    sheet as well, and the sheet turns to the other orientation when the
    configured one cannot hold the panel.
    """
    placed = []
    for panel in doc.panels:
        x0, y0, x1, y1 = panel.bbox()
        cx, cy = (panel.center.x, panel.center.y) if panel.center else (0.0, 0.0)
        wx0, wy0, wx1, wy1 = x0 + cx, y0 + cy, x1 + cx, y1 + cy
        raw_w, raw_h = wx1 - wx0, wy1 - wy0
        allowance = max((e.seam_allowance_mm for e in panel.edges), default=0.0)
        w, h = raw_w + 2 * allowance, raw_h + 2 * allowance

        page_w, page_h = _page_size(cfg, size_name, orient)
        turned = _page_size(cfg, size_name, "portrait" if orient == "landscape" else "landscape")
        if not _on_sheet(w, h, page_w, page_h):
            # The configured orientation cannot hold the cut line: turn when that
            # holds it, or when it at least holds the sewing line and the
            # configured orientation holds neither.
            if _on_sheet(w, h, *turned) or (
                _on_sheet(raw_w, raw_h, *turned) and not _on_sheet(raw_w, raw_h, page_w, page_h)
            ):
                page_w, page_h = turned

        sx = (page_w - w) / 2 - (wx0 - allowance)
        sy = (page_h - h) / 2 - (wy0 - allowance)
        placed.append((panel, sx, sy, page_w, page_h))
    return placed


def _on_sheet(w: float, h: float, page_w: float, page_h: float) -> bool:
    return w <= page_w and h <= page_h


def _draw_sheet(c, index, panel: PatternPanel, sx, sy, cutting, size_name, page_w, page_h, cfg) -> None:
    spacing = drafting_config()["sample_spacing_mm"]
    margin = cfg["margin_mm"]

    def X(p): return (p.x + sx) * mm

    def Y(p): return page_h * mm - (p.y + sy) * mm

    orient = "landscape" if page_w >= page_h else "portrait"
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin * mm, page_h * mm - margin * mm - 14,
                 f"Pattern sheet {index + 1} - {size_name} {orient} - 1:1")

    # construction
    _set_line(c, 0.4, Color(0.2, 0.4, 0.9), DASH_CONSTRUCTION)
    for a, b in panel.construction:
        p, q = world_point(panel, a), world_point(panel, b)
        path = c.beginPath()
        path.moveTo(X(p), Y(p))
        path.lineTo(X(q), Y(q))
        c.drawPath(path)

    # cutting lines
    line = cutting.get(panel.key)
    if line:
        _set_line(c, 0.8, Color(0.75, 0.1, 0.1), DASH_CUTTING)
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
    for edge in panel.edges:
        p = world_point(panel, edge.point_at_param(0.5))
        c.drawString(X(p) + 6, Y(p) - 3, f"{edge.label} {edge.length_mm:.1f} mm")
