"""Raster preview. Same geometry as the SVG and the production exports, drawn with
Pillow so no external SVG rasteriser is needed in the image."""
from __future__ import annotations

import io

from PIL import Image, ImageDraw

from ..config import drafting_config
from ..drafting.cutting import CuttingLine
from ..drafting.geometry import PatternDocument, world_point

from .svg import RenderOptions, document_extent

COLORS = {
    "bg": (247, 247, 248), "sewing": (17, 24, 39), "cutting": (185, 28, 28),
    "construction": (37, 99, 235), "handle": (124, 58, 237), "label": (55, 65, 81),
    "notch": (15, 118, 110), "dart": (254, 202, 202), "title": (17, 24, 39),
    "base": (156, 163, 175), "displacement": (219, 39, 119),
}
DARK_COLORS = {
    "bg": (11, 13, 16), "sewing": (229, 231, 235), "cutting": (248, 113, 113),
    "construction": (96, 165, 250), "handle": (192, 132, 252), "label": (156, 163, 175),
    "notch": (45, 212, 191), "dart": (127, 29, 29), "title": (229, 231, 235),
    "base": (107, 114, 128), "displacement": (244, 114, 182),
}


def render_png(
    doc: PatternDocument,
    cutting: dict[str, CuttingLine],
    options: RenderOptions | None = None,
    width_px: int = 1600,
    base: PatternDocument | None = None,
) -> bytes:
    """Raster preview. With `base` supplied this is the base-versus-target comparison:
    the base block in a muted outline, the transformed block in the primary style, and a
    displacement vector per landmark."""
    options = options or RenderOptions()
    cfg = drafting_config()
    spacing = cfg["sample_spacing_mm"]
    colors = DARK_COLORS if options.dark else COLORS
    x0, y0, x1, y1 = document_extent(doc, options)
    if base is not None:
        bx0, by0, bx1, by1 = document_extent(base, options)
        x0, y0, x1, y1 = min(x0, bx0), min(y0, by0), max(x1, bx1), max(y1, by1)
    scale = width_px / max(1e-9, x1 - x0)
    height_px = int((y1 - y0) * scale)
    img = Image.new("RGB", (width_px, height_px), colors["bg"])
    draw = ImageDraw.Draw(img)

    def to_px(p):
        return ((p.x - x0) * scale, (p.y - y0) * scale)

    def pts(points):
        return [to_px(p) for p in points]

    if base is not None:
        for panel in base.panels:
            for edge in panel.edges:
                world = [world_point(panel, p) for p in edge.polyline(spacing)]
                draw.line(pts(world), fill=colors["base"], width=2)
            for dart in panel.darts:
                for leg in dart.contours:
                    world = [world_point(panel, p) for p in leg]
                    draw.line(pts(world) + [to_px(world[0])], fill=colors["base"], width=1)

    if options.active("construction"):
        for panel in doc.panels:
            for a, b in panel.construction:
                draw.line(pts([world_point(panel, a), world_point(panel, b)]),
                          fill=colors["construction"], width=1)

    if options.active("cutting"):
        for panel in doc.panels:
            line = cutting.get(panel.key)
            if line:
                world = [_shift_to_world(panel, p) for p in line.contour]
                draw.line(pts(world) + [to_px(world[0])], fill=colors["cutting"], width=2)

    for panel in doc.panels:
        if options.active("darts"):
            for dart in panel.darts:
                for leg in dart.contours:
                    draw.polygon(pts([world_point(panel, p) for p in leg]), fill=colors["dart"])
        for edge in panel.edges:
            world = [world_point(panel, p) for p in edge.polyline(spacing)]
            draw.line(pts(world), fill=colors["sewing"], width=2)
        if options.active("labels"):
            for edge in panel.edges:
                mid = world_point(panel, edge.point_at_param(0.5))
                px, py = to_px(mid)
                draw.text((px + 4, py - 4), f"{edge.label} {edge.length_mm:.0f}", fill=colors["label"])
        if options.active("landmarks"):
            for name, p in panel.landmarks.items():
                px, py = to_px(world_point(panel, p))
                draw.ellipse([px - 3, py - 3, px + 3, py + 3], fill=colors["sewing"])
                draw.text((px + 6, py - 6), name, fill=colors["label"])

    if base is not None:
        base_panels = {p.key: p for p in base.panels}
        for panel in doc.panels:
            reference = base_panels.get(panel.key)
            if reference is None:
                continue
            for role, point in reference.landmarks.items():
                moved = panel.landmarks.get(role)
                if moved is None:
                    continue
                a = world_point(reference, point)
                b = world_point(panel, moved)
                if a.dist(b) < 0.5:
                    continue
                draw.line([to_px(a), to_px(b)], fill=colors["displacement"], width=2)
                bx, by = to_px(b)
                draw.ellipse([bx - 3, by - 3, bx + 3, by + 3], fill=colors["displacement"])
                draw.text((bx + 5, by + 2), f"{role} {a.dist(b):.1f}",
                          fill=colors["displacement"])

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _shift_to_world(panel, p):
    from ..drafting.geometry import Point2D
    return Point2D(p.x + panel.center.x, p.y + panel.center.y)
