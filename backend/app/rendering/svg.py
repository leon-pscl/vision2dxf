"""Interactive preview source: an SVG that mirrors exactly the exported geometry."""
from __future__ import annotations

from dataclasses import dataclass
from xml.sax.saxutils import escape

from ..config import drafting_config
from ..drafting.cutting import CuttingLine
from ..drafting.geometry import PatternDocument, PatternEdge, Point2D
from ..drafting.panels import world_point

VIEW_LAYERS = (
    "construction", "landmarks", "controls", "sewing", "cutting", "darts",
    "notches", "labels", "rulers",
)

STYLES = """
:root { color-scheme: light dark; }
.bg { fill: var(--bg, #f7f7f8); }
.paper { fill: var(--paper, #ffffff); stroke: var(--grid, #d4d4d8); stroke-width: 1; }
.grid { stroke: var(--grid, #e4e4e7); stroke-width: 0.4; fill: none; }
.cutting { fill: none; stroke: var(--cutting, #b91c1c); stroke-width: 1.6; stroke-dasharray: 14 5; }
.sewing { fill: none; stroke: var(--sewing, #111827); stroke-width: 2.2; }
.dart { fill: none; stroke: var(--sewing, #111827); stroke-width: 1.4; stroke-dasharray: 6 4; }
.dart-fill { fill: var(--dartfill, #fecaca); stroke: none; opacity: .55; }
.construction { fill: none; stroke: var(--construction, #2563eb); stroke-width: 0.8; stroke-dasharray: 9 4 2 4; }
.handle { fill: none; stroke: var(--handle, #7c3aed); stroke-width: 0.8; stroke-dasharray: 3 3; }
.handle-dot { fill: var(--handle, #7c3aed); stroke: none; }
.landmark { fill: var(--sewing, #111827); stroke: none; }
.landmark-key { fill: var(--construction, #2563eb); stroke: none; }
.notch { fill: none; stroke: var(--notch, #0f766e); stroke-width: 1.3; }
.label { fill: var(--label, #374151); stroke: none; font: 500 11px ui-sans-serif, system-ui, sans-serif; }
.ruler { fill: none; stroke: var(--ruler, #a16207); stroke-width: 1.2; }
.ruler-text { fill: var(--ruler, #a16207); stroke: none; font: 600 11px ui-monospace, monospace; }
.scalebar { stroke: var(--ruler, #a16207); stroke-width: 2; fill: none; }
.scaletext { fill: var(--ruler, #a16207); stroke: none; font: 600 11px ui-monospace, monospace; }
.title { fill: var(--label, #111827); stroke: none; font: 600 14px ui-sans-serif, system-ui, sans-serif; }
""".strip()


@dataclass
class RenderOptions:
    layers: set[str] | None = None
    margin_mm: float = 20.0
    dark: bool = False
    show_grid: bool = True

    def active(self, layer: str) -> bool:
        return not self.layers or layer in self.layers


def _path_d(edges: list[PatternEdge], panel, spacing: float, *, absolute_start: bool) -> str:
    parts = []
    for i, edge in enumerate(edges):
        pts = _edge_points(edge, panel, spacing)
        if not pts:
            continue
        if i == 0 or absolute_start:
            parts.append(f"M {pts[0].x:.4f} {pts[0].y:.4f}")
        else:
            parts.append(f"L {pts[0].x:.4f} {pts[0].y:.4f}")
        for p in pts[1:]:
            parts.append(f"L {p.x:.4f} {p.y:.4f}")
    return " ".join(parts)


def _edge_points(edge: PatternEdge, panel, spacing: float) -> list[Point2D]:
    poly = edge.polyline(spacing)
    return [world_point(panel, p) for p in poly]


def document_extent(doc: PatternDocument, options: RenderOptions) -> tuple[float, float, float, float]:
    spacing = drafting_config()["sample_spacing_mm"]
    xs, ys = [], []
    for panel in doc.panels:
        for edge in panel.edges:
            for p in _edge_points(edge, panel, spacing):
                xs.append(p.x)
                ys.append(p.y)
        for p in panel.construction:
            for q in p:
                w = world_point(panel, q)
                xs.append(w.x)
                ys.append(w.y)
    m = options.margin_mm
    return min(xs) - m, min(ys) - m, max(xs) + m, max(ys) + m


def render_svg(doc: PatternDocument, cutting: dict[str, CuttingLine], options: RenderOptions | None = None) -> str:
    """Every layer is emitted into its own <g data-layer>, so the UI can toggle
    visibility client-side without a round trip. One render, identical geometry."""
    options = options or RenderOptions()
    cfg = drafting_config()
    spacing = cfg["sample_spacing_mm"]
    x0, y0, x1, y1 = document_extent(doc, options)
    width, height = x1 - x0, y1 - y0

    layers: dict[str, list[str]] = {name: [] for name in VIEW_LAYERS}
    if options.show_grid:
        layers["construction"].append(_grid(x0, y0, x1, y1))

    for panel in doc.panels:
        for a, b in panel.construction:
            layers["construction"].append(_line(world_point(panel, a), world_point(panel, b), "construction"))

        line = cutting.get(panel.key)
        if line:
            world = [world_point(panel, p) for p in line.contour]
            layers["cutting"].append(f'<path class="cutting" d="{_closed(world)}"/>')
            for contour in line.dart_contours:
                layers["cutting"].append(f'<path class="cutting" d="{_closed([world_point(panel, p) for p in contour])}"/>')

        for dart in panel.darts:
            for leg in dart.contours:
                pts = [world_point(panel, p) for p in leg]
                layers["darts"].append(f'<polygon class="dart-fill" points="{_poly(pts)}"/>')
                layers["darts"].append(f'<path class="dart" d="{_closed(pts)}"/>')

        layers["sewing"].append(
            f'<path class="sewing" d="{_path_d(panel.edges, panel, spacing, absolute_start=False)}"/>'
        )

        for edge in panel.edges:
            if edge.curve is not None:
                c = edge.curve
                pts = [world_point(panel, p) for p in (c.p0, c.c1, c.c2, c.p3)]
                layers["controls"].append(f'<path class="handle" d="{_path(pts)}"/>')
                for p in (c.c1, c.c2):
                    wp = world_point(panel, p)
                    layers["controls"].append(
                        f'<circle class="handle-dot" cx="{wp.x:.3f}" cy="{wp.y:.3f}" r="2.4"/>'
                    )
            for notch in edge.notches:
                layers["notches"].append(_notch(edge, panel, notch, spacing))
            mid = world_point(panel, edge.point_at_param(0.5))
            layers["labels"].append(
                f'<text class="label" x="{mid.x + 4:.3f}" y="{mid.y - 4:.3f}">'
                f'{escape(edge.label)} ({edge.length_mm:.1f} mm)</text>'
            )

        for name, p in panel.landmarks.items():
            wp = world_point(panel, p)
            layers["landmarks"].append(f'<circle class="landmark" cx="{wp.x:.3f}" cy="{wp.y:.3f}" r="2.6"/>')
            layers["landmarks"].append(
                f'<text class="label" x="{wp.x + 5:.3f}" y="{wp.y - 5:.3f}">{escape(name)}</text>'
            )

    layers["rulers"].append(_scale_bar(x0 + 20, y1 - 20))

    theme = ('<style>:root{--bg:#0b0d10;--grid:#23262d;--sewing:#e5e7eb;--cutting:#f87171;'
             '--construction:#60a5fa;--handle:#c084fc;--label:#9ca3af;--notch:#2dd4bf;'
             '--ruler:#fbbf24;--dartfill:#7f1d1d;}</style>') if options.dark else ""

    groups = "".join(
        f'<g data-layer="{name}"{"" if options.active(name) else " hidden"}>'
        + "".join(items)
        + "</g>"
        for name, items in layers.items()
    )
    background = (f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{width:.2f}" height="{height:.2f}" '
                  f'fill="var(--bg,#f7f7f8)"/>')
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x0:.2f} {y0:.2f} {width:.2f} {height:.2f}" '
        f'preserveAspectRatio="xMidYMid meet">'
        f"<style>{STYLES}</style>{theme}{background}<g>{groups}</g></svg>"
    )


def _closed(points: list[Point2D]) -> str:
    return _path(points) + " Z"


def _path(points: list[Point2D]) -> str:
    if not points:
        return ""
    return f"M {points[0].x:.4f} {points[0].y:.4f} " + " ".join(f"L {p.x:.4f} {p.y:.4f}" for p in points[1:])


def _poly(points: list[Point2D]) -> str:
    return " ".join(f"{p.x:.4f},{p.y:.4f}" for p in points)


def _line(a: Point2D, b: Point2D, cls: str) -> str:
    return f'<line class="{cls}" x1="{a.x:.3f}" y1="{a.y:.3f}" x2="{b.x:.3f}" y2="{b.y:.3f}"/>'


def _notch(edge: PatternEdge, panel, notch, spacing: float) -> str:
    p = world_point(panel, edge.point_at_param(notch.at))
    q = edge.polyline(spacing)
    before = q[0] if q else edge.start
    direction = (before.x - edge.start.x, before.y - edge.start.y)
    length = max(1e-9, (direction[0] ** 2 + direction[1] ** 2) ** 0.5)
    ux, uy = direction[0] / length, direction[1] / length
    nx, ny = uy, -ux
    size = 4.0 * notch_count(notch.kind)
    a = Point2D(p.x - size * ux, p.y - size * uy)
    b = Point2D(p.x + size * nx, p.y + size * ny)
    c = Point2D(p.x - size * nx, p.y - size * ny)
    return f'<path class="notch" d="{_path([a, b, c, a])}"/>'


def notch_count(kind: str) -> int:
    return {"single": 1, "double": 2, "triple": 3}.get(kind, 1)


def _scale_bar(x: float, y: float, length_mm: float = 100.0) -> str:
    return (
        f'<path class="scalebar" d="M {x:.2f} {y:.2f} L {x + length_mm:.2f} {y:.2f} '
        f'M {x:.2f} {y - 5:.2f} L {x:.2f} {y + 5:.2f} M {x + length_mm:.2f} {y - 5:.2f} '
        f'L {x + length_mm:.2f} {y + 5:.2f}"/>'
        f'<text class="scaletext" x="{x + 6:.2f}" y="{y - 8:.2f}">{length_mm:.0f} mm</text>'
    )


def _grid(x0: float, y0: float, x1: float, y1: float, step: float = 100.0) -> str:
    parts = []
    x = (x0 // step) * step
    while x <= x1:
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{y0:.1f}" x2="{x:.1f}" y2="{y1:.1f}"/>')
        x += step
    y = (y0 // step) * step
    while y <= y1:
        parts.append(f'<line class="grid" x1="{x0:.1f}" y1="{y:.1f}" x2="{x1:.1f}" y2="{y:.1f}"/>')
        y += step
    return "".join(parts)
