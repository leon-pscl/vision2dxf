"""Curve and contour utilities: sampling, bezier flattening, outward offsetting."""
from __future__ import annotations

from math import atan2, cos, hypot, sin

from .geometry import Point2D

JOIN_STYLES = ("miter", "bevel", "round")


def signed_area(points: list[Point2D]) -> float:
    total = 0.0
    for a, b in zip(points, points[1:] + points[:1]):
        total += a.x * b.y - b.x * a.y
    return total / 2.0


def centroid(points: list[Point2D]) -> Point2D:
    n = len(points) or 1
    return Point2D(sum(p.x for p in points) / n, sum(p.y for p in points) / n)


def arc_points(a: Point2D, b: Point2D, center: Point2D, steps: int = 8) -> list[Point2D]:
    """Arc from a to b around center; degenerate cases collapse to a straight join."""
    if center.dist(a) < 1e-9 or abs(center.dist(a) - center.dist(b)) > 1e-6:
        return [b]
    a0, a1 = atan2(a.y - center.y, a.x - center.x), atan2(b.y - center.y, b.x - center.x)
    while a1 < a0:
        a1 += 2 * 3.141592653589793
    r = center.dist(a)
    return [
        Point2D(center.x + r * cos(a0 + (a1 - a0) * i / steps),
                center.y + r * sin(a0 + (a1 - a0) * i / steps))
        for i in range(1, steps + 1)
    ]


def _edge_normals(pts: list[Point2D], mid: Point2D) -> list[tuple[float, float]]:
    normals = []
    for i in range(len(pts)):
        a = pts[max(i - 1, 0)]
        b = pts[min(i + 1, len(pts) - 1)]
        dx, dy = b.x - a.x, b.y - a.y
        length = hypot(dx, dy)
        if length < 1e-12:
            normals.append((0.0, 0.0))
            continue
        nx, ny = dy / length, -dx / length
        if (pts[i].x - mid.x) * nx + (pts[i].y - mid.y) * ny < 0:
            nx, ny = -nx, -ny
        normals.append((nx, ny))
    return normals


def _convex_at(polylines: list[list[Point2D]], orientation: float, i: int) -> bool:
    prev = (i - 1) % len(polylines)
    vertex = polylines[i][0]
    p_in = polylines[prev][0]
    p_out = polylines[i][1] if len(polylines[i]) > 1 else polylines[i][-1]
    v_in = (vertex.x - p_in.x, vertex.y - p_in.y)
    v_out = (p_out.x - vertex.x, p_out.y - vertex.y)
    cross = v_in[0] * v_out[1] - v_in[1] * v_out[0]
    if orientation >= 0.0:
        return cross > 1e-12
    return cross < -1e-12


def _ring_is_simple(points: list[Point2D]) -> bool:
    if len(points) < 4:
        return True
    segs = [(points[i], points[(i + 1) % len(points)]) for i in range(len(points))]

    def orient(a: Point2D, b: Point2D, c: Point2D) -> float:
        return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)

    def on_segment(a: Point2D, b: Point2D, c: Point2D) -> bool:
        return (min(a.x, b.x) - 1e-9 <= c.x <= max(a.x, b.x) + 1e-9 and
                min(a.y, b.y) - 1e-9 <= c.y <= max(a.y, b.y) + 1e-9)

    for i in range(len(segs)):
        a, b = segs[i]
        for j in range(i + 2, len(segs)):
            if j == len(segs) - 1 and i == 0:
                continue
            c, d = segs[j]
            d1 = orient(c, d, a)
            d2 = orient(c, d, b)
            d3 = orient(a, b, c)
            d4 = orient(a, b, d)
            if ((d1 > 1e-9) != (d2 > 1e-9)) and ((d3 > 1e-9) != (d4 > 1e-9)):
                return False
            if abs(d1) <= 1e-9 and on_segment(c, d, a):
                return False
            if abs(d2) <= 1e-9 and on_segment(c, d, b):
                return False
            if abs(d3) <= 1e-9 and on_segment(a, b, c):
                return False
            if abs(d4) <= 1e-9 and on_segment(a, b, d):
                return False
    return True


def offset_polylines(
    polylines: list[list[Point2D]],
    distances: list[float],
    join: str = "miter",
    miter_limit: float = 4.0,
    arc_steps: int = 8,
) -> list[Point2D]:
    if join not in JOIN_STYLES:
        raise ValueError(f"unknown join style: {join}")
    if len(polylines) != len(distances) or len(polylines) < 3:
        raise ValueError("offset_polylines needs at least three edges and one distance each")
    if any(len(p) < 2 for p in polylines):
        raise ValueError("every edge polyline needs at least two points")

    mid = centroid([p for line in polylines for p in line])
    normals = [_edge_normals(line, mid) for line in polylines]
    offset = [
        [Point2D(p.x + d * n[0], p.y + d * n[1]) for p, n in zip(line, ns)]
        for line, ns, d in zip(polylines, normals, distances)
    ]

    corners = [line[0] for line in polylines]
    orientation = signed_area(corners)

    count = len(polylines)
    out: list[Point2D] = []
    for i in range(count):
        prev = (i - 1) % count
        a = offset[prev][-1]
        b = offset[i][0]
        vertex = polylines[i][0]
        r_in, r_out = distances[prev], distances[i]
        n_in, n_out = normals[prev][-1], normals[i][0]

        is_concave = not _convex_at(polylines, orientation, i)
        if a.dist(b) < 1e-12:
            out.append(b)
        elif join == "round" and abs(r_in - r_out) < 1e-9 and r_out > 1e-9 and not is_concave:
            out.extend(arc_points(a, b, vertex, arc_steps))
        elif (join == "bevel" or is_concave or abs(r_in - r_out) > 1e-9 or
              r_out <= 1e-9 or r_in <= 1e-9):
            out.extend([a, b])
        else:
            bx, by = n_in[0] + n_out[0], n_in[1] + n_out[1]
            blen = hypot(bx, by)
            if blen < 1e-12:
                out.extend([a, b])
            else:
                bx, by = bx / blen, by / blen
                scale = 1.0 / max(0.2, bx * n_out[0] + by * n_out[1])
                if scale <= miter_limit:
                    out.append(Point2D(vertex.x + r_out * bx * scale, vertex.y + r_out * by * scale))
                else:
                    out.extend([a, b])
        out.extend(offset[i][1:-1])

    cleaned: list[Point2D] = [out[0]]
    for p in out[1:]:
        if p.dist(cleaned[-1]) > 1e-9:
            cleaned.append(p)
    if len(cleaned) < 3:
        raise ValueError("offset contour self-intersects and could not be repaired")
    if not _ring_is_simple(cleaned):
        from shapely.geometry import Polygon
        original_area = signed_area(cleaned)
        repaired_poly = Polygon([(p.x, p.y) for p in cleaned]).buffer(0)
        if repaired_poly.is_empty or repaired_poly.geom_type not in ("Polygon",):
            raise ValueError("offset contour self-intersects and could not be repaired")
        repaired_points = [Point2D(x, y) for x, y in repaired_poly.exterior.coords[:-1]]
        if original_area * signed_area(repaired_points) < 0:
            repaired_points.reverse()
        cleaned = repaired_points
        if len(cleaned) < 3:
            raise ValueError("offset contour self-intersects and could not be repaired")
    return cleaned


def polyline_length(points: list[Point2D]) -> float:
    return sum(a.dist(b) for a, b in zip(points, points[1:]))