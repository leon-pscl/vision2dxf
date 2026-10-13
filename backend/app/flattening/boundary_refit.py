"""Boundary reconstruction.

A flattened panel boundary has a few hundred vertices. Serialising all of them as pattern
corners would produce a technically valid file that no one can sew from, so each semantic
interval of the boundary becomes one pattern edge: a straight line where the interval is
straight, a single cubic where it curves.

The order is the one the design sets out: extract, split on the semantic intervals,
simplify, classify, fit, keep the semantic endpoints exact, measure, reject a fit that
misses. The dense boundary is kept beside the result for the diagnostics, never instead
of it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import reconstruction_config
from ..drafting.ease import allowance_for
from ..drafting.geometry import CubicBezier, PatternEdge, Point2D

ROLE_LABELS = {
    "center_front": "Centre front (fold)",
    "center_back": "Centre back (fold)",
    "side_seam": "Side seam",
    "neckline": "Neckline",
    "armscye": "Armhole",
    "shoulder": "Shoulder seam",
    "waist": "Waist / hem",
}


class RefitError(Exception):
    def __init__(self, errors: list[dict]) -> None:
        self.errors = errors
        super().__init__("; ".join(e["message"] for e in errors))


@dataclass
class EdgeFit:
    role: str
    kind: str                 # line | cubic
    dense_points: int
    simplified_points: int
    max_deviation_mm: float
    dense_length_mm: float
    fitted_length_mm: float

    def as_dict(self) -> dict:
        return {
            "role": self.role,
            "kind": self.kind,
            "densePoints": self.dense_points,
            "simplifiedPoints": self.simplified_points,
            "maxDeviationMm": round(self.max_deviation_mm, 5),
            "denseLengthMm": round(self.dense_length_mm, 4),
            "fittedLengthMm": round(self.fitted_length_mm, 4),
            "lengthErrorMm": round(abs(self.fitted_length_mm - self.dense_length_mm), 5),
        }


def refit_panel(panel, uv: np.ndarray, allowances: dict[str, float]) -> tuple[list[PatternEdge], list[EdgeFit]]:
    """One pattern edge per semantic boundary interval, in walk order."""
    cfg = reconstruction_config()["boundary_refit"]
    edges: list[PatternEdge] = []
    fits: list[EdgeFit] = []
    errors: list[dict] = []

    for role in panel.boundary_order:
        ids = panel.semantic_boundaries[role]
        dense = uv[ids]
        simplified = simplify(dense, float(cfg["simplify_tolerance_mm"]), int(cfg["min_points"]))
        edge, fit = _fit_interval(panel, role, dense, simplified, cfg, allowances)
        if fit.max_deviation_mm > float(cfg["max_deviation_mm"]):
            errors.append({
                "field": f"{panel.key}.{role}", "code": "BOUNDARY_FIT_ABOVE_TOLERANCE",
                "message": (f"The {role} edge of the {panel.key} panel cannot be "
                            f"represented within {cfg['max_deviation_mm']} mm; the best "
                            f"{fit.kind} misses by {fit.max_deviation_mm:.2f} mm."),
                "suggestedAction": "Raise the fitting tolerance, or the simplification "
                                   "tolerance that bounds the polyline fallback.",
                "severity": "blocking",
            })
        edges.append(edge)
        fits.append(fit)

    if errors:
        raise RefitError(errors)

    # The walk is closed, so every edge ends exactly where the next one starts. The
    # semantic corners are shared vertices, so this is exact rather than within tolerance.
    for a, b in zip(edges, edges[1:] + edges[:1]):
        if a.end.dist(b.start) > 1e-9:
            raise RefitError([{
                "field": panel.key, "code": "BOUNDARY_NOT_CLOSED",
                "message": (f"The {panel.key} boundary leaves a {a.end.dist(b.start):.4f} mm "
                            f"gap between {a.role} and {b.role}."),
                "suggestedAction": "This is a defect in the refit, not in the input.",
                "severity": "blocking",
            }])

    # The dense boundary was already checked for crossings; the fitted one has to be
    # checked again, because a curve fitted through its points can still leave the
    # corridor they ran in and cross the edge it meets.
    crossing = _crossing_roles(edges)
    if crossing:
        raise RefitError([{
            "field": f"{panel.key}.{crossing[0]}", "code": "REFIT_BOUNDARY_SELF_INTERSECTION",
            "message": (f"The fitted {crossing[0]} and {crossing[1]} edges of the "
                        f"{panel.key} panel cross each other."),
            "suggestedAction": "Lower the handle ratio or the fitting tolerance so those "
                               "intervals fall back to a polyline.",
            "severity": "blocking",
        }])
    return edges, fits


def _crossing_roles(edges: list[PatternEdge]) -> tuple[str, str] | None:
    """The first pair of boundary edges that meet anywhere they should not."""
    from shapely.geometry import LineString

    lines = [(e.role, LineString([(p.x, p.y) for p in e.polyline(2.0)])) for e in edges]
    count = len(lines)
    for i, (role_a, line_a) in enumerate(lines):
        if not line_a.is_simple:
            return role_a, role_a
        for j in range(i + 1, count):
            role_b, line_b = lines[j]
            shared = (j == i + 1) or (i == 0 and j == count - 1)
            meeting = line_a.intersection(line_b)
            if meeting.is_empty:
                continue
            # Neighbours are allowed to meet, at their shared corner and nowhere else.
            if not shared or meeting.geom_type != "Point":
                return role_a, role_b
    return None


def _fit_interval(panel, role, dense, simplified, cfg, allowances) -> tuple[PatternEdge, EdgeFit]:
    start = Point2D(float(dense[0, 0]), float(dense[0, 1]))
    end = Point2D(float(dense[-1, 0]), float(dense[-1, 1]))
    dense_length = _length(dense)
    straight = float(cfg["straight_tolerance_mm"])
    deviation = _chord_deviation(dense)

    edge = PatternEdge(
        role=role, start=start, end=end,
        label=f"{panel.key} {ROLE_LABELS.get(role, role)}",
        key=f"{panel.key}:{role}", rule_id=f"seam:{role}",
        seam_allowance_mm=allowance_for(role, allowances),
    )

    if deviation <= straight:
        edge.curve = None
        fit = EdgeFit(role, "line", len(dense), 2, deviation, dense_length, start.dist(end))
        return edge, fit

    curve = fit_cubic(dense)
    samples = int(cfg["samples"])
    sampled = np.array([_bezier(curve, t / samples) for t in range(samples + 1)])
    max_dev = _max_distance(dense, sampled)
    chord = float(np.linalg.norm(dense[-1] - dense[0]))
    handle = max(float(np.linalg.norm(curve[1] - curve[0])),
                 float(np.linalg.norm(curve[2] - curve[3])))
    overfit = (len(dense) < int(cfg["min_points_for_cubic"])
               or handle > float(cfg["max_handle_ratio"]) * max(chord, 1e-9))
    if max_dev <= float(cfg["max_deviation_mm"]) and not overfit:
        edge.curve = CubicBezier(
            Point2D(*curve[0]), Point2D(*curve[1]), Point2D(*curve[2]), Point2D(*curve[3])
        )
        return edge, EdgeFit(role, "cubic", len(dense), len(simplified), max_dev,
                             dense_length, _length(sampled))

    # The cubic is rejected for one of two reasons: a long S-shaped interval - a side seam
    # or a hem across a whole panel - that no single cubic follows, or too few points to
    # pin the handles down. Either way the simplified polyline is kept instead. That is
    # the representation the reference fixture itself stores, and its error is bounded by
    # the simplification tolerance rather than by a curve fit.
    edge.points = [Point2D(float(x), float(y)) for x, y in simplified]
    return edge, EdgeFit(role, "polyline", len(dense), len(simplified),
                         _max_distance(dense, simplified), dense_length, _length(simplified))


# ---- primitives ----------------------------------------------------------


def simplify(points: np.ndarray, tolerance: float, minimum: int) -> np.ndarray:
    """Douglas-Peucker, endpoints always kept."""
    if len(points) <= 2:
        return points
    keep = np.zeros(len(points), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        lo, hi = stack.pop()
        if hi - lo < 2:
            continue
        segment = points[lo:hi + 1]
        distances = _point_line_distance(segment, points[lo], points[hi])
        k = int(np.argmax(distances))
        if distances[k] > tolerance:
            keep[lo + k] = True
            stack.append((lo, lo + k))
            stack.append((lo + k, hi))
    if int(keep.sum()) < minimum:
        step = max(1, (len(points) - 1) // (minimum - 1))
        for i in range(0, len(points), step):
            keep[i] = True
        keep[-1] = True
    return points[keep]


def fit_cubic(points: np.ndarray, refinements: int = 2) -> np.ndarray:
    """Least-squares cubic through a polyline with its endpoints pinned.

    The two interior control points are the only unknowns, which keeps the semantic
    endpoints exact by construction rather than by a constraint.

    The parameters start from chord length and are then re-estimated by projecting each
    point onto the current curve. A chord-length guess alone costs about a percent of the
    span on a tight S-curve, which is the difference between an interval that fits as a
    cubic and one that falls back to a polyline.
    """
    p0, p3 = points[0], points[-1]
    t = _chord_parameters(points)
    control = np.array([p0, p0, p3, p3])
    for _ in range(max(1, refinements + 1)):
        control = _solve_cubic(points, t, p0, p3)
        if refinements:
            t = _reproject(points, control)
    return control


def _solve_cubic(points: np.ndarray, t: np.ndarray, p0, p3) -> np.ndarray:
    b0 = (1 - t) ** 3
    b1 = 3 * (1 - t) ** 2 * t
    b2 = 3 * (1 - t) * t ** 2
    b3 = t ** 3
    basis = np.column_stack([b1, b2])
    target = points - np.outer(b0, p0) - np.outer(b3, p3)
    solution, *_ = np.linalg.lstsq(basis, target, rcond=None)
    return np.array([p0, solution[0], solution[1], p3])


def _reproject(points: np.ndarray, control: np.ndarray, samples: int = 256) -> np.ndarray:
    """Parameter of the nearest point on the curve, for each boundary point."""
    grid = np.linspace(0.0, 1.0, samples + 1)
    curve = np.array([_bezier(control, float(u)) for u in grid])
    out = np.empty(len(points))
    for i, p in enumerate(points):
        out[i] = grid[int(np.argmin(np.linalg.norm(curve - p, axis=1)))]
    out[0], out[-1] = 0.0, 1.0
    # A non-monotone parameterisation would fold the fit back on itself.
    return np.maximum.accumulate(out)


def _bezier(control: np.ndarray, t: float) -> np.ndarray:
    u = 1.0 - t
    return (u ** 3 * control[0] + 3 * u * u * t * control[1]
            + 3 * u * t * t * control[2] + t ** 3 * control[3])


def _chord_parameters(points: np.ndarray) -> np.ndarray:
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total = steps.sum()
    if total <= 0:
        return np.linspace(0.0, 1.0, len(points))
    return np.concatenate([[0.0], np.cumsum(steps) / total])


def _length(points: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def _point_line_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    direction = b - a
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        return np.linalg.norm(points - a, axis=1)
    offset = points - a
    cross = direction[0] * offset[:, 1] - direction[1] * offset[:, 0]
    return np.abs(cross) / norm


def _chord_deviation(points: np.ndarray) -> float:
    return float(np.max(_point_line_distance(points, points[0], points[-1])))


def _max_distance(points: np.ndarray, polyline: np.ndarray) -> float:
    """Largest distance from any boundary point to the fitted polyline.

    Distance to the *segments*, not to their endpoints: a simplified polyline has only a
    handful of vertices, and measuring against those alone would report a deviation many
    times the real one.
    """
    a = polyline[:-1]
    b = polyline[1:]
    direction = b - a
    length2 = np.maximum(np.einsum("ij,ij->i", direction, direction), 1e-18)
    worst = 0.0
    for p in points:
        t = np.clip(np.einsum("ij,ij->i", p - a, direction) / length2, 0.0, 1.0)
        closest = a + direction * t[:, None]
        worst = max(worst, float(np.min(np.linalg.norm(closest - p, axis=1))))
    return worst
