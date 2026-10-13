"""Seam balancing.

Two panels that share a seam are flattened independently, so the seam comes out of the
parameterisation at two slightly different lengths even though it was one curve on the
garment surface. The difference is pure flattening distortion, and it has to be removed
before the pattern is sewable.

Balancing happens on the fitted curve, not on the mesh: the two interior control points of
the longer edge are moved towards the chord (or the shorter edge's away from it) until the
two lengths agree. The endpoints never move, so the semantic corners stay exact and the
panel stays closed.

`ponytail:` this corrects the symptom. The principled fix is to flatten both panels in one
solve with a hard equal-arc-length constraint across the shared seam. Worth doing when a
garment arrives whose paired seams differ by more than the refit tolerance can absorb -
the diagnostics below say when that happens.
"""
from __future__ import annotations

import numpy as np

from ..config import reconstruction_config
from ..drafting.geometry import CubicBezier, PatternEdge, Point2D


def balance_pairs(pairs: list[tuple[PatternEdge, PatternEdge]]) -> dict:
    """Match each paired seam's two lengths. Returns the per-pair diagnostics."""
    cfg = reconstruction_config()["seams"]["balance"]
    tolerance = float(cfg["tolerance_mm"])
    max_iterations = int(cfg["max_iterations"])

    rounds = []
    for a, b in pairs:
        before = (a.length_mm, b.length_mm)
        target = _target_length(a, b)
        moved_a = _adjust_to_length(a, target, tolerance, max_iterations)
        moved_b = _adjust_to_length(b, target, tolerance, max_iterations)
        after = (a.length_mm, b.length_mm)
        rounds.append({
            "edges": [a.ensure_id(), b.ensure_id()],
            "roles": [a.role, b.role],
            "panels": [a.key.split(":")[0], b.key.split(":")[0]],
            "beforeMm": [round(before[0], 4), round(before[1], 4)],
            "afterMm": [round(after[0], 4), round(after[1], 4)],
            "targetMm": round(target, 4),
            "mismatchBeforeMm": round(abs(before[0] - before[1]), 5),
            "mismatchAfterMm": round(abs(after[0] - after[1]), 5),
            "toleranceMm": tolerance,
            "balanced": abs(after[0] - after[1]) <= tolerance,
            "adjustments": [moved_a, moved_b],
        })
    worst = max((r["mismatchAfterMm"] for r in rounds), default=0.0)
    return {
        "toleranceMm": tolerance,
        "pairs": rounds,
        "maxMismatchAfterMm": round(worst, 5),
        "allBalanced": all(r["balanced"] for r in rounds),
    }


def _adjustable(edge: PatternEdge) -> bool:
    """Whether the edge has interior geometry whose offset from its chord can be scaled."""
    return edge.curve is not None or len(edge.points) > 2


def _target_length(a: PatternEdge, b: PatternEdge) -> float:
    """A straight edge cannot change length, so it sets the target for the pair."""
    if not _adjustable(a) and _adjustable(b):
        return a.length_mm
    if not _adjustable(b) and _adjustable(a):
        return b.length_mm
    return 0.5 * (a.length_mm + b.length_mm)


def _adjust_to_length(edge: PatternEdge, target: float, tolerance: float, iterations: int) -> dict:
    """Bisect the control-point offset until the edge reaches the target length."""
    if not _adjustable(edge):
        return {"edge": edge.ensure_id(), "method": "none", "reason": "straight edge",
                "lambda": 0.0, "maxControlShiftMm": 0.0,
                "residualMm": round(abs(edge.length_mm - target), 5)}
    if abs(edge.length_mm - target) <= tolerance:
        return {"edge": edge.ensure_id(), "method": "none", "reason": "already within tolerance",
                "lambda": 0.0, "maxControlShiftMm": 0.0,
                "residualMm": round(abs(edge.length_mm - target), 5)}

    original_curve, original_points = edge.curve, list(edge.points)

    def apply(factor: float) -> float:
        _set(edge, original_curve, original_points, factor)
        return edge.length_mm

    # Length falls monotonically as the factor runs from -1 (interior geometry pushed
    # further off the chord) to 1 (pulled onto it, giving a straight segment).
    lo, hi = -1.0, 1.0
    if apply(hi) > target:
        chosen = hi
    elif apply(lo) < target:
        chosen = lo
    else:
        chosen = 0.0
        for _ in range(max(1, iterations) * 4):
            chosen = 0.5 * (lo + hi)
            if abs(apply(chosen) - target) <= tolerance * 0.1:
                break
            if edge.length_mm > target:
                lo = chosen
            else:
                hi = chosen
    _set(edge, original_curve, original_points, chosen)

    if original_curve is not None and edge.curve is not None:
        shift = max(original_curve.c1.dist(edge.curve.c1), original_curve.c2.dist(edge.curve.c2))
    else:
        shift = max((a.dist(b) for a, b in zip(original_points, edge.points)), default=0.0)
    return {
        "edge": edge.ensure_id(), "method": "chord_offset_scaling",
        "lambda": round(chosen, 6), "maxControlShiftMm": round(shift, 5),
        "residualMm": round(abs(edge.length_mm - target), 5),
    }


def _set(edge: PatternEdge, curve, points, factor: float) -> None:
    """Rewrite the edge's interior geometry at the given chord-offset factor."""
    if curve is not None:
        edge.curve = _shift(curve, factor)
        return
    a, b = points[0], points[-1]
    edge.points = [points[0]] + [
        _toward_chord(p, a, b, factor) for p in points[1:-1]
    ] + [points[-1]]


def _shift(curve: CubicBezier, factor: float) -> CubicBezier:
    """Move both interior control points `factor` of the way onto the chord."""
    return CubicBezier(
        curve.p0,
        _toward_chord(curve.c1, curve.p0, curve.p3, factor),
        _toward_chord(curve.c2, curve.p0, curve.p3, factor),
        curve.p3,
    )


def _toward_chord(point: Point2D, a: Point2D, b: Point2D, factor: float) -> Point2D:
    direction = np.array([b.x - a.x, b.y - a.y])
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        return point
    unit = direction / norm
    offset = np.array([point.x - a.x, point.y - a.y])
    projected = np.array([a.x, a.y]) + unit * float(offset @ unit)
    target = np.array([point.x, point.y]) + factor * (projected - np.array([point.x, point.y]))
    return Point2D(float(target[0]), float(target[1]))


def _length(curve: CubicBezier, samples: int = 128) -> float:
    pts = np.array([[p.x, p.y] for p in (curve.point_at(i / samples) for i in range(samples + 1))])
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def surface_seam_lengths(panels, pairs) -> list[dict]:
    """Paired-seam lengths on the garment surface, before any flattening.

    These are expected to be identical: the two sides of a seam are the same curve on the
    surface, or its mirror image. Any difference here is a defect in the seam network, not
    in the flattening, which is why it is measured separately.
    """
    index = {p.key: p for p in panels}
    out = []
    for (key_a, role_a), (key_b, role_b) in pairs:
        a = _path_length(index[key_a], role_a)
        b = _path_length(index[key_b], role_b)
        out.append({
            "seam": role_a,
            "panels": [key_a, key_b],
            "lengthsMm": [round(a, 6), round(b, 6)],
            "differenceMm": round(abs(a - b), 9),
        })
    return out


def _path_length(panel, role: str) -> float:
    ids = panel.semantic_boundaries[role]
    pts = panel.vertices_3d[ids]
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
