"""Pure geometry: widths, depths, girths and contour lengths.

Every function here is deterministic and free of I/O, model calls and
configuration lookups, so it can be tested against synthetic silhouettes with
known answers. Masks are boolean arrays indexed ``[row, column]``; scales are
passed in explicitly. Internal unit is millimetres (float).

Converting between an image row and a height above the floor is not here: it
belongs to a view, and is ``landmarks.ViewScale``.
"""

from __future__ import annotations

import math

import numpy as np

Run = tuple[int, int]

# Identifiers written into the provenance record so a girth can always be
# recomputed from the same inputs by the same equation (brief 12.1).
ELLIPSE_EQUATION_1 = "ellipse_equation_1"
ELLIPSE_EQUATION_2 = "ellipse_equation_2"
MIN_SEGMENT_POINTS = 2  # fewer than two points is not a path and has no length


def run_containing(mask: np.ndarray, y: int, x_ref: int) -> Run | None:
    """Return the inclusive ``(x_start, x_end)`` of the foreground run holding ``x_ref``.

    ``None`` when ``x_ref`` is background or the row is outside the mask.
    """
    if not 0 <= y < mask.shape[0] or not 0 <= x_ref < mask.shape[1]:
        return None
    row = np.asarray(mask[y], dtype=bool)
    if not row[x_ref]:
        return None
    # Walk out to the first background pixel on each side.
    left = x_ref
    while left > 0 and row[left - 1]:
        left -= 1
    right = x_ref
    while right < row.size - 1 and row[right + 1]:
        right += 1
    return left, right


def runs_in_row(mask: np.ndarray, y: int) -> list[Run]:
    """Return every contiguous foreground run on row ``y`` as inclusive spans."""
    if not 0 <= y < mask.shape[0]:
        return []
    row = np.asarray(mask[y], dtype=bool)
    if not row.any():
        return []
    padded = np.concatenate(([False], row, [False])).astype(np.int8)
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    return [(int(start), int(end)) for start, end in zip(starts, ends, strict=True)]


def gap_containing(mask: np.ndarray, y: int, x_ref: int) -> Run | None:
    """Return the background gap on row ``y`` that holds ``x_ref``, between two runs.

    ``None`` when ``x_ref`` is foreground, or when no run lies on both sides of it:
    that is how the armpit and crotch rows are recognised.
    """
    if run_containing(mask, y, x_ref) is not None:
        return None
    runs = runs_in_row(mask, y)
    before = [run for run in runs if run[1] < x_ref]
    after = [run for run in runs if run[0] > x_ref]
    if not before or not after:
        return None
    return before[-1][1] + 1, after[0][0] - 1


def width_at(mask: np.ndarray, y: int, x_ref: int, mm_per_px_x: float) -> float:
    """Return the width in mm of the run containing ``x_ref`` on row ``y``.

    ``0.0`` when there is no foreground run at ``x_ref``.
    """
    run = run_containing(mask, y, x_ref)
    if run is None:
        return 0.0
    start, end = run
    return (end - start + 1) * mm_per_px_x


def depth_at(side_mask: np.ndarray, y: int, x_ref: int, mm_per_px_x: float) -> float:
    """Return the body depth in mm on row ``y`` of a side view.

    Depth runs along the image x axis in a side view, so this is ``width_at``
    under another name; both exist because the call sites read very differently.
    """
    return width_at(side_mask, y, x_ref, mm_per_px_x)


def split_depth_at(
    side_mask: np.ndarray, y: int, balance_x: int, mm_per_px_x: float
) -> tuple[float, float]:
    """Split row ``y`` of a side view into front and back depth in mm.

    The body faces +x after mirroring (brief 5.4), so the run to the right of the
    balance line is the front depth and the run to its left is the back depth.
    Returns ``(0.0, 0.0)`` when the balance line misses the body on this row.
    """
    run = run_containing(side_mask, y, balance_x)
    if run is None:
        return 0.0, 0.0
    start, end = run
    d_front = (end - balance_x + 1) * mm_per_px_x
    d_back = (balance_x - start) * mm_per_px_x
    return d_front, d_back


def ellipse_perimeter(a: float, b: float) -> float:
    """Return the perimeter of an ellipse with semi-axes ``a`` and ``b``.

    Ramanujan's second approximation; exact for a circle and within 0.01% of the
    true value over the aspect ratios a human cross-section produces.
    """
    if a < 0 or b < 0:
        raise ValueError("semi-axes must be non-negative")
    total = a + b
    if total == 0:
        return 0.0
    h = ((a - b) / total) ** 2
    return math.pi * total * (1 + 3 * h / (10 + math.sqrt(4 - 3 * h)))


def semi_ellipse_arc(a: float, b: float) -> float:
    """Return half the perimeter of an ellipse with semi-axes ``a`` and ``b``."""
    return ellipse_perimeter(a, b) / 2


def two_semi_ellipse_girth(width_mm: float, d_front_mm: float, d_back_mm: float) -> float:
    """Return a girth in mm from one width and an asymmetric front/back depth.

    The body is not front-back symmetric, so the cross-section is modelled as two
    half ellipses sharing the width: ``girth = P(a, d_f)/2 + P(a, d_b)/2``
    (brief 5.6).
    """
    a = width_mm / 2
    return semi_ellipse_arc(a, d_front_mm) + semi_ellipse_arc(a, d_back_mm)


def back_arc(width_mm: float, d_back_mm: float) -> float:
    """Return the back half-arc in mm: the ``*_back_width`` measurements."""
    return semi_ellipse_arc(width_mm / 2, d_back_mm)


def _to_mm(points: np.ndarray, mm_per_px_x: float, mm_per_px_y: float) -> np.ndarray:
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    return points * np.array([mm_per_px_x, mm_per_px_y])


def contour_length(points: np.ndarray, mm_per_px_x: float, mm_per_px_y: float) -> float:
    """Return the polyline length in mm through ``points``, an ``(N, 2)`` array of px."""
    mm = _to_mm(points, mm_per_px_x, mm_per_px_y)
    if len(mm) <= 1:
        return 0.0
    return float(np.hypot(*np.diff(mm, axis=0).T).sum())


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    """Return the 2-D cross product; np.cross dropped 2-D vectors in numpy 2."""
    return float(a[0] * b[1] - a[1] * b[0])


def hull_chain(points: np.ndarray, side: str = "front") -> np.ndarray:
    """Return the convex chain over ``points`` that bridges hollows on one side.

    ``points`` is an ``(N, 2)`` array of pixels along a contour that is monotone
    in row. ``side="front"`` keeps the chain at maximum x, ``"back"`` at minimum
    x. Monotone chain in the ``(y, x)`` plane; no cv2, numpy alone.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(pts) <= 1:
        return pts
    swapped = pts[:, ::-1]  # (y, x): row is the abscissa
    swapped = swapped[np.argsort(swapped[:, 0], kind="stable")]
    sign = 1.0 if side == "front" else -1.0
    chain: list[np.ndarray] = []
    for point in swapped:
        while len(chain) > 1 and sign * _cross2(chain[-1] - chain[-2], point - chain[-2]) >= 0:
            chain.pop()
        chain.append(point)
    return np.array(chain)[:, ::-1]


def hull_length(
    points: np.ndarray, mm_per_px_x: float, mm_per_px_y: float, side: str = "front"
) -> float:
    """Return the length in mm of the tape path over ``points``.

    Tape-measure behaviour: the tape bridges hollows (under the bust, the small
    of the back) instead of sinking into them, so this is ``contour_length`` of
    the one-sided convex chain and is never longer than the contour itself.
    """
    return contour_length(hull_chain(points, side), mm_per_px_x, mm_per_px_y)


def ramanujan_i(a: float, b: float) -> float:
    """Return Ramanujan's first ellipse-perimeter approximation.

    ``pi * [3(a + b) - sqrt((3a + b)(a + 3b))]``. Exact for a circle and within
    0.04% while the axes stay within a factor of three of each other, which is
    every human torso cross-section; it falls away on a very flat ellipse.

    Raises:
        ValueError: if either semi-axis is negative.

    """
    if a < 0 or b < 0:
        raise ValueError("semi-axes must be non-negative")
    return math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))


def ramanujan_ii(a: float, b: float) -> float:
    """Return Ramanujan's second ellipse-perimeter approximation.

    The elongated-ellipse equation: it holds to 0.01% where
    :func:`ramanujan_i` has already drifted, so it is what a limb or a very flat
    torso section uses.

    Raises:
        ValueError: if either semi-axis is negative.

    """
    return ellipse_perimeter(a, b)


def ellipse_perimeter_montazerian(a: float, b: float) -> tuple[float, str]:
    """Return a girth in mm and the identifier of the equation that produced it.

    Montazerian's two-equation rule (brief 12.1): the first approximation while
    ``a <= 3b``, the elongated-ellipse approximation beyond it. ``a`` is the
    larger semi-axis by construction at the call site; this function sorts them
    anyway so the rule cannot be read against the wrong axis.

    Raises:
        ValueError: if either semi-axis is negative.

    """
    major, minor = max(a, b), min(a, b)
    if minor == 0 and major == 0:
        return 0.0, ELLIPSE_EQUATION_1
    if major <= 3 * minor:
        return ramanujan_i(major, minor), ELLIPSE_EQUATION_1
    return ramanujan_ii(major, minor), ELLIPSE_EQUATION_2


def euclidean_length(points: np.ndarray, mm_per_px: float) -> float:
    """Return the straight-line distance in mm between the first and last point."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(pts) < MIN_SEGMENT_POINTS:
        return 0.0
    return float(np.hypot(*(pts[-1] - pts[0]))) * mm_per_px


def polyline_length(points_mm: np.ndarray) -> float:
    """Return the length in mm of a polyline whose points are already in mm."""
    return contour_length(points_mm, 1.0, 1.0)


def convex_hull_path_length(points_mm: np.ndarray, side: str = "front") -> float:
    """Return the tape-path length in mm over points already in mm.

    Named for the GarmentCodeData measurement family; the body is
    :func:`hull_length`, which bridges hollows the way a tape does.
    """
    return hull_length(points_mm, 1.0, 1.0, side)


def angle_deg(p1: tuple[float, float], vertex: tuple[float, float],
              p2: tuple[float, float]) -> float:
    """Return the angle in degrees at ``vertex`` between the rays to ``p1`` and ``p2``.

    ``0.0`` when either ray has no length, because an angle at a coincident
    point is undefined and a measurement must never be invented.
    """
    a = np.asarray(p1, dtype=float) - np.asarray(vertex, dtype=float)
    b = np.asarray(p2, dtype=float) - np.asarray(vertex, dtype=float)
    norms = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
    if norms == 0:
        return 0.0
    cosine = float(np.dot(a, b)) / norms
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def montazerian_girth(width_mm: float, depth_mm: float) -> tuple[float, str]:
    """Return a circumference in mm from a front breadth and a side depth.

    The cross-section is modelled as one ellipse whose axes are the measured
    breadth and depth, so the semi-axes are ``max/2`` and ``min/2`` and the
    perimeter follows Montazerian's two-equation rule (brief 12.1). Returns the
    girth and the equation identifier, both of which go into provenance.
    """
    return ellipse_perimeter_montazerian(max(width_mm, depth_mm) / 2, min(width_mm, depth_mm) / 2)
