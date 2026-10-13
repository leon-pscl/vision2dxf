"""Constraint-preserving smoothing.

A loft that interpolates the half width and half depth linearly between levels leaves a
crease at every level. This pass replaces the linear interpolation with a monotone cubic
(Fritsch-Carlson tangents) through the same level values, which does three useful things
at once:

  * the curve passes exactly through every control value, so a measured circumference
    comes out of the smoothing untouched rather than merely close;
  * it is C1 between the levels, so the crease is gone;
  * monotone means it cannot overshoot. Smoothing can never invent a waist narrower or a
    bust wider than the one that was measured, which a plain spline would.

It has no tuning parameter, which is deliberate: the tangents are determined by the data.

A Laplacian pass over the same stack was the obvious first choice and is the wrong one:
with the level rows pinned it leaves a piecewise-linear profile exactly as it found it,
because the only non-zero Laplacian is at the kinks, and the kinks are the pinned rows.
"""
from __future__ import annotations

import numpy as np

from .lofting import RingSurface, validate_mesh


def smooth(surface: RingSurface) -> tuple[RingSurface, dict]:
    rows = sorted({row for row in surface.level_rows.values()})
    control_y = surface.ys[rows]
    before = surface.perimeters()

    a = monotone_cubic(control_y, surface.a[rows], surface.ys)
    b = monotone_cubic(control_y, surface.b[rows], surface.ys)

    smoothed = RingSurface(surface.ys, a, b, surface.exponent, surface.columns,
                           surface.level_rows, surface.label)
    validate_mesh(smoothed)
    after = smoothed.perimeters()

    drift = {
        name: round(float(abs(after[row] - before[row])), 9)
        for name, row in surface.level_rows.items()
    }
    free = [i for i in range(len(surface.ys)) if i not in set(rows)]
    return smoothed, {
        "method": "monotone_cubic",
        "controlRows": rows,
        "levelPerimeterDriftMm": drift,
        "maxLevelPerimeterDriftMm": round(max(drift.values()) if drift else 0.0, 9),
        "maxRowChangeMm": round(
            float(np.max(np.abs(after[free] - before[free]))) if free else 0.0, 6),
        "overshoot": False,  # guaranteed by the monotone tangents
    }


def monotone_cubic(x: np.ndarray, y: np.ndarray, at: np.ndarray) -> np.ndarray:
    """Fritsch-Carlson monotone cubic interpolation of (x, y), evaluated at `at`."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return np.interp(at, x, y)

    h = np.diff(x)
    delta = np.diff(y) / h
    tangents = np.empty_like(y)
    tangents[0], tangents[-1] = delta[0], delta[-1]
    for i in range(1, len(y) - 1):
        if delta[i - 1] * delta[i] <= 0.0:
            tangents[i] = 0.0          # a turning point stays a turning point
        else:
            w1, w2 = 2.0 * h[i] + h[i - 1], h[i] + 2.0 * h[i - 1]
            tangents[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])

    index = np.clip(np.searchsorted(x, at, side="right") - 1, 0, len(x) - 2)
    hi = h[index]
    t = (at - x[index]) / hi
    t2, t3 = t * t, t * t * t
    return (
        (2 * t3 - 3 * t2 + 1) * y[index]
        + (t3 - 2 * t2 + t) * hi * tangents[index]
        + (-2 * t3 + 3 * t2) * y[index + 1]
        + (t3 - t2) * hi * tangents[index + 1]
    )
