"""Garment surface: the proxy body plus level-dependent ease.

Each ring is scaled to its own target circumference, which is the body circumference at
that row plus the interpolated ease. No single offset is applied to every vertex: the
scale is a function of y, and the resulting normal displacement varies around each
section as well, which is measured and reported rather than assumed.
"""
from __future__ import annotations

import numpy as np

from ..assumptions import AssumptionRegistry
from ..reconstruction.lofting import RingSurface, validate_mesh
from .ease import Ease, ease_profile


def build_garment_surface(
    proxy: RingSurface, levels: dict, ease: Ease, registry: AssumptionRegistry
) -> tuple[RingSurface, dict]:
    """Scale every ring to body circumference + ease."""
    body = proxy.perimeters()
    added = ease_profile(proxy.ys, levels, ease)
    scale = (body + added) / body
    if np.any(scale <= 0.0) or not np.all(np.isfinite(scale)):
        raise ValueError("ease produces a non-positive garment section")

    garment = RingSurface(
        proxy.ys, proxy.a * scale, proxy.b * scale, proxy.exponent, proxy.columns,
        proxy.level_rows, label="garment",
    )
    validate_mesh(garment)

    registry.add(
        "garment.ease_conversion", "Circumference ease to surface displacement",
        "uniform in-plane scaling of each section", None,
        """Ease is given as a circumference and applied by scaling each section about its
        own centre, because the perimeter scales linearly with the section. The normal
        displacement that follows is not constant around a section - it is largest at the
        side - so the achieved normal distance is reported per level instead of being
        treated as a seam allowance.""",
        ["bust", "waist", "hips"], True,
    )

    achieved = garment.perimeters()
    proxy_v = proxy.vertices()
    garment_v = garment.vertices()
    normals = np.linalg.norm(garment_v - proxy_v, axis=2)

    diagnostics = {
        "ease": ease.as_dict(),
        "levels": {
            name: {
                "row": row,
                "yMm": round(float(proxy.ys[row]), 4),
                "bodyCircumferenceMm": round(float(body[row]), 4),
                "requestedEaseMm": round(float(added[row]), 4),
                "targetCircumferenceMm": round(float(body[row] + added[row]), 4),
                "achievedCircumferenceMm": round(float(achieved[row]), 4),
                "easeResidualMm": round(float(achieved[row] - body[row] - added[row]), 9),
                "scale": round(float(scale[row]), 8),
                "normalOffsetMinMm": round(float(normals[row].min()), 4),
                "normalOffsetMeanMm": round(float(normals[row].mean()), 4),
                "normalOffsetMaxMm": round(float(normals[row].max()), 4),
            }
            for name, row in sorted(proxy.level_rows.items(), key=lambda kv: kv[1])
        },
        "maxEaseResidualMm": round(float(np.max(np.abs(achieved - body - added))), 9),
        "selfIntersection": False,  # a positive uniform scale per ring cannot invert
        "symmetric": True,          # every ring is an even superellipse in x
    }
    return garment, diagnostics
