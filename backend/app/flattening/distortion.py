"""Flattening diagnostics.

Every measure compares the flattened triangle against the same triangle's isometric
unfolding, which is an exact copy of the 3D triangle, so a zero reading means the panel
developed without stretching at all. The plane and cylinder fixtures are expected to read
zero; a torso panel cannot, which is the whole reason the diagnostics are exported rather
than merely checked.
"""
from __future__ import annotations

import numpy as np
from shapely.geometry import LinearRing

from .triangulation import _angles, triangle_areas


def measure(uv: np.ndarray, faces: np.ndarray, local: np.ndarray, boundary: list[int]) -> dict:
    flat = uv[faces]
    ref_area = triangle_areas(local)
    flat_area = triangle_areas(flat)

    # The reference unfolding is wound positively by construction, so a flattened
    # triangle with the opposite sign has turned over.
    flipped = np.where(np.sign(flat_area) != np.sign(ref_area))[0]

    ref_angles = np.degrees(_angles(local))
    flat_angles = np.degrees(_angles(flat))
    angular = np.abs(flat_angles - ref_angles)

    area_ratio = np.abs(flat_area) / np.maximum(np.abs(ref_area), 1e-12)
    area_distortion = np.abs(area_ratio - 1.0)

    ref_edges = _edge_lengths(local)
    flat_edges = _edge_lengths(flat)
    edge_distortion = np.abs(flat_edges / np.maximum(ref_edges, 1e-12) - 1.0)

    ring_points = [(float(uv[i, 0]), float(uv[i, 1])) for i in boundary]
    try:
        simple = LinearRing(ring_points).is_simple
    except Exception:
        simple = False

    per_triangle = np.max(angular, axis=1)
    return {
        "triangles": int(len(faces)),
        "meanAngularDistortionDeg": round(float(angular.mean()), 6),
        "maxAngularDistortionDeg": round(float(angular.max()), 6),
        "meanAreaDistortion": round(float(area_distortion.mean()), 6),
        "maxAreaDistortion": round(float(area_distortion.max()), 6),
        "meanEdgeLengthDistortion": round(float(edge_distortion.mean()), 6),
        "maxEdgeLengthDistortion": round(float(edge_distortion.max()), 6),
        "flippedTriangles": int(len(flipped)),
        "flippedTriangleIds": [int(i) for i in flipped[:50]],
        "boundarySelfIntersections": 0 if simple else 1,
        "surfaceAreaMm2": round(float(np.sum(np.abs(ref_area))), 4),
        "flattenedAreaMm2": round(float(np.sum(np.abs(flat_area))), 4),
        "perTriangleAngularDeg": [round(float(v), 4) for v in per_triangle],
    }


def heat_map(uv: np.ndarray, faces: np.ndarray, values: list[float]) -> dict:
    """Flattened triangles plus one value each, for the interface's heat map."""
    return {
        "triangles": [[[round(float(c), 3) for c in uv[v]] for v in face] for face in faces],
        "values": values,
        "max": round(max(values), 4) if values else 0.0,
    }


def _edge_lengths(points: np.ndarray) -> np.ndarray:
    return np.stack([
        np.linalg.norm(points[:, 1] - points[:, 0], axis=1),
        np.linalg.norm(points[:, 2] - points[:, 1], axis=1),
        np.linalg.norm(points[:, 0] - points[:, 2], axis=1),
    ], axis=1)
