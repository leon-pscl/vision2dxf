"""Constrained triangulation of a surface panel.

The panel arrives as a structured grid whose grid lines *are* the seam paths, so the
triangulation is constrained without a constrained triangulator: no triangle can cross a
seam, because every seam is a row or a column of the grid. This module therefore only
splits the quads (already done by the segmentation) and measures the mesh quality that the
parameterisation depends on, plus the isometric local frame of every triangle, which is
the one piece of geometry both LSCM and ARAP need.
"""
from __future__ import annotations

import numpy as np


def local_frames(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """(F, 3, 2) isometric unfolding of each triangle into its own plane.

    Vertex 0 at the origin, vertex 1 on +x, vertex 2 above the x axis. Edge lengths and
    angles are preserved exactly, so these coordinates are the reference shape every
    distortion measure compares the flattened triangle against.
    """
    p0, p1, p2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    e1, e2 = p1 - p0, p2 - p0
    len1 = np.linalg.norm(e1, axis=1)
    safe = np.where(len1 > 0, len1, 1.0)
    x3 = np.einsum("ij,ij->i", e2, e1) / safe
    y3 = np.linalg.norm(np.cross(e1, e2), axis=1) / safe
    out = np.zeros((len(faces), 3, 2))
    out[:, 1, 0] = len1
    out[:, 2, 0] = x3
    out[:, 2, 1] = y3
    return out


def triangle_areas(points: np.ndarray) -> np.ndarray:
    """Signed area of every triangle given as (F, 3, 2)."""
    a, b, c = points[:, 0], points[:, 1], points[:, 2]
    return 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                  - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))


def mesh_quality(verts: np.ndarray, faces: np.ndarray) -> dict:
    """Quality of the 3D triangulation, before anything is flattened."""
    local = local_frames(verts, faces)
    areas = np.abs(triangle_areas(local))
    edges = np.stack([
        np.linalg.norm(local[:, 1] - local[:, 0], axis=1),
        np.linalg.norm(local[:, 2] - local[:, 1], axis=1),
        np.linalg.norm(local[:, 0] - local[:, 2], axis=1),
    ], axis=1)
    angles = np.degrees(_angles(local))
    return {
        "triangles": int(len(faces)),
        "vertices": int(len(verts)),
        "minAngleDeg": round(float(angles.min()), 4),
        "maxAngleDeg": round(float(angles.max()), 4),
        "minAreaMm2": round(float(areas.min()), 6),
        "maxAspectRatio": round(float(np.max(edges.max(axis=1) / edges.min(axis=1))), 4),
        "degenerateTriangles": int(np.sum(areas <= 1e-9)),
    }


def _angles(points: np.ndarray) -> np.ndarray:
    """(F, 3) interior angles in radians."""
    out = np.empty((len(points), 3))
    for k in range(3):
        a = points[:, k]
        b = points[:, (k + 1) % 3]
        c = points[:, (k + 2) % 3]
        u, v = b - a, c - a
        nu = np.linalg.norm(u, axis=1)
        nv = np.linalg.norm(v, axis=1)
        cos = np.einsum("ij,ij->i", u, v) / np.maximum(nu * nv, 1e-12)
        out[:, k] = np.arccos(np.clip(cos, -1.0, 1.0))
    return out


def cotangents(points: np.ndarray) -> np.ndarray:
    """(F, 3) cotangent of the angle at each corner of each triangle."""
    out = np.empty((len(points), 3))
    for k in range(3):
        a = points[:, k]
        u, v = points[:, (k + 1) % 3] - a, points[:, (k + 2) % 3] - a
        cross = np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])
        out[:, k] = np.einsum("ij,ij->i", u, v) / np.maximum(cross, 1e-12)
    return out
