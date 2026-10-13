"""Proxy torso lofting.

The surface is a stack of consistently sampled rings. Every ring is one superellipse, so
adjacent rings share their angular correspondence exactly, every ring has a computable
perimeter, and left/right symmetry is exact rather than approximate. Rows are inserted
between the anatomical levels and the half width and half depth are interpolated along y.

Storing the stack as (y, a, b) rather than as loose vertices is what lets the smoothing
pass preserve a measured circumference exactly, and lets a panel be cut at any y without
re-meshing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import reconstruction_config
from .perimeter_solver import perimeter, ring


class MeshError(Exception):
    def __init__(self, errors: list[dict]) -> None:
        self.errors = errors
        super().__init__("; ".join(e["message"] for e in errors))


@dataclass
class RingSurface:
    """A lofted surface as a ring stack."""

    ys: np.ndarray            # (R,) ascending, mm
    a: np.ndarray             # (R,) half width
    b: np.ndarray             # (R,) half depth
    exponent: float
    columns: int
    level_rows: dict[str, int]
    label: str = "proxy"

    # ---- sampling ---------------------------------------------------------

    def section_at(self, y: float) -> tuple[float, float]:
        """Half width and half depth at any y, clamped to the stack."""
        return (
            float(np.interp(y, self.ys, self.a)),
            float(np.interp(y, self.ys, self.b)),
        )

    def point(self, y: float, column: int) -> np.ndarray:
        a, b = self.section_at(y)
        pt = ring(a, b, self.exponent, self.columns)[column % self.columns]
        return np.array([pt[0], y, pt[1]])

    def vertices(self) -> np.ndarray:
        """(R, M, 3) grid of surface points."""
        out = np.empty((len(self.ys), self.columns, 3))
        for i, (y, a, b) in enumerate(zip(self.ys, self.a, self.b)):
            pts = ring(float(a), float(b), self.exponent, self.columns)
            out[i, :, 0] = pts[:, 0]
            out[i, :, 1] = y
            out[i, :, 2] = pts[:, 1]
        return out

    def faces(self) -> np.ndarray:
        """Triangles over the closed tube, consistently wound outwards."""
        rows, cols = len(self.ys), self.columns
        tris = []
        for i in range(rows - 1):
            for j in range(cols):
                k = (j + 1) % cols
                v00, v01 = i * cols + j, i * cols + k
                v10, v11 = (i + 1) * cols + j, (i + 1) * cols + k
                tris.append((v00, v11, v10))
                tris.append((v00, v01, v11))
        return np.array(tris, dtype=int)

    def ring_perimeter(self, row: int) -> float:
        return perimeter(float(self.a[row]), float(self.b[row]), self.exponent)

    def perimeters(self) -> np.ndarray:
        return np.array([self.ring_perimeter(i) for i in range(len(self.ys))])

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "exponent": self.exponent,
            "columns": self.columns,
            "rows": len(self.ys),
            "levelRows": self.level_rows,
            "rings": [
                {
                    "row": i,
                    "yMm": round(float(y), 4),
                    "halfWidthMm": round(float(a), 4),
                    "halfDepthMm": round(float(b), 4),
                    "perimeterMm": round(self.ring_perimeter(i), 4),
                }
                for i, (y, a, b) in enumerate(zip(self.ys, self.a, self.b))
            ],
        }

    def obj(self) -> bytes:
        """Wavefront OBJ of the closed tube, for the debugging download."""
        verts = self.vertices().reshape(-1, 3)
        lines = [f"# {self.label} surface, millimetres"]
        lines += [f"v {x:.4f} {y:.4f} {z:.4f}" for x, y, z in verts]
        lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in self.faces()]
        return ("\n".join(lines) + "\n").encode("utf-8")


def loft(sections: dict, levels: dict, columns: int | None = None,
         segments: int | None = None) -> RingSurface:
    """Ring stack through the reconstructed sections."""
    cfg = reconstruction_config()["mesh"]
    columns = int(columns or cfg["cross_section_resolution"])
    segments = int(segments or cfg["longitudinal_segments"])
    if columns % 4:
        raise MeshError([_err("cross_section_resolution", "RESOLUTION_NOT_DIVISIBLE",
                              "The ring resolution must be a multiple of four.")])

    ordered = sorted(sections.values(), key=lambda s: s.y_mm)
    # The neck and upper-chest sections share y = 0; only the upper chest carries the
    # surface, the neck section is a construction section used to place the neckline.
    ordered = [s for s in ordered if s.name != "neck"]
    if len(ordered) < 2:
        raise MeshError([_err("sections", "TOO_FEW_SECTIONS",
                              "At least two sections are needed to loft a surface.")])

    ys: list[float] = []
    for lower, upper in zip(ordered, ordered[1:]):
        for k in range(segments):
            ys.append(lower.y_mm + (upper.y_mm - lower.y_mm) * k / segments)
    ys.append(ordered[-1].y_mm)

    ys_arr = np.array(ys)
    level_y = np.array([s.y_mm for s in ordered])
    a = np.interp(ys_arr, level_y, [s.a_mm for s in ordered])
    b = np.interp(ys_arr, level_y, [s.b_mm for s in ordered])
    level_rows = {s.name: int(np.argmin(np.abs(ys_arr - s.y_mm))) for s in ordered}
    level_rows["neck"] = 0

    surface = RingSurface(ys_arr, a, b, ordered[0].exponent, columns, level_rows)
    validate_mesh(surface)
    return surface


def validate_mesh(surface: RingSurface) -> None:
    """Finite vertices, no zero-area face, consistent winding, manifold edges."""
    errors: list[dict] = []
    verts = surface.vertices().reshape(-1, 3)
    faces = surface.faces()

    if not np.all(np.isfinite(verts)):
        errors.append(_err("vertices", "NON_FINITE_VERTEX",
                           "The lofted surface carries a non-finite vertex."))
    if np.any(surface.a <= 0) or np.any(surface.b <= 0):
        errors.append(_err("sections", "DEGENERATE_SECTION",
                           "A ring has a non-positive width or depth."))

    tri = verts[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    areas = 0.5 * np.linalg.norm(normals, axis=1)
    if np.any(areas <= 1e-9):
        errors.append(_err("faces", "ZERO_AREA_FACE",
                           f"{int(np.sum(areas <= 1e-9))} face(s) have no area."))

    # Outward orientation: the horizontal part of every normal must point away from the
    # vertical axis. A tube built from rings has no other way to be inconsistent.
    centroids = tri.mean(axis=1)
    radial = centroids.copy()
    radial[:, 1] = 0.0
    flat = normals.copy()
    flat[:, 1] = 0.0
    inward = int(np.sum(np.sum(flat * radial, axis=1) < 0))
    if inward:
        errors.append(_err("faces", "INCONSISTENT_FACE_ORIENTATION",
                           f"{inward} face normal(s) point inwards."))

    counts: dict[tuple[int, int], int] = {}
    for a, b, c in faces:
        for edge in ((a, b), (b, c), (c, a)):
            key = (min(edge), max(edge))
            counts[key] = counts.get(key, 0) + 1
    non_manifold = sum(1 for v in counts.values() if v > 2)
    if non_manifold:
        errors.append(_err("faces", "NON_MANIFOLD_EDGE",
                           f"{non_manifold} edge(s) are shared by more than two faces."))
    if errors:
        raise MeshError(errors)


def boundary_edge_count(surface: RingSurface) -> int:
    """Edges used by exactly one face. An open tube has one per column, top and bottom."""
    counts: dict[tuple[int, int], int] = {}
    for a, b, c in surface.faces():
        for edge in ((a, b), (b, c), (c, a)):
            key = (min(edge), max(edge))
            counts[key] = counts.get(key, 0) + 1
    return sum(1 for v in counts.values() if v == 1)


def _err(field: str, code: str, message: str) -> dict:
    return {
        "field": field, "code": code, "message": " ".join(message.split()),
        "suggestedAction": "Lower the mesh resolution or re-check the reconstructed sections.",
        "severity": "blocking",
    }
