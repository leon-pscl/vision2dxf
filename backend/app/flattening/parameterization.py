"""Flattening: least-squares conformal map, then as-rigid-as-possible refinement.

LSCM (Levy et al., 2002) gives the initial layout. It is a single linear least-squares
problem over the discrete Cauchy-Riemann conditions, which makes it robust and free of a
starting guess, but it preserves angles at the cost of area. ARAP (Liu et al., 2008) then
refines it: alternate a per-triangle best-fit rotation against the triangle's isometric
shape (local step) with a cotangent-Laplacian solve for the vertex positions (global
step). The result trades a little angle distortion for much less length and area
distortion, which is what a sewn panel needs.

The objective the design asks for maps onto this as follows:

    E_distortion  the ARAP energy, weighted by w_distortion. This is the term that
                  decides the result.
    E_boundary    soft positional constraints holding the semantic interval corners
                  where the conformal initialisation put them, weighted by w_boundary.
    E_seam        the same mechanism on the corners shared by two paired seams, weighted
                  by w_seam, so a seam end cannot slide along the boundary.
    E_grain       realised exactly, not approximately: the grain is fixed by a rigid
                  rotation after the solve, and a rigid motion changes no distortion, so
                  the minimiser of the grain term is reached at zero cost to the others.
    E_flip        a hard rejection rather than a penalty, which is the strongest form the
                  term can take: a layout with any flipped triangle is refused.

Rigid-motion ambiguity is removed by pinning two non-collinear vertices in LSCM and by
the positional constraints in ARAP.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import reconstruction_config
from .triangulation import cotangents, local_frames, triangle_areas


class FlatteningError(Exception):
    def __init__(self, errors: list[dict]) -> None:
        self.errors = errors
        super().__init__("; ".join(e["message"] for e in errors))


@dataclass
class Flattening:
    uv: np.ndarray                 # (N, 2) millimetres
    faces: np.ndarray
    local: np.ndarray              # (F, 3, 2) isometric reference shapes
    method: str
    iterations: int
    converged: bool
    energy: float
    energy_history: list[float] = field(default_factory=list)
    pinned: list[int] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "iterations": self.iterations,
            "converged": self.converged,
            "energy": round(self.energy, 9),
            "energyHistory": [round(e, 9) for e in self.energy_history],
            "pinnedVertices": self.pinned,
        }


def lscm(verts: np.ndarray, faces: np.ndarray, pin_a: int, pin_b: int) -> np.ndarray:
    """Least-squares conformal map with two pinned vertices."""
    local = local_frames(verts, faces)
    n, m = len(verts), len(faces)
    area = np.abs(triangle_areas(local))
    weight = 1.0 / np.sqrt(np.maximum(2.0 * area, 1e-12))

    free = [i for i in range(n) if i not in (pin_a, pin_b)]
    index = {v: k for k, v in enumerate(free)}
    columns = 2 * len(free)
    matrix = np.zeros((2 * m, columns))
    rhs = np.zeros(2 * m)

    # Pin the two vertices a unit apart along +x; the layout is rescaled afterwards.
    chord = float(np.linalg.norm(verts[pin_b] - verts[pin_a]))
    fixed = {pin_a: (0.0, 0.0), pin_b: (chord, 0.0)}

    for t in range(m):
        w = weight[t]
        for j in range(3):
            p1, p2 = local[t, (j + 1) % 3], local[t, (j + 2) % 3]
            wr, wi = (p2[0] - p1[0]) * w, (p2[1] - p1[1]) * w
            vertex = int(faces[t, j])
            if vertex in fixed:
                u0, v0 = fixed[vertex]
                rhs[2 * t] -= wr * u0 - wi * v0
                rhs[2 * t + 1] -= wi * u0 + wr * v0
                continue
            k = index[vertex]
            matrix[2 * t, k] += wr
            matrix[2 * t, len(free) + k] += -wi
            matrix[2 * t + 1, k] += wi
            matrix[2 * t + 1, len(free) + k] += wr

    solution, *_ = np.linalg.lstsq(matrix, rhs, rcond=None)
    uv = np.zeros((n, 2))
    for vertex, (u0, v0) in fixed.items():
        uv[vertex] = (u0, v0)
    for vertex, k in index.items():
        uv[vertex] = (solution[k], solution[len(free) + k])
    return _rescale_to_surface(uv, faces, local)


def _rescale_to_surface(uv: np.ndarray, faces: np.ndarray, local: np.ndarray) -> np.ndarray:
    """Scale the conformal layout so its total area matches the surface area.

    LSCM fixes the layout only up to a similarity. Starting ARAP from a layout of the
    right size costs nothing and keeps the first local step honest.
    """
    surface = float(np.sum(np.abs(triangle_areas(local))))
    flat = float(np.sum(np.abs(triangle_areas(uv[faces]))))
    if flat <= 1e-12 or surface <= 1e-12:
        return uv
    return uv * np.sqrt(surface / flat)


def arap(
    verts: np.ndarray,
    faces: np.ndarray,
    uv0: np.ndarray,
    pins: dict[int, tuple[float, float]],
    weights: dict[int, float],
) -> Flattening:
    """As-rigid-as-possible refinement of an initial layout.

    `pins` holds the target position of every softly constrained vertex and `weights` its
    weight, so the boundary and seam terms of the objective can be weighted differently
    from the two anchors that remove the rigid motion.
    """
    cfg = reconstruction_config()["flattening"]["arap"]
    local = local_frames(verts, faces)
    cot = cotangents(local)
    n = len(verts)

    laplacian = np.zeros((n, n))
    # Edge (i, j) of a triangle is weighted by the cotangent of the angle opposite it.
    for t, (v0, v1, v2) in enumerate(faces):
        for k, (i, j) in enumerate(((v1, v2), (v2, v0), (v0, v1))):
            w = cot[t, k]
            laplacian[i, i] += w
            laplacian[j, j] += w
            laplacian[i, j] -= w
            laplacian[j, i] -= w
    for vertex in pins:
        laplacian[vertex, vertex] += weights[vertex]
    # A Laplacian with only soft pins is still singular up to nothing, but a tiny
    # regulariser keeps the dense solve well conditioned on degenerate inputs.
    laplacian += np.eye(n) * 1e-9

    uv = uv0.copy()
    history: list[float] = []
    converged = False
    iterations = 0
    for iterations in range(1, int(cfg["max_iterations"]) + 1):
        rotations = _best_rotations(uv, faces, local, cot)
        rhs = np.zeros((n, 2))
        for t, (v0, v1, v2) in enumerate(faces):
            rot = rotations[t]
            for k, (i, j) in enumerate(((v1, v2), (v2, v0), (v0, v1))):
                w = cot[t, k]
                a = local[t, (k + 1) % 3]
                b = local[t, (k + 2) % 3]
                delta = rot @ (a - b)
                rhs[i] += w * delta
                rhs[j] -= w * delta
        for vertex, target in pins.items():
            rhs[vertex] += weights[vertex] * np.asarray(target)
        uv = np.linalg.solve(laplacian, rhs)

        energy = arap_energy(uv, faces, local, cot, rotations)
        history.append(energy)
        if len(history) > 1:
            previous = history[-2]
            if previous > 0 and abs(previous - energy) / previous < float(cfg["energy_tolerance"]):
                converged = True
                break
    return Flattening(
        uv=uv, faces=faces, local=local, method="lscm_arap", iterations=iterations,
        converged=converged, energy=history[-1] if history else 0.0,
        energy_history=history, pinned=sorted(pins),
    )


def _best_rotations(uv, faces, local, cot) -> np.ndarray:
    """(F, 2, 2) per-triangle rotation closest to the current layout."""
    out = np.empty((len(faces), 2, 2))
    for t, (v0, v1, v2) in enumerate(faces):
        cov = np.zeros((2, 2))
        for k, (i, j) in enumerate(((v1, v2), (v2, v0), (v0, v1))):
            w = cot[t, k]
            a = local[t, (k + 1) % 3] - local[t, (k + 2) % 3]
            b = uv[i] - uv[j]
            cov += w * np.outer(b, a)
        u, _s, vt = np.linalg.svd(cov)
        rot = u @ vt
        if np.linalg.det(rot) < 0:
            u[:, -1] *= -1.0
            rot = u @ vt
        out[t] = rot
    return out


def arap_energy(uv, faces, local, cot, rotations) -> float:
    total = 0.0
    for t, (v0, v1, v2) in enumerate(faces):
        rot = rotations[t]
        for k, (i, j) in enumerate(((v1, v2), (v2, v0), (v0, v1))):
            a = local[t, (k + 1) % 3] - local[t, (k + 2) % 3]
            residual = (uv[i] - uv[j]) - rot @ a
            total += cot[t, k] * float(residual @ residual)
    return total


def align_grain(uv: np.ndarray, start: int, end: int) -> np.ndarray:
    """Rotate so the grain line (start -> end) runs along +y, then move to the origin.

    A rigid motion, so no distortion measure changes. This is how the grain term is
    satisfied exactly instead of being traded off against the others.
    """
    direction = uv[end] - uv[start]
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        return uv - uv.min(axis=0)
    dx, dy = direction / norm
    # Rotation taking (dx, dy) onto (0, 1).
    rot = np.array([[dy, -dx], [dx, dy]])
    out = uv @ rot.T
    return out - out.min(axis=0)


def ensure_positive_orientation(uv: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Mirror the layout if the majority of triangles came out wound the wrong way.

    The flattening is free to produce either chirality; a panel is cut from one side of
    the cloth, so the whole layout is flipped at once rather than per triangle.
    """
    signed = triangle_areas(uv[faces])
    if float(np.sum(signed)) < 0.0:
        out = uv.copy()
        out[:, 0] *= -1.0
        return out
    return uv
