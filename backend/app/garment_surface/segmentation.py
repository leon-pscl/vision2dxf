"""Panel segmentation.

The seam network cuts the garment surface into two patches: the right front quarter
(centre front to the side) and the right back quarter (side to centre back). Centre front
and centre back are folds, so the left half is the mirror of what is flattened.

Each patch is sampled as a structured grid: one column of the ring per grid column, and
`panel_rows` samples down each column from the trim curve to the hem. A structured grid is
a topological disk by construction, which is what the parameterisation needs, and its
boundary loop comes out already ordered, which is what the boundary refit needs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import reconstruction_config
from ..reconstruction.lofting import RingSurface
from .seams import SeamNetwork


class SegmentationError(Exception):
    def __init__(self, errors: list[dict]) -> None:
        self.errors = errors
        super().__init__("; ".join(e["message"] for e in errors))


@dataclass
class SurfacePanel:
    id: str
    label: str
    vertices_3d: np.ndarray
    faces: np.ndarray
    boundary_loops: list[list[int]]
    semantic_boundaries: dict[str, list[int]]
    # Grid provenance, kept so the flattened panel can be mapped back to the surface.
    key: str
    side: str
    rows: int
    cols: int
    columns: list[int] = field(default_factory=list)
    # Seam roles in boundary walk order, so the refit never has to re-derive it.
    boundary_order: list[str] = field(default_factory=list)

    def vertex(self, row: int, col: int) -> int:
        return row * self.cols + col

    def boundary(self) -> list[int]:
        return self.boundary_loops[0]

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "key": self.key,
            "label": self.label,
            "side": self.side,
            "rows": self.rows,
            "cols": self.cols,
            "columns": self.columns,
            "vertexCount": int(len(self.vertices_3d)),
            "faceCount": int(len(self.faces)),
            "boundaryLength": len(self.boundary()),
            "semanticBoundaries": {
                role: len(ids) for role, ids in self.semantic_boundaries.items()
            },
        }

    def obj(self) -> bytes:
        lines = [f"# panel {self.key}, millimetres"]
        lines += [f"v {x:.4f} {y:.4f} {z:.4f}" for x, y, z in self.vertices_3d]
        lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in self.faces]
        return ("\n".join(lines) + "\n").encode("utf-8")


PANELS = (
    # key, label, panel side used by the trim curve, first column, last column
    ("front", "Front (cut on the centre-front fold)", "front", "center_front", "side"),
    ("back", "Back (cut on the centre-back fold)", "back", "side", "center_back"),
)


def segment(surface: RingSurface, network: SeamNetwork,
            panel_rows: int | None = None) -> list[SurfacePanel]:
    rows = int(panel_rows or reconstruction_config()["mesh"]["panel_rows"]) + 1
    bounds = {"center_front": 0, "side": network.side_column, "center_back": network.back_column}
    panels = []
    for key, label, side, first, last in PANELS:
        panels.append(_panel(surface, network, key, label, side,
                             bounds[first], bounds[last], rows))
    return panels


def _panel(
    surface: RingSurface, network: SeamNetwork, key: str, label: str, side: str,
    first_column: int, last_column: int, rows: int,
) -> SurfacePanel:
    columns = list(range(first_column, last_column + 1))
    cols = len(columns)
    errors: list[dict] = []
    if cols < 3 or rows < 3:
        errors.append(_err(key, "PANEL_GRID_TOO_COARSE",
                           f"The {key} panel grid is {cols} x {rows}; at least 3 x 3 is "
                           f"needed. Raise the mesh resolution."))
        raise SegmentationError(errors)

    verts = np.empty((rows * cols, 3))
    for c, column in enumerate(columns):
        top = network.trim_y(side, column)
        bottom = network.waist_y_mm
        if not top < bottom:
            errors.append(_err(key, "PANEL_COLUMN_INVERTED",
                               f"Column {column} of the {key} panel is cut at "
                               f"{top:.1f} mm, at or below the hem at {bottom:.1f} mm."))
            continue
        for r in range(rows):
            y = top + (bottom - top) * r / (rows - 1)
            verts[r * cols + c] = surface.point(y, column)
    if errors:
        raise SegmentationError(errors)

    faces = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            v00, v01 = r * cols + c, r * cols + c + 1
            v10, v11 = (r + 1) * cols + c, (r + 1) * cols + c + 1
            faces.append((v00, v10, v11))
            faces.append((v00, v11, v01))
    faces_arr = np.array(faces, dtype=int)

    loop = (
        [c for c in range(cols)]                                   # trim curve, left->right
        + [r * cols + (cols - 1) for r in range(1, rows)]           # right column, down
        + [(rows - 1) * cols + c for c in range(cols - 2, -1, -1)]  # hem, right->left
        + [r * cols for r in range(rows - 2, 0, -1)]                # left column, up
    )

    semantic, order = _semantic_boundaries(network, key, columns, rows, cols)
    panel = SurfacePanel(
        id="", label=label, vertices_3d=verts, faces=faces_arr, boundary_loops=[loop],
        semantic_boundaries=semantic, key=key, side=side, rows=rows, cols=cols,
        columns=columns, boundary_order=order,
    )
    validate_panel(panel)
    return panel


def _semantic_boundaries(
    network: SeamNetwork, key: str, columns: list[int], rows: int, cols: int
) -> tuple[dict[str, list[int]], list[str]]:
    """Boundary vertices per seam role, plus the roles in walk order around the loop."""
    out: dict[str, list[int]] = {}
    order: list[str] = []

    # The trim curve, split at the neck point and the shoulder tip. Row 0 of the grid is
    # the trim curve, so a grid column index is already the vertex index. The two corner
    # columns are named outright, mirrored for the back, rather than found by scanning for
    # a role change: a scan places the shared corner on whichever side it walks from,
    # which would make the front and back shoulder seams differ by one column.
    first = columns[0]
    neck, tip = network.neck_column, network.tip_column
    side, back = network.side_column, network.back_column
    if key == "front":
        intervals = [("neckline", 0, neck), ("shoulder", neck, tip), ("armscye", tip, side)]
    else:
        intervals = [("armscye", side, back - tip), ("shoulder", back - tip, back - neck),
                     ("neckline", back - neck, back)]
    for role, start, end in intervals:
        out[role] = list(range(start - first, end - first + 1))
        order.append(role)

    left_role = "center_front" if key == "front" else "side_seam"
    right_role = "side_seam" if key == "front" else "center_back"
    out[right_role] = [r * cols + (cols - 1) for r in range(rows)]
    out["waist"] = [(rows - 1) * cols + c for c in range(cols - 1, -1, -1)]
    out[left_role] = [r * cols for r in range(rows - 1, -1, -1)]
    order += [right_role, "waist", left_role]
    return out, order


def validate_panel(panel: SurfacePanel) -> None:
    """Disk topology, manifold edges, one boundary loop, no degenerate face."""
    errors: list[dict] = []
    verts, faces = panel.vertices_3d, panel.faces

    if not np.all(np.isfinite(verts)):
        errors.append(_err(panel.key, "NON_FINITE_VERTEX",
                           f"The {panel.key} panel carries a non-finite vertex."))

    tri = verts[faces]
    areas = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
    )
    if np.any(areas <= 1e-9):
        errors.append(_err(panel.key, "ZERO_AREA_FACE",
                           f"{int(np.sum(areas <= 1e-9))} face(s) of the {panel.key} panel "
                           f"have no area."))

    counts: dict[tuple[int, int], int] = {}
    for a, b, c in faces:
        for edge in ((a, b), (b, c), (c, a)):
            k = (min(edge), max(edge))
            counts[k] = counts.get(k, 0) + 1
    if any(v > 2 for v in counts.values()):
        errors.append(_err(panel.key, "NON_MANIFOLD_PANEL",
                           f"The {panel.key} panel has a non-manifold edge."))

    boundary = [k for k, v in counts.items() if v == 1]
    if len(boundary) != len(panel.boundary()):
        errors.append(_err(
            panel.key, "BOUNDARY_NOT_A_SINGLE_LOOP",
            f"The {panel.key} panel has {len(boundary)} boundary edges but a walked loop "
            f"of {len(panel.boundary())}; the patch is not a disk.",
        ))

    # Connectivity: every vertex must be reachable from vertex 0 through the faces.
    adjacency: dict[int, set[int]] = {}
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            adjacency.setdefault(u, set()).add(v)
            adjacency.setdefault(v, set()).add(u)
    seen, stack = {0}, [0]
    while stack:
        for nxt in adjacency.get(stack.pop(), ()):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    if len(seen) != len(verts):
        errors.append(_err(panel.key, "PANEL_DISCONNECTED",
                           f"The {panel.key} panel splits into more than one piece."))

    for role, ids in panel.semantic_boundaries.items():
        if len(ids) < 2:
            errors.append(_err(panel.key, "SEAM_INTERVAL_TOO_SHORT",
                               f"The {role} boundary of the {panel.key} panel has "
                               f"{len(ids)} vertex; a seam needs at least two. Raise the "
                               f"cross-section resolution."))
    if errors:
        raise SegmentationError(errors)


def _err(field: str, code: str, message: str) -> dict:
    return {
        "field": field, "code": code, "message": " ".join(message.split()),
        "suggestedAction": "Raise the mesh resolution or re-check the seam network.",
        "severity": "blocking",
    }
