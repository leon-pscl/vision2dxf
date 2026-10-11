"""Render front and side silhouettes from a body mesh (brief M4b).

Orthographic projection at a fixed millimetres-per-pixel, feet on a fixed floor
row, so the rendering carries the same calibration contract the camera does.

There is no GL renderer here on purpose: a headless station should not need one.
Every triangle is projected and filled, and their union is the silhouette, which
is exact for a closed mesh.

Run directly to write PNGs for inspection::

    PYTHONPATH=src python bench/render_silhouettes.py vendor/GarmentCode/assets/bodies/mean_all.obj
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import trimesh

MM_PER_PX = 1.0
FLOOR_MARGIN_PX = 60  # blank rows below the feet, so the mask is inside the frame
HEAD_MARGIN_PX = 80
GREY_BACKGROUND = 235
LIGHT = np.array([0.3, 0.6, 0.75])  # a light from the front, slightly above


@dataclass(frozen=True)
class Axes:
    """Which mesh axis is up, which faces the camera, and which is sideways."""

    up: int
    depth: int
    width: int


@dataclass(frozen=True)
class View:
    """One rendered view: its silhouette and a shaded image for the pose model."""

    mask: np.ndarray
    shaded: np.ndarray
    floor_row: int


def load_body(path: Path) -> trimesh.Trimesh:
    """Load a body mesh, concatenating it if the file holds a scene.

    Raises:
        ValueError: if the file holds no geometry.

    """
    loaded = trimesh.load(path, process=False, force="mesh")
    if not isinstance(loaded, trimesh.Trimesh) or loaded.is_empty:
        raise ValueError(f"no mesh geometry in {path}")
    return loaded


def find_axes(mesh: trimesh.Trimesh) -> Axes:
    """Work out the mesh orientation from its bounding box.

    A standing human is tallest along the up axis and widest across the
    shoulders, which leaves the shallowest axis pointing at the camera. This
    avoids hard-coding an export convention that may differ per asset.
    """
    extents = mesh.extents
    up = int(np.argmax(extents))
    remaining = [axis for axis in range(3) if axis != up]
    depth = min(remaining, key=lambda axis: extents[axis])
    width = next(axis for axis in remaining if axis != depth)
    return Axes(up=up, depth=depth, width=width)


def _project(
    mesh: trimesh.Trimesh,
    horizontal: int,
    up: int,
    mm_per_px: float,
    image_size: tuple[int, int],
    flip: bool,
) -> np.ndarray:
    """Return every vertex in pixel coordinates for one orthographic view."""
    height, width = image_size
    vertices = mesh.vertices * 1000.0 / mm_per_px  # metres to pixels
    across = vertices[:, horizontal] * (-1.0 if flip else 1.0)
    columns = across - across.min()
    columns = columns + (width - columns.max()) / 2  # centre the body
    rows = (height - FLOOR_MARGIN_PX) - (vertices[:, up] - vertices[:, up].min())
    return np.stack([columns, rows], axis=1)


def _fill(points: np.ndarray, faces: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    """Rasterise every triangle; their union is the silhouette.

    One call per triangle, deliberately: passing them all to ``fillPoly`` applies
    the even-odd rule across contours, so a front face and the back face behind
    it cancel out and leave a hollow wireframe.
    """
    canvas = np.zeros(image_size, dtype=np.uint8)
    for triangle in points[faces].round().astype(np.int32):
        cv2.fillConvexPoly(canvas, triangle, 1)
    return canvas.astype(bool)


def _shade(
    mesh: trimesh.Trimesh,
    points: np.ndarray,
    image_size: tuple[int, int],
    flip: bool,
) -> np.ndarray:
    """Return a diffusely shaded grey rendering, back faces first.

    A flat silhouette gives a pose model nothing to work with, so faces are
    shaded by their normals and drawn far-to-near, which is enough of a painter
    algorithm for a convex-ish body.
    """
    height, width = image_size
    canvas = np.full((height, width, 3), GREY_BACKGROUND, dtype=np.uint8)
    normals = mesh.face_normals.copy()
    if flip:
        normals[:, 0] *= -1
    shade = np.clip(normals @ (LIGHT / np.linalg.norm(LIGHT)), 0.0, 1.0)
    tone = (60 + 150 * shade).astype(np.uint8)

    depth = mesh.triangles_center[:, 0]
    order = np.argsort(depth)
    triangles = points[mesh.faces].round().astype(np.int32)
    for index in order:
        value = int(tone[index])
        cv2.fillConvexPoly(canvas, triangles[index], (value, value, value))
    return canvas


def render_view(
    mesh: trimesh.Trimesh,
    axes: Axes,
    which: str,
    mm_per_px: float = MM_PER_PX,
) -> View:
    """Render the front or the (body-faces-right) side view."""
    height_px = int(mesh.extents[axes.up] * 1000.0 / mm_per_px) + FLOOR_MARGIN_PX + HEAD_MARGIN_PX
    width_px = int(mesh.extents[axes.width] * 1000.0 / mm_per_px) + 2 * HEAD_MARGIN_PX
    image_size = (height_px, max(width_px, 400))

    horizontal = axes.width if which == "front" else axes.depth
    flip = which != "front"  # so the body faces +x, as brief 5.4 requires
    points = _project(mesh, horizontal, axes.up, mm_per_px, image_size, flip)
    return View(
        mask=_fill(points, mesh.faces, image_size),
        shaded=_shade(mesh, points, image_size, flip),
        floor_row=image_size[0] - FLOOR_MARGIN_PX,
    )


def render_body(path: Path, mm_per_px: float = MM_PER_PX) -> dict[str, View]:
    """Render the front and side views of one body mesh."""
    mesh = load_body(path)
    axes = find_axes(mesh)
    return {which: render_view(mesh, axes, which, mm_per_px) for which in ("front", "side")}


def main(argv: list[str]) -> int:
    """Write the two views of each mesh given on the command line to ``bench/out``."""
    if not argv:
        print(__doc__)
        return 1
    out = Path(__file__).resolve().parent / "out"
    out.mkdir(parents=True, exist_ok=True)
    for name in argv:
        path = Path(name)
        views = render_body(path)
        for which, view in views.items():
            cv2.imwrite(
                str(out / f"{path.stem}_{which}_mask.png"), view.mask.astype(np.uint8) * 255
            )
            cv2.imwrite(str(out / f"{path.stem}_{which}.png"), view.shaded)
            print(
                f"{path.stem} {which}: {view.mask.shape[1]}x{view.mask.shape[0]} px, "
                f"floor row {view.floor_row}, {int(view.mask.sum())} body pixels"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
