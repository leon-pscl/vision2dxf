"""The bench renderer, checked against meshes whose size is known exactly.

Skipped without trimesh, which only the M4b bench needs
(``pip install -r requirements-bench.txt``).
"""

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh", reason="trimesh is a bench-only dependency")

from bench.render_silhouettes import (  # noqa: E402 - must follow the skip above
    FLOOR_MARGIN_PX,
    find_axes,
    render_body,
    render_view,
)


def _standing_box(width_m: float, depth_m: float, height_m: float) -> trimesh.Trimesh:
    """Return a box standing on the ground, as a stand-in for a body."""
    return trimesh.creation.box(extents=(width_m, depth_m, height_m))


def test_the_up_axis_is_the_tallest_one() -> None:
    axes = find_axes(_standing_box(0.5, 0.25, 1.7))
    assert axes.up == 2  # the tallest extent
    assert axes.width == 0  # wider than it is deep
    assert axes.depth == 1


def test_a_box_renders_at_one_millimetre_per_pixel() -> None:
    """1.7 m tall must come out 1700 px tall: the whole bench rests on this."""
    box = _standing_box(0.5, 0.25, 1.7)
    view = render_view(box, find_axes(box), "front")

    rows = np.flatnonzero(view.mask.any(axis=1))
    columns = np.flatnonzero(view.mask.any(axis=0))
    assert rows.max() - rows.min() + 1 == pytest.approx(1700, abs=2)
    assert columns.max() - columns.min() + 1 == pytest.approx(500, abs=2)


def test_the_side_view_shows_the_depth_not_the_width() -> None:
    box = _standing_box(0.5, 0.25, 1.7)
    axes = find_axes(box)
    side = render_view(box, axes, "side")

    columns = np.flatnonzero(side.mask.any(axis=0))
    assert columns.max() - columns.min() + 1 == pytest.approx(250, abs=2)


def test_the_feet_sit_on_the_declared_floor_row() -> None:
    box = _standing_box(0.5, 0.25, 1.7)
    view = render_view(box, find_axes(box), "front")
    lowest = int(np.flatnonzero(view.mask.any(axis=1)).max())

    assert abs(lowest - view.floor_row) <= 1
    assert view.mask.shape[0] - view.floor_row == FLOOR_MARGIN_PX


def test_both_views_share_one_floor_row(tmp_path: "object") -> None:
    """Front and side rows must mean the same height, as brief 5.4 requires."""
    box = _standing_box(0.5, 0.25, 1.7)
    path = tmp_path / "box.obj"
    box.export(path)

    views = render_body(path)
    assert views["front"].floor_row == views["side"].floor_row
    assert views["front"].mask.shape[0] == views["side"].mask.shape[0]


def test_the_shaded_render_is_not_flat() -> None:
    """A pose model needs shading; a flat silhouette gives it nothing."""
    box = _standing_box(0.5, 0.25, 1.7)
    view = render_view(box, find_axes(box), "front")
    body_pixels = view.shaded[view.mask]

    assert body_pixels.std() > 0, "the body must not be a single flat tone"
    assert view.shaded.shape[:2] == view.mask.shape


def test_the_silhouette_is_solid_not_a_wireframe() -> None:
    """Overlapping faces must union, not cancel: a box projects to a filled rectangle."""
    box = _standing_box(0.5, 0.25, 1.7)
    view = render_view(box, find_axes(box), "front")

    rows = np.flatnonzero(view.mask.any(axis=1))
    columns = np.flatnonzero(view.mask.any(axis=0))
    inside = view.mask[rows.min() : rows.max() + 1, columns.min() : columns.max() + 1]
    assert inside.mean() > 0.99


def test_a_sphere_is_solid_through_the_middle() -> None:
    """A closed curved surface is the case where face overlap is total."""
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    view = render_view(sphere, find_axes(sphere), "front")
    middle = view.mask[view.mask.any(axis=1).nonzero()[0].mean().astype(int)]

    assert middle.sum() == pytest.approx(1000, abs=10), "a 0.5 m radius sphere is 1000 px across"
    assert middle[middle.argmax() : middle.argmax() + 900].all(), "no holes through the body"
