"""Cutting lines: offset distances, joins, fold edges, closure."""
from __future__ import annotations

import pytest

from app.config import drafting_config
from app.drafting.cutting import build_cutting_lines, flatten_boundary
from app.drafting.curves import arc_points, offset_polylines, signed_area
from app.drafting.ease import Settings
from app.drafting.geometry import Point2D
from app.drafting.pipeline import draft

TOL = 1e-6


def _square() -> list[list[Point2D]]:
    return [[Point2D(0, 0), Point2D(10, 0)],
            [Point2D(10, 0), Point2D(10, 10)],
            [Point2D(10, 10), Point2D(0, 10)],
            [Point2D(0, 10), Point2D(0, 0)]]


def test_uniform_offset_grows_the_contour():
    out = offset_polylines(_square(), [2.0] * 4)
    assert len(out) == 4
    xs, ys = [p.x for p in out], [p.y for p in out]
    assert (min(xs), min(ys), max(xs), max(ys)) == (-2.0, -2.0, 12.0, 12.0)


def test_offset_moves_outwards_whatever_the_winding():
    forward = _square()
    backward = [list(reversed(line)) for line in reversed(forward)]
    a = {(p.x, p.y) for p in offset_polylines(forward, [2.0] * 4)}
    b = {(p.x, p.y) for p in offset_polylines(backward, [2.0] * 4)}
    assert a == b


def test_zero_distance_keeps_a_fold_edge_in_place():
    out = offset_polylines(_square(), [0.0, 2.0, 2.0, 0.0])
    assert Point2D(0, 0) in out and Point2D(10, 0) in out  # fold edges untouched
    xs, ys = [p.x for p in out], [p.y for p in out]
    assert (min(xs), min(ys)) == (0.0, 0.0)
    assert (max(xs), max(ys)) == (12.0, 12.0)


@pytest.mark.parametrize("join", ["miter", "bevel", "round"])
def test_every_join_style_produces_a_closed_contour(join, basic_body):
    result = draft(basic_body, Settings(cutting_join=join))
    for key, line in result.cutting.items():
        assert line.join == join
        assert len(line.contour) >= 3
        assert all(isinstance(p, Point2D) for p in line.contour)


def test_join_styles_differ_at_a_corner():
    miter = offset_polylines(_square(), [5.0] * 4, join="miter")
    bevel = offset_polylines(_square(), [5.0] * 4, join="bevel")
    round_ = offset_polylines(_square(), [5.0] * 4, join="round")
    assert len(round_) > len(miter)
    assert miter != bevel


def test_unknown_join_is_rejected():
    with pytest.raises(ValueError, match="unknown join style"):
        offset_polylines(_square(), [1.0] * 4, join="zigzag")


def test_arc_points_fall_back_to_a_straight_join():
    assert arc_points(Point2D(0, 0), Point2D(1, 1), Point2D(0, 0), 4) == [Point2D(1, 1)]


def test_cutting_line_sits_outside_the_sewing_line(basic_result):
    front = next(p for p in basic_result.document.panels if p.key == "front")
    cutting = basic_result.cutting["front"]
    sewing = [p for line in flatten_boundary(front) for p in line]
    assert signed_area(cutting.contour) * signed_area(sewing) > 0  # same winding, larger area
    assert abs(signed_area(cutting.contour)) > abs(signed_area(sewing))
    assert min(p.x for p in cutting.contour) < min(p.x for p in sewing)
    assert max(p.x for p in cutting.contour) > max(p.x for p in sewing)


def test_centre_edge_keeps_zero_allowance(basic_result):
    front = next(p for p in basic_result.document.panels if p.key == "front")
    assert front.edge_by_role("center_front").seam_allowance_mm == 0.0
    assert front.edge_by_role("waist").seam_allowance_mm == \
        drafting_config()["seam_allowance_mm"]["waist"]


def test_dart_cutting_contours_exist(basic_result):
    front = next(p for p in basic_result.document.panels if p.key == "front")
    assert len(front.darts) == 1
    assert len(basic_result.cutting["front"].dart_contours) == 2


def test_seam_allowance_override_is_applied(basic_body):
    result = draft(basic_body, Settings(seam_allowances={"armscye": 25.0}))
    front = next(p for p in result.document.panels if p.key == "front")
    assert front.edge_by_role("armscye").seam_allowance_mm == 25.0


def test_cut_line_validation_check_runs(basic_result):
    check = next(c for c in basic_result.report.checks if c.id == "cutting_lines")
    assert check.status == "pass"


def test_cutting_lines_are_rebuilt_deterministically(basic_result):
    again = build_cutting_lines(basic_result.document, basic_result.settings)
    for key, line in basic_result.cutting.items():
        assert [p.as_triple() for p in line.contour] == [p.as_triple() for p in again[key].contour]
