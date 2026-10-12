"""Geometry against synthetic silhouettes and against numerical integration."""

import math

import numpy as np
import pytest
from tests.synthetic.figures import SLICES, make_figure

from bodymeasure import geometry
from bodymeasure.geometry import (
    back_arc,
    contour_length,
    ellipse_perimeter,
    hull_length,
    run_containing,
    split_depth_at,
    two_semi_ellipse_girth,
    width_at,
)
from bodymeasure.landmarks import ViewScale


def _integrated_perimeter(a: float, b: float, steps: int = 2_000_000) -> float:
    """Arc length of the ellipse by the midpoint rule, as an independent check."""
    dt = 2 * math.pi / steps
    t = (np.arange(steps) + 0.5) * dt
    return float(np.sum(np.hypot(a * np.sin(t), b * np.cos(t)) * dt))


def test_ellipse_perimeter_is_exact_for_a_circle() -> None:
    for radius in (1.0, 7.5, 420.0):
        assert ellipse_perimeter(radius, radius) == pytest.approx(2 * math.pi * radius, rel=1e-12)


def test_ellipse_perimeter_matches_numerical_integration() -> None:
    a = 100.0
    for ratio in (0.3, 0.5, 0.7, 0.9, 1.0):
        approx = ellipse_perimeter(a, a * ratio)
        assert approx == pytest.approx(_integrated_perimeter(a, a * ratio), rel=1e-4)


def test_ellipse_perimeter_rejects_negative_axes() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        ellipse_perimeter(10.0, -1.0)


def test_width_at_recovers_the_drawn_widths() -> None:
    figure = make_figure()
    tolerance = figure.mm_per_px_x  # one pixel
    for name, (w_mm, _, _) in SLICES.items():
        measured = width_at(figure.front, figure.row_of(name), figure.torso_x, figure.mm_per_px_x)
        assert measured == pytest.approx(w_mm, abs=tolerance), name


def test_split_depth_at_recovers_asymmetric_depths() -> None:
    figure = make_figure()
    tolerance = figure.mm_per_px_x
    for name, (_, d_front_mm, d_back_mm) in SLICES.items():
        front, back = split_depth_at(
            figure.side, figure.row_of(name), figure.balance_x, figure.mm_per_px_x
        )
        assert front == pytest.approx(d_front_mm, abs=tolerance), f"{name} front"
        assert back == pytest.approx(d_back_mm, abs=tolerance), f"{name} back"


def test_runs_off_the_body_are_empty() -> None:
    figure = make_figure()
    waist = figure.row_of("waist")
    assert run_containing(figure.front, waist, 5) is None
    assert width_at(figure.front, waist, 5, figure.mm_per_px_x) == 0.0
    assert width_at(figure.front, -1, figure.torso_x, figure.mm_per_px_x) == 0.0
    assert split_depth_at(figure.side, waist, 5, figure.mm_per_px_x) == (0.0, 0.0)


def test_symmetric_girth_is_one_whole_ellipse() -> None:
    girth = two_semi_ellipse_girth(width_mm=300.0, d_front_mm=100.0, d_back_mm=100.0)
    assert girth == pytest.approx(ellipse_perimeter(150.0, 100.0), rel=1e-12)


def test_asymmetric_girth_lies_between_its_two_symmetric_cases() -> None:
    shallow = ellipse_perimeter(150.0, 100.0)
    deep = ellipse_perimeter(150.0, 140.0)
    girth = two_semi_ellipse_girth(300.0, 140.0, 100.0)
    assert shallow < girth < deep
    assert back_arc(300.0, 100.0) == pytest.approx(shallow / 2, rel=1e-12)


def test_contour_length_follows_the_polyline() -> None:
    square = np.array([[0, 0], [30, 0], [30, 40]])
    assert contour_length(square, 1.0, 1.0) == pytest.approx(70.0)
    assert contour_length(square, 2.0, 1.0) == pytest.approx(60.0 + 40.0)
    assert contour_length(np.array([[1, 1]]), 1.0, 1.0) == 0.0


def test_hull_length_bridges_a_hollow_but_keeps_a_bulge() -> None:
    hollow = np.array([[10, 0], [5, 10], [10, 20]])
    bulge = np.array([[10, 0], [15, 10], [10, 20]])
    diagonal = 2 * math.hypot(5, 10)

    assert hull_length(hollow, 1.0, 1.0) == pytest.approx(20.0)
    assert hull_length(bulge, 1.0, 1.0) == pytest.approx(diagonal)
    assert contour_length(hollow, 1.0, 1.0) == pytest.approx(diagonal)
    assert hull_length(hollow, 1.0, 1.0, side="back") == pytest.approx(diagonal)


def test_a_rows_and_heights_round_trip_on_a_views_scale() -> None:
    """The scale is per view, so this round trip goes through ViewScale."""
    figure = make_figure()
    scale = ViewScale(figure.floor_row, figure.floor_row - 1000, figure.mm_per_px_y)
    for name in SLICES:
        row = figure.row_of(name)
        assert scale.row_of(scale.height_mm(row)) == row


def test_each_ellipse_equation_is_exact_for_a_circle() -> None:
    """Brief 19.1: the circle case, for each formula."""
    for radius in (1.0, 5.0, 123.456):
        expected = 2 * math.pi * radius
        assert geometry.ramanujan_i(radius, radius) == pytest.approx(expected)
        assert geometry.ramanujan_ii(radius, radius) == pytest.approx(expected)
        value, formula = geometry.ellipse_perimeter_montazerian(radius, radius)
        assert value == pytest.approx(expected)
        assert formula == geometry.ELLIPSE_EQUATION_1


def _exact_perimeter(a: float, b: float, steps: int = 400_000) -> float:
    """Return the ellipse perimeter by numerical integration, as the check value."""
    theta = np.linspace(0.0, 2 * math.pi, steps)
    dx = -a * np.sin(theta)
    dy = b * np.cos(theta)
    return float(np.trapezoid(np.hypot(dx, dy), theta))


@pytest.mark.parametrize(
    ("a", "b"),
    [(50.0, 40.0), (143.2, 110.9), (90.0, 30.0), (100.0, 20.0), (120.0, 8.0)],
)
def test_the_montazerian_rule_stays_close_to_the_true_perimeter(a: float, b: float) -> None:
    """Brief 19.1: compare the chosen equation with numerical integration."""
    value, formula = geometry.ellipse_perimeter_montazerian(a, b)
    exact = _exact_perimeter(a, b)

    assert value == pytest.approx(exact, rel=0.005), formula
    assert formula == (
        geometry.ELLIPSE_EQUATION_1 if a <= 3 * b else geometry.ELLIPSE_EQUATION_2
    )


def test_the_elongated_equation_is_chosen_where_the_first_one_drifts() -> None:
    """The whole point of the two-equation rule: a flat section needs the second."""
    a, b = 120.0, 8.0  # a > 3b
    exact = _exact_perimeter(a, b)

    assert abs(geometry.ramanujan_i(a, b) - exact) > abs(geometry.ramanujan_ii(a, b) - exact)
    assert geometry.ellipse_perimeter_montazerian(a, b)[1] == geometry.ELLIPSE_EQUATION_2


def test_the_rule_reads_the_axes_in_either_order() -> None:
    assert geometry.ellipse_perimeter_montazerian(8.0, 120.0) == (
        geometry.ellipse_perimeter_montazerian(120.0, 8.0)
    )


def test_a_girth_uses_the_breadth_and_the_depth_as_the_two_axes() -> None:
    """Brief 12.1: semi-axes are max/2 and min/2 of the measured breadth and depth."""
    width, depth = 286.4, 221.8
    value, formula = geometry.montazerian_girth(width, depth)

    assert (value, formula) == geometry.ellipse_perimeter_montazerian(width / 2, depth / 2)
    assert value == pytest.approx(_exact_perimeter(width / 2, depth / 2), rel=0.005)
    # A circular section: breadth and depth equal, so the girth is pi x diameter.
    assert geometry.montazerian_girth(100.0, 100.0)[0] == pytest.approx(math.pi * 100.0)


def test_a_negative_semi_axis_is_refused_not_absorbed() -> None:
    for function in (geometry.ramanujan_i, geometry.ramanujan_ii):
        with pytest.raises(ValueError, match="non-negative"):
            function(-1.0, 1.0)


def test_the_brief_12_helpers_measure_what_they_say() -> None:
    points = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]])

    assert geometry.euclidean_length(points, 2.0) == pytest.approx(10.0), "end to end, not along"
    assert geometry.polyline_length(points) == pytest.approx(7.0), "along the path"
    assert geometry.euclidean_length(points[:1], 2.0) == 0.0
    assert geometry.convex_hull_path_length(points) <= geometry.polyline_length(points)


def test_an_angle_is_degrees_at_the_vertex() -> None:
    assert geometry.angle_deg((1.0, 0.0), (0.0, 0.0), (0.0, 1.0)) == pytest.approx(90.0)
    assert geometry.angle_deg((1.0, 0.0), (0.0, 0.0), (1.0, 0.0)) == pytest.approx(0.0)
    assert geometry.angle_deg((-1.0, 0.0), (0.0, 0.0), (1.0, 0.0)) == pytest.approx(180.0)
    assert geometry.angle_deg((0.0, 0.0), (0.0, 0.0), (1.0, 0.0)) == 0.0, "undefined, not invented"
