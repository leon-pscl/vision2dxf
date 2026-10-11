"""Geometry against synthetic silhouettes and against numerical integration."""

import math

import numpy as np
import pytest
from tests.synthetic.figures import SLICES, make_figure

from bodymeasure.geometry import (
    back_arc,
    contour_length,
    ellipse_perimeter,
    height_of_row,
    hull_length,
    row_for_height,
    run_containing,
    split_depth_at,
    two_semi_ellipse_girth,
    width_at,
)


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


def test_rows_and_heights_round_trip() -> None:
    figure = make_figure()
    for name in SLICES:
        row = figure.row_of(name)
        h_mm = height_of_row(row, figure.floor_row, figure.mm_per_px_y)
        assert row_for_height(h_mm, figure.floor_row, figure.mm_per_px_y) == row
