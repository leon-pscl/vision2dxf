"""Anatomical levels, perimeter solving, lofting and constraint-preserving smoothing."""
from __future__ import annotations

import math
import numpy as np
import pytest

from app.assumptions import AssumptionRegistry
from app.reconstruction.anatomical_levels import (
    AnatomicalLevel,
    LevelError,
    build_levels,
    validate_levels,
)
from app.reconstruction.cross_sections import build_sections
from app.reconstruction.landmarks import column_for_half_width, key_columns, ring_x
from app.reconstruction.lofting import MeshError, boundary_edge_count, loft, validate_mesh
from app.reconstruction.perimeter_solver import (
    SectionError,
    perimeter,
    ring,
    solve_depth,
    solve_scale,
    unit_perimeter,
)
from app.reconstruction.smoothing import smooth


# ---- perimeter and the inverse problems -----------------------------------


def test_an_ellipse_perimeter_matches_ramanujan():
    """n = 2 is an exact ellipse, so the numerical perimeter can be checked against a
    published approximation rather than against itself."""
    a, b = 150.0, 100.0
    h = ((a - b) / (a + b)) ** 2
    ramanujan = math.pi * (a + b) * (1 + 3 * h / (10 + math.sqrt(4 - 3 * h)))
    assert perimeter(a, b, 2.0) == pytest.approx(ramanujan, rel=2e-5)


def test_a_circle_perimeter_is_two_pi_r():
    assert perimeter(50.0, 50.0, 2.0) == pytest.approx(2 * math.pi * 50.0, rel=1e-6)


def test_a_high_exponent_approaches_the_bounding_rectangle():
    """As n grows the superellipse fills its box, so the perimeter tends to 8a."""
    assert perimeter(100.0, 100.0, 40.0) == pytest.approx(800.0, rel=0.02)


def test_the_perimeter_is_homogeneous_in_the_axes():
    """What makes solve_scale exact rather than iterative."""
    assert perimeter(60.0, 40.0, 2.3) == pytest.approx(2.0 * perimeter(30.0, 20.0, 2.3), rel=1e-9)


@pytest.mark.parametrize("n", [2.0, 2.3, 3.0])
def test_solve_scale_hits_the_target_circumference(n):
    a, b = solve_scale(900.0, 0.72, n)
    assert perimeter(a, b, n) == pytest.approx(900.0, abs=1e-6)
    assert b / a == pytest.approx(0.72, rel=1e-12)


@pytest.mark.parametrize("n", [2.0, 2.3, 3.0])
def test_solve_depth_hits_the_target_circumference(n):
    b, iterations = solve_depth(900.0, 200.0, n)
    assert perimeter(200.0, b, n) == pytest.approx(900.0, abs=1e-5)
    assert 0 < iterations <= 200


def test_solve_depth_rejects_an_impossible_width():
    """A section 400 mm wide has a perimeter of at least 800 mm whatever its depth."""
    with pytest.raises(SectionError) as exc:
        solve_depth(700.0, 200.0, 2.3)
    assert exc.value.errors[0]["code"] == "SECTION_IMPOSSIBLE"


def test_solve_depth_rejects_a_non_positive_width():
    with pytest.raises(SectionError):
        solve_depth(900.0, 0.0, 2.3)


def test_unit_perimeter_is_the_scale_factor():
    assert unit_perimeter(1.0, 2.0) == pytest.approx(2 * math.pi, rel=1e-6)


# ---- ring correspondence ---------------------------------------------------


def test_the_ring_starts_at_centre_front_and_turns_through_the_side():
    points = ring(100.0, 70.0, 2.3, 48)
    assert points[0] == pytest.approx([0.0, 70.0], abs=1e-9)    # centre front
    assert points[12] == pytest.approx([100.0, 0.0], abs=1e-9)  # right side
    assert points[24] == pytest.approx([0.0, -70.0], abs=1e-9)  # centre back


def test_two_rings_share_their_angular_correspondence():
    """Column j is the same angular position on every ring, which is what lets the loft
    connect them without a search."""
    a = ring(100.0, 70.0, 2.3, 48)
    b = ring(130.0, 91.0, 2.3, 48)
    with np.errstate(invalid="ignore"):
        ratio = np.divide(b, a, out=np.full_like(a, 1.3), where=np.abs(a) > 1e-9)
    assert np.allclose(ratio, 1.3, atol=1e-9)


def test_a_section_is_symmetric_about_its_width_axis():
    points = ring(100.0, 70.0, 2.3, 48)
    assert points[1, 0] == pytest.approx(-points[47, 0], abs=1e-9)
    assert points[1, 1] == pytest.approx(points[47, 1], abs=1e-9)


def test_key_columns_need_a_resolution_divisible_by_four():
    with pytest.raises(ValueError):
        key_columns(50)
    assert key_columns(48) == {"center_front": 0, "right_side": 12, "center_back": 24,
                               "left_side": 36}


def test_a_landmark_snaps_to_the_nearest_column_in_its_quarter():
    front = column_for_half_width(200.0, 140.0, 2.3, 48, 200.0, "front")
    back = column_for_half_width(200.0, 140.0, 2.3, 48, 200.0, "back")
    assert front == 12 and back == 12
    assert column_for_half_width(200.0, 140.0, 2.3, 48, 0.0, "front") == 0


# ---- anatomical levels -----------------------------------------------------


def test_levels_come_from_the_confirmed_measurements(measurements):
    levels = build_levels(measurements, AssumptionRegistry())
    assert levels["neck"].y_mm == 0.0
    assert levels["bust"].y_mm == measurements.neck_to_bust_level
    assert levels["waist"].y_mm == measurements.neck_to_waist_level
    assert levels["bust"].y_mm < levels["underbust"].y_mm < levels["waist"].y_mm


def test_the_underbust_level_is_recorded_as_an_assumption(measurements):
    registry = AssumptionRegistry()
    build_levels(measurements, registry)
    ids = {a["id"] for a in registry.as_list()}
    assert "level.underbust" in ids


def test_levels_out_of_order_are_refused_not_reordered():
    levels = {
        "upper_chest": AnatomicalLevel("upper_chest", 0.0, None, "x", 1.0),
        "bust": AnatomicalLevel("bust", 400.0, "bust_line", "x", 1.0),
        "waist": AnatomicalLevel("waist", 300.0, "waist_line", "x", 1.0),
    }
    with pytest.raises(LevelError) as exc:
        validate_levels(levels)
    assert exc.value.errors[0]["code"] == "LEVEL_OUT_OF_ORDER"
    # Untouched: the caller is asked to confirm, not corrected behind its back.
    assert levels["waist"].y_mm == 300.0


def test_levels_too_close_together_are_refused():
    levels = {
        "upper_chest": AnatomicalLevel("upper_chest", 0.0, None, "x", 1.0),
        "bust": AnatomicalLevel("bust", 300.0, "bust_line", "x", 1.0),
        "waist": AnatomicalLevel("waist", 302.0, "waist_line", "x", 1.0),
    }
    with pytest.raises(LevelError) as exc:
        validate_levels(levels)
    assert exc.value.errors[0]["code"] == "LEVELS_TOO_CLOSE"


def test_a_level_above_the_origin_is_refused():
    levels = {
        "upper_chest": AnatomicalLevel("upper_chest", 0.0, None, "x", 1.0),
        "bust": AnatomicalLevel("bust", -10.0, "bust_line", "x", 1.0),
    }
    with pytest.raises(LevelError) as exc:
        validate_levels(levels)
    assert exc.value.errors[0]["code"] == "LEVEL_ABOVE_ORIGIN"


# ---- cross-sections --------------------------------------------------------


def test_every_section_reports_its_target_and_achieved_perimeter(measurements):
    registry = AssumptionRegistry()
    levels = build_levels(measurements, registry)
    sections = build_sections(measurements, levels, "superellipse", registry)
    assert {"neck", "upper_chest", "bust", "underbust", "waist"} <= set(sections)
    for section in sections.values():
        if section.target_perimeter_mm is not None:
            assert section.achieved_perimeter_mm == pytest.approx(
                section.target_perimeter_mm, abs=0.5
            )
            assert section.residual_mm < 0.5


def test_the_waist_solves_its_depth_because_it_has_both_dimensions(measurements):
    registry = AssumptionRegistry()
    levels = build_levels(measurements, registry)
    sections = build_sections(measurements, levels, "superellipse", registry)
    waist = sections["waist"]
    assert waist.solved_for == "depth"
    assert waist.a_mm * 2 == pytest.approx(measurements.waist_back_width_full)
    assert waist.source_fields == ["waist", "waist_back_width"]


def test_a_level_with_only_a_circumference_assumes_a_ratio(measurements):
    registry = AssumptionRegistry()
    levels = build_levels(measurements, registry)
    sections = build_sections(measurements, levels, "superellipse", registry)
    assert sections["bust"].solved_for == "scale"
    assert "section.ratio.bust" in {a["id"] for a in registry.as_list()}


def test_the_ellipse_model_uses_exponent_two(measurements):
    registry = AssumptionRegistry()
    levels = build_levels(measurements, registry)
    sections = build_sections(measurements, levels, "ellipse", registry)
    assert all(s.exponent == 2.0 for s in sections.values())


def test_an_impossible_width_and_circumference_stop_the_section(measurements):
    """A waist as wide as its own circumference cannot be a closed section."""
    broken = measurements.model_copy(
        update={"waist_back_width_full": measurements.waist_circumference})
    registry = AssumptionRegistry()
    levels = build_levels(broken, registry)
    with pytest.raises(SectionError) as exc:
        build_sections(broken, levels, "superellipse", registry)
    assert any(e["code"] == "SECTION_IMPOSSIBLE" for e in exc.value.errors)


# ---- lofting ---------------------------------------------------------------


def test_the_loft_passes_through_every_section(measurements):
    registry = AssumptionRegistry()
    levels = build_levels(measurements, registry)
    sections = build_sections(measurements, levels, "superellipse", registry)
    surface = loft(sections, levels)
    for name, row in surface.level_rows.items():
        if name == "neck":
            continue
        assert surface.ys[row] == pytest.approx(sections[name].y_mm, abs=1e-9)
        assert surface.a[row] == pytest.approx(sections[name].a_mm, abs=1e-9)


def test_the_lofted_mesh_is_manifold_and_outward_facing(basic_result):
    validate_mesh(basic_result.proxy)  # raises if not
    faces = basic_result.proxy.faces()
    verts = basic_result.proxy.vertices().reshape(-1, 3)
    tri = verts[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    radial = tri.mean(axis=1).copy()
    radial[:, 1] = 0.0
    flat = normals.copy()
    flat[:, 1] = 0.0
    assert np.all(np.sum(flat * radial, axis=1) > 0)


def test_the_tube_is_open_at_the_top_and_bottom_only(basic_result):
    """An explicitly open manifold: one boundary edge per column, twice."""
    assert boundary_edge_count(basic_result.proxy) == 2 * basic_result.proxy.columns


def test_no_face_has_zero_area(basic_result):
    verts = basic_result.proxy.vertices().reshape(-1, 3)
    tri = verts[basic_result.proxy.faces()]
    areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    assert areas.min() > 1e-6


def test_a_degenerate_ring_is_refused(basic_result):
    broken = basic_result.proxy
    bad = type(broken)(broken.ys, broken.a * 0.0, broken.b, broken.exponent, broken.columns,
                       broken.level_rows)
    with pytest.raises(MeshError):
        validate_mesh(bad)


# ---- smoothing -------------------------------------------------------------


def test_smoothing_preserves_every_measured_circumference(measurements):
    registry = AssumptionRegistry()
    levels = build_levels(measurements, registry)
    sections = build_sections(measurements, levels, "superellipse", registry)
    raw = loft(sections, levels)
    before = raw.perimeters()
    smoothed, diagnostics = smooth(raw)
    after = smoothed.perimeters()
    for name, row in raw.level_rows.items():
        assert after[row] == pytest.approx(before[row], abs=1e-9), name
    assert diagnostics["maxLevelPerimeterDriftMm"] < 1e-9


def test_smoothing_actually_moves_the_rows_between_levels(measurements):
    """The point of the pass: the rows between the levels must stop being collinear."""
    registry = AssumptionRegistry()
    levels = build_levels(measurements, registry)
    sections = build_sections(measurements, levels, "superellipse", registry)
    raw = loft(sections, levels)
    smoothed, diagnostics = smooth(raw)
    assert diagnostics["maxRowChangeMm"] > 0.0
    free = [i for i in range(len(raw.ys)) if i not in set(raw.level_rows.values())]
    assert any(abs(smoothed.a[i] - raw.a[i]) > 1e-9 for i in free)


def test_smoothing_never_overshoots_a_measured_section(measurements):
    """Monotone tangents, so the smoothed loft stays inside the measured envelope."""
    registry = AssumptionRegistry()
    levels = build_levels(measurements, registry)
    sections = build_sections(measurements, levels, "superellipse", registry)
    raw = loft(sections, levels)
    smoothed, _ = smooth(raw)
    rows = sorted({r for r in raw.level_rows.values()})
    for lo, hi in zip(rows, rows[1:]):
        span = smoothed.a[lo:hi + 1]
        bounds = sorted((raw.a[lo], raw.a[hi]))
        assert bounds[0] - 1e-9 <= span.min() and span.max() <= bounds[1] + 1e-9


def test_monotone_cubic_passes_through_its_control_points():
    from app.reconstruction.smoothing import monotone_cubic

    x = np.array([0.0, 100.0, 250.0, 400.0])
    y = np.array([10.0, 30.0, 25.0, 60.0])
    assert monotone_cubic(x, y, x) == pytest.approx(y, abs=1e-9)
