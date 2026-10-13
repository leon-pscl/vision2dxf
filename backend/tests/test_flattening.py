"""Flattening: controlled geometry fixtures, convergence, distortion, refit, balancing.

The three analytic fixtures are the only cases where the right answer is known in advance:
a plane develops exactly, a cylinder patch develops into a rectangle, and a truncated cone
patch develops into an annular sector. They are the calibration for every number the
diagnostics report on a torso panel, where no closed form exists.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from app.flattening.boundary_refit import _max_distance, fit_cubic, simplify
from app.flattening.distortion import measure
from app.flattening.parameterization import align_grain, arap, ensure_positive_orientation, lscm
from app.flattening.triangulation import local_frames, mesh_quality, triangle_areas
from conftest import generate_profile


# ---- controlled geometry ---------------------------------------------------


def _grid_mesh(point, rows: int, cols: int):
    """Vertices and triangles of a structured (rows x cols) patch of `point(u, v)`."""
    verts = np.array([point(r / (rows - 1), c / (cols - 1))
                      for r in range(rows) for c in range(cols)])
    faces = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            v00, v01 = r * cols + c, r * cols + c + 1
            v10, v11 = (r + 1) * cols + c, (r + 1) * cols + c + 1
            faces.append((v00, v10, v11))
            faces.append((v00, v11, v01))
    return verts, np.array(faces, dtype=int)


def _flatten(verts, faces):
    uv = lscm(verts, faces, 0, len(verts) - 1)
    out = arap(verts, faces, uv, {0: tuple(uv[0])}, {0: 1.0})
    flat = ensure_positive_orientation(out.uv, faces)
    return flat, out


def _boundary(rows: int, cols: int) -> list[int]:
    return (
        [c for c in range(cols)]
        + [r * cols + (cols - 1) for r in range(1, rows)]
        + [(rows - 1) * cols + c for c in range(cols - 2, -1, -1)]
        + [r * cols for r in range(rows - 2, 0, -1)]
    )


PLANE_ROWS, PLANE_COLS = 7, 9


def test_a_plane_flattens_with_no_distortion():
    verts, faces = _grid_mesh(lambda u, v: (200.0 * v, 150.0 * u, 0.0), PLANE_ROWS, PLANE_COLS)
    flat, solver = _flatten(verts, faces)
    d = measure(flat, faces, solver.local, _boundary(PLANE_ROWS, PLANE_COLS))
    assert d["maxAngularDistortionDeg"] < 1e-4
    assert d["maxAreaDistortion"] < 1e-9
    assert d["maxEdgeLengthDistortion"] < 1e-9
    assert d["flippedTriangles"] == 0


def test_a_cylinder_patch_develops_into_a_rectangle():
    """A cylinder is developable, so the patch must come out as its unrolled rectangle:
    the arc length around becomes the width, the height stays the height."""
    radius, height, sweep = 120.0, 240.0, math.radians(100.0)
    verts, faces = _grid_mesh(
        lambda u, v: (radius * math.sin(sweep * v), height * u, radius * math.cos(sweep * v)),
        9, 13,
    )
    flat, solver = _flatten(verts, faces)
    d = measure(flat, faces, solver.local, _boundary(9, 13))
    assert d["maxAngularDistortionDeg"] < 0.05
    assert d["maxEdgeLengthDistortion"] < 2e-3
    assert d["flippedTriangles"] == 0
    # The development is a rectangle: measure its sides along the boundary, which does not
    # depend on how the layout happens to be rotated.
    rows, cols = 9, 13
    around = flat[[c for c in range(cols)]]
    along = flat[[r * cols for r in range(rows)]]
    assert _length(around) == pytest.approx(radius * sweep, rel=5e-3)
    assert _length(along) == pytest.approx(height, rel=5e-3)
    # Opposite sides equal, and the corner angle is square.
    far = flat[[(rows - 1) * cols + c for c in range(cols)]]
    assert _length(far) == pytest.approx(_length(around), rel=5e-3)
    corner = flat[cols - 1] - flat[0]
    side = flat[(rows - 1) * cols] - flat[0]
    cosine = float(corner @ side) / (np.linalg.norm(corner) * np.linalg.norm(side))
    assert abs(cosine) < 0.02


def test_a_truncated_cone_patch_develops_into_a_sector():
    """A cone is developable too, and its development is an annular sector: the two
    straight edges converge, so the flattened patch is wider at one end."""
    r0, r1, height, sweep = 90.0, 150.0, 200.0, math.radians(120.0)

    def point(u, v):
        r = r0 + (r1 - r0) * u
        return (r * math.sin(sweep * v), height * u, r * math.cos(sweep * v))

    rows, cols = 11, 15
    verts, faces = _grid_mesh(point, rows, cols)
    flat, solver = _flatten(verts, faces)
    d = measure(flat, faces, solver.local, _boundary(rows, cols))
    assert d["maxAngularDistortionDeg"] < 0.2
    assert d["maxEdgeLengthDistortion"] < 5e-3
    assert d["flippedTriangles"] == 0
    # The narrow end of the sector really is narrower than the wide end.
    top = flat[[c for c in range(cols)]]
    bottom = flat[[(rows - 1) * cols + c for c in range(cols)]]
    assert _length(top) == pytest.approx(r0 * sweep, rel=1e-2)
    assert _length(bottom) == pytest.approx(r1 * sweep, rel=1e-2)


def _length(points: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def test_a_sphere_patch_cannot_develop_and_says_so():
    """The negative control: a sphere is not developable, so the diagnostics must report
    real distortion rather than near zero."""
    radius = 150.0
    verts, faces = _grid_mesh(
        lambda u, v: (radius * math.sin(1.0 * u) * math.sin(1.0 * v),
                      radius * math.cos(1.0 * u),
                      radius * math.sin(1.0 * u) * math.cos(1.0 * v)),
        9, 9,
    )
    flat, solver = _flatten(verts, faces)
    d = measure(flat, faces, solver.local, _boundary(9, 9))
    assert d["maxEdgeLengthDistortion"] > 0.01


# ---- the solver on real panels --------------------------------------------


def test_every_panel_converges(basic_result):
    for p in basic_result.panels:
        assert p.flattening.converged, p.panel.key
        assert p.flattening.method == "lscm_arap"
        assert p.flattening.energy_history[-1] <= p.flattening.energy_history[0]


def test_no_panel_contains_a_flipped_triangle(generated):
    for name, result in generated.items():
        for p in result.panels:
            assert p.distortion["flippedTriangles"] == 0, (name, p.panel.key)


def test_a_flipped_triangle_is_detected():
    """The detector itself, on a layout with one triangle deliberately turned over."""
    verts, faces = _grid_mesh(lambda u, v: (100.0 * v, 100.0 * u, 0.0), 4, 4)
    local = local_frames(verts, faces)
    uv = verts[:, :2].copy()
    uv[5] = uv[5] + np.array([0.0, 300.0])  # drag one interior vertex across its ring
    d = measure(uv, faces, local, _boundary(4, 4))
    assert d["flippedTriangles"] > 0


def test_the_boundary_does_not_cross_itself(generated):
    for name, result in generated.items():
        for p in result.panels:
            assert p.distortion["boundarySelfIntersections"] == 0, (name, p.panel.key)


def test_a_self_intersecting_boundary_is_detected():
    verts, faces = _grid_mesh(lambda u, v: (100.0 * v, 100.0 * u, 0.0), 4, 4)
    local = local_frames(verts, faces)
    uv = verts[:, :2].copy()
    uv[0] = np.array([150.0, 150.0])  # pull a corner through the middle
    d = measure(uv, faces, local, _boundary(4, 4))
    assert d["boundarySelfIntersections"] == 1


def test_distortion_is_reported_for_every_panel(basic_result):
    for p in basic_result.panels:
        for key in ("meanAngularDistortionDeg", "maxAngularDistortionDeg",
                    "meanAreaDistortion", "maxAreaDistortion",
                    "meanEdgeLengthDistortion", "maxEdgeLengthDistortion"):
            assert p.distortion[key] >= 0.0
        assert p.distortion["meanAngularDistortionDeg"] <= p.distortion["maxAngularDistortionDeg"]
        assert len(p.distortion["perTriangleAngularDeg"]) == len(p.panel.faces)


def test_a_torso_panel_distorts_a_little_but_not_much(basic_result):
    """A torso is not developable, so some distortion is expected; a mean angle error of
    several degrees would mean the flattening had gone wrong."""
    for p in basic_result.panels:
        assert 0.0 < p.distortion["meanAngularDistortionDeg"] < 3.0
        assert p.distortion["maxEdgeLengthDistortion"] < 0.2


def test_the_grain_runs_along_the_fold(basic_result):
    """The grain is realised as a rigid rotation, so the fold edge ends up vertical."""
    for p in basic_result.panels:
        fold = "center_front" if p.panel.key == "front" else "center_back"
        edge = p.pattern_panel.edge_by_role(fold)
        assert abs(edge.end.x - edge.start.x) < 0.05 * abs(edge.end.y - edge.start.y)


def test_rigid_alignment_changes_no_distortion():
    verts, faces = _grid_mesh(lambda u, v: (90.0 * v, 70.0 * u, 20.0 * u * v), 6, 7)
    flat, solver = _flatten(verts, faces)
    before = measure(flat, faces, solver.local, _boundary(6, 7))
    after = measure(align_grain(flat, 0, len(verts) - 1), faces, solver.local,
                    _boundary(6, 7))
    assert after["maxAngularDistortionDeg"] == pytest.approx(
        before["maxAngularDistortionDeg"], abs=1e-9)
    assert after["flattenedAreaMm2"] == pytest.approx(before["flattenedAreaMm2"], abs=1e-6)


def test_mesh_quality_is_reported(basic_result):
    for p in basic_result.panels:
        assert p.quality["degenerateTriangles"] == 0
        assert p.quality["minAngleDeg"] > 1.0
        assert p.quality["triangles"] == len(p.panel.faces)


def test_an_unknown_flattening_method_is_refused():
    from app.pipeline import GenerationBlocked

    with pytest.raises(GenerationBlocked) as exc:
        generate_profile("body_measurements.yaml", flattening_method="neural_v2")
    assert exc.value.code == "unknown_flattening_method"


# ---- boundary extraction, simplification and curve fitting -----------------


def test_simplify_keeps_the_endpoints_and_respects_the_tolerance():
    t = np.linspace(0, 1, 60)
    points = np.column_stack([100.0 * t, 20.0 * np.sin(math.pi * t)])
    out = simplify(points, 0.5, 4)
    assert len(out) < len(points)
    assert np.allclose(out[0], points[0]) and np.allclose(out[-1], points[-1])
    # Every dropped point stays within the tolerance of the kept polyline.
    assert _max_distance(points, out) <= 0.5 + 1e-9


def test_simplify_leaves_a_straight_line_as_two_points():
    points = np.column_stack([np.linspace(0, 100, 40), np.zeros(40)])
    assert len(simplify(points, 0.1, 2)) == 2


def test_a_cubic_fit_follows_the_curve_it_was_sampled_from():
    control = np.array([[0.0, 0.0], [30.0, 80.0], [120.0, 60.0], [150.0, 0.0]])
    t = np.linspace(0, 1, 50)[:, None]
    curve = ((1 - t) ** 3 * control[0] + 3 * (1 - t) ** 2 * t * control[1]
             + 3 * (1 - t) * t ** 2 * control[2] + t ** 3 * control[3])
    fitted = fit_cubic(curve)
    assert np.allclose(fitted[0], control[0]) and np.allclose(fitted[3], control[3])
    # The control points do not come back identical: the fit re-estimates its own
    # parameters and stops short of full convergence. On this deliberately extreme shape -
    # handles reaching 80 mm off a 150 mm chord, far beyond any pattern edge - under a
    # millimetre is what the fitter is good for, and that is the figure every refit
    # reports and is gated on.
    t2 = np.linspace(0, 1, 200)[:, None]
    refit = ((1 - t2) ** 3 * fitted[0] + 3 * (1 - t2) ** 2 * t2 * fitted[1]
             + 3 * (1 - t2) * t2 ** 2 * fitted[2] + t2 ** 3 * fitted[3])
    assert _max_distance(curve, refit) < 1.0


def test_parameter_refinement_improves_the_fit():
    """The refinement is what lets a tight interval stay a cubic instead of falling back
    to a polyline, so it is worth pinning."""
    control = np.array([[0.0, 0.0], [30.0, 80.0], [120.0, 60.0], [150.0, 0.0]])
    t = np.linspace(0, 1, 50)[:, None]
    curve = ((1 - t) ** 3 * control[0] + 3 * (1 - t) ** 2 * t * control[1]
             + 3 * (1 - t) * t ** 2 * control[2] + t ** 3 * control[3])

    def deviation(refinements):
        fitted = fit_cubic(curve, refinements=refinements)
        u = np.linspace(0, 1, 200)[:, None]
        drawn = ((1 - u) ** 3 * fitted[0] + 3 * (1 - u) ** 2 * u * fitted[1]
                 + 3 * (1 - u) * u ** 2 * fitted[2] + u ** 3 * fitted[3])
        return _max_distance(curve, drawn)

    assert deviation(2) < deviation(0)


def test_a_cubic_fit_always_pins_its_endpoints():
    points = np.column_stack([np.linspace(0, 100, 30), np.linspace(0, 100, 30) ** 1.7 / 40])
    fitted = fit_cubic(points)
    assert np.allclose(fitted[0], points[0])
    assert np.allclose(fitted[3], points[-1])


def test_every_edge_is_within_the_fitting_tolerance(generated, reconstruction_config):
    limit = reconstruction_config["boundary_refit"]["max_deviation_mm"]
    for name, result in generated.items():
        for p in result.panels:
            for fit in p.fits:
                assert fit.max_deviation_mm <= limit, (name, p.panel.key, fit.role)
                assert fit.kind in ("line", "cubic", "polyline")


def test_the_boundary_is_reduced_to_one_edge_per_seam(basic_result):
    for p in basic_result.panels:
        assert len(p.pattern_panel.edges) == len(p.panel.boundary_order)
        assert [e.role for e in p.pattern_panel.edges] == p.panel.boundary_order
        # And it is a reduction: far fewer corners than the dense boundary had.
        assert len(p.pattern_panel.edges) < len(p.panel.boundary()) / 3


def test_the_dense_boundary_is_kept_beside_the_result(basic_result):
    mesh = basic_result.flattened_mesh()
    assert len(mesh["panels"]) == 2
    for entry in mesh["panels"]:
        assert len(entry["boundary"]) > 30
        assert len(entry["vertices2d"]) == len(entry["vertices3d"])


def test_semantic_endpoints_are_preserved_exactly(basic_result):
    for p in basic_result.panels:
        edges = p.pattern_panel.edges
        for a, b in zip(edges, edges[1:] + edges[:1]):
            assert a.end.dist(b.start) < 1e-9


# ---- seam balancing --------------------------------------------------------


def test_balancing_closes_the_paired_seam_gap(basic_result):
    pairs = basic_result.seam_balance["pairs"]
    assert pairs
    for pair in pairs:
        assert pair["mismatchAfterMm"] <= pair["mismatchBeforeMm"] + 1e-9
        assert pair["balanced"], pair["roles"]
    assert basic_result.seam_balance["allBalanced"]


def test_balancing_never_moves_a_semantic_endpoint(basic_result):
    for p in basic_result.panels:
        for edge in p.pattern_panel.edges:
            points = edge.polyline(2.0)
            assert points[0].dist(edge.start) < 1e-9
            assert points[-1].dist(edge.end) < 1e-9


def test_the_seam_mismatch_check_passes(generated):
    for name, result in generated.items():
        check = next(c for c in result.report.checks if c.id == "seam_mismatch")
        assert check.status == "pass", name


# ---- determinism -----------------------------------------------------------


def test_two_runs_of_one_profile_agree_exactly():
    a = generate_profile("body_measurements_target_a.yaml")
    b = generate_profile("body_measurements_target_a.yaml")
    for pa, pb in zip(a.panels, b.panels):
        assert np.array_equal(pa.uv, pb.uv)
        assert pa.flattening.iterations == pb.flattening.iterations
    assert a.flattening_diagnostics() == b.flattening_diagnostics()


def test_a_symmetric_profile_gives_consistent_left_and_right(basic_result):
    """Centre front and centre back are folds, so the left half is the mirror of the
    right by construction: what must hold is that the two panels agree on the seams they
    share and on the hem level they both end at."""
    front, back = basic_result.panels
    assert front.panel.rows == back.panel.rows
    assert front.panel.cols == back.panel.cols
    front_side = front.pattern_panel.edge_by_role("side_seam")
    back_side = back.pattern_panel.edge_by_role("side_seam")
    assert front_side.length_mm == pytest.approx(back_side.length_mm, abs=0.5)
