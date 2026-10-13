"""Ease, the garment surface, the seam network and panel segmentation."""
from __future__ import annotations

import numpy as np
import pytest

from app.assumptions import AssumptionRegistry
from app.garment_surface.ease import ease_profile, resolve_ease, underarm_drop
from app.garment_surface.offset import build_garment_surface
from app.garment_surface.seams import SEAM_ROLES, SeamError, build_seam_network
from app.garment_surface.segmentation import SegmentationError, segment, validate_panel
from app.reconstruction.anatomical_levels import build_levels
from app.reconstruction.cross_sections import build_sections
from app.reconstruction.lofting import loft
from app.reconstruction.smoothing import smooth
from conftest import generate_profile, settings as make_settings


@pytest.fixture(scope="module")
def surface(measurements):
    registry = AssumptionRegistry()
    levels = build_levels(measurements, registry)
    sections = build_sections(measurements, levels, "superellipse", registry)
    proxy, _ = smooth(loft(sections, levels))
    return proxy, levels, sections, registry


# ---- ease ------------------------------------------------------------------


def test_a_preset_supplies_all_four_ease_values():
    ease = resolve_ease(make_settings(ease_preset="fitted"))
    assert (ease.bust_mm, ease.waist_mm, ease.hip_mm) == (80.0, 80.0, 80.0)
    assert ease.armhole_mm > 0


def test_an_override_beats_the_preset():
    ease = resolve_ease(make_settings(ease_preset="fitted", bust_ease_mm=25.0))
    assert ease.bust_mm == 25.0 and ease.waist_mm == 80.0


def test_an_unknown_preset_falls_back_to_the_default():
    assert resolve_ease(make_settings(ease_preset="nonsense")).preset == "fitted"


def test_ease_is_interpolated_between_levels_not_applied_uniformly(surface):
    proxy, levels, _sections, _registry = surface
    ease = resolve_ease(make_settings(bust_ease_mm=20.0, waist_ease_mm=120.0))
    profile = ease_profile(proxy.ys, levels, ease)
    bust_row = proxy.level_rows["bust"]
    waist_row = proxy.level_rows["waist"]
    assert profile[bust_row] == pytest.approx(20.0)
    assert profile[waist_row] == pytest.approx(120.0)
    between = profile[bust_row + 1:waist_row]
    assert np.all(np.diff(between) > 0)          # monotone
    assert 20.0 < between.min() and between.max() < 120.0
    # Above the bust the ease is held, not extrapolated.
    assert np.allclose(profile[:bust_row], 20.0)


def test_the_ease_profile_is_smooth(surface):
    """A smoothstep between levels, so there is no crease where two values meet."""
    proxy, levels, _s, _r = surface
    ease = resolve_ease(make_settings(bust_ease_mm=0.0, waist_ease_mm=100.0))
    profile = ease_profile(proxy.ys, levels, ease)
    second = np.diff(profile, 2)
    assert np.max(np.abs(second)) < 30.0


def test_armhole_ease_becomes_an_underarm_drop():
    ease = resolve_ease(make_settings(armhole_ease_mm=40.0))
    assert underarm_drop(ease) == pytest.approx(20.0)


# ---- the garment surface ---------------------------------------------------


def test_the_garment_surface_achieves_the_requested_circumferences(surface):
    proxy, levels, _sections, registry = surface
    ease = resolve_ease(make_settings(bust_ease_mm=80.0, waist_ease_mm=60.0))
    garment, diagnostics = build_garment_surface(proxy, levels, ease, registry)
    bust = diagnostics["levels"]["bust"]
    assert bust["achievedCircumferenceMm"] == pytest.approx(
        bust["bodyCircumferenceMm"] + 80.0, abs=1e-6
    )
    assert diagnostics["maxEaseResidualMm"] < 1e-6


def test_circumference_ease_is_not_a_constant_normal_offset(surface):
    """The documented consequence of scaling a section: the surface moves furthest where
    the radius is largest."""
    proxy, levels, _sections, registry = surface
    ease = resolve_ease(make_settings(bust_ease_mm=80.0))
    _garment, diagnostics = build_garment_surface(proxy, levels, ease, registry)
    bust = diagnostics["levels"]["bust"]
    assert bust["normalOffsetMaxMm"] > bust["normalOffsetMinMm"] * 1.2


def test_the_garment_surface_stays_symmetric(surface):
    proxy, levels, _sections, registry = surface
    garment, _ = build_garment_surface(proxy, levels, resolve_ease(make_settings()), registry)
    verts = garment.vertices()
    # Column j and column -j are mirror images in x on every row.
    assert np.allclose(verts[:, 1, 0], -verts[:, -1, 0], atol=1e-9)
    assert np.allclose(verts[:, 1, 2], verts[:, -1, 2], atol=1e-9)


def test_the_conversion_from_ease_to_displacement_is_recorded(surface):
    proxy, levels, _sections, registry = surface
    build_garment_surface(proxy, levels, resolve_ease(make_settings()), registry)
    assert "garment.ease_conversion" in {a["id"] for a in registry.as_list()}


# ---- the seam network ------------------------------------------------------


def test_the_network_places_every_documented_seam(basic_result):
    network = basic_result.network
    assert set(SEAM_ROLES) == set(network.as_dict()["roles"])
    assert 0 < network.neck_column < network.tip_column < network.side_column
    assert network.side_column * 2 == network.back_column


def test_the_trim_curve_is_continuous_across_its_three_intervals(basic_result):
    """The three intervals must agree at the two columns they share, or the panel would
    be cut at a step."""
    network = basic_result.network
    for panel in ("front", "back"):
        ys = [network.trim_y(panel, j) for j in range(network.side_column + 1)]
        assert ys[0] == pytest.approx(network.neck_drop(panel))
        # neckline ends where the shoulder seam begins, at the neck reference level
        assert ys[network.neck_column] == pytest.approx(0.0, abs=1e-9)
        # the shoulder ends where the armscye begins, at the shoulder tip
        assert ys[network.tip_column] == pytest.approx(network.shoulder_tip_y_mm())
        # and the armscye ends at the underarm, where the side seam starts
        assert ys[network.side_column] == pytest.approx(network.underarm_y_mm)
        assert all(y >= -1e-9 for y in ys)
        # Below the neck point the trim curve only ever descends.
        tail = ys[network.neck_column:]
        assert all(b >= a - 1e-9 for a, b in zip(tail, tail[1:]))


def test_the_trim_curve_is_symmetric_front_to_back_except_for_the_neck(basic_result):
    network = basic_result.network
    back = network.back_column
    for j in range(network.neck_column + 1, network.side_column + 1):
        assert network.trim_y("front", j) == pytest.approx(network.trim_y("front", back - j))


def test_the_shoulder_seam_follows_the_confirmed_inclination(basic_result, measurements):
    network = basic_result.network
    assert network.shoulder_slope_deg == measurements.shoulder_slope_degrees
    import math
    expected = (network.tip_x_mm - network.neck_x_mm) * math.tan(
        math.radians(network.shoulder_slope_deg))
    assert network.shoulder_tip_y_mm() == pytest.approx(expected)


def test_the_underarm_sits_between_the_shoulder_tip_and_the_waist(basic_result):
    network = basic_result.network
    assert network.shoulder_tip_y_mm() < network.underarm_y_mm < network.waist_y_mm


def test_a_shoulder_wider_than_the_chest_section_is_refused(surface, measurements):
    proxy, levels, sections, registry = surface
    ease = resolve_ease(make_settings())
    garment, _ = build_garment_surface(proxy, levels, ease, registry)
    broken = measurements.model_copy(update={"shoulder_width_full": 4000.0})
    with pytest.raises(SeamError) as exc:
        build_seam_network(garment, sections, levels, broken, make_settings(),
                           underarm_drop(ease), registry)
    assert exc.value.errors[0]["code"] == "SHOULDER_TIP_OUTSIDE_SECTION"


def test_an_unknown_seam_preset_is_refused(surface, measurements):
    proxy, levels, sections, registry = surface
    ease = resolve_ease(make_settings())
    garment, _ = build_garment_surface(proxy, levels, ease, registry)
    with pytest.raises(SeamError) as exc:
        build_seam_network(garment, sections, levels, measurements,
                           make_settings(seam_preset="princess_v9"),
                           underarm_drop(ease), registry)
    assert exc.value.errors[0]["code"] == "UNKNOWN_SEAM_PRESET"


def test_a_deeper_neckline_moves_only_the_neckline(basic_result):
    deeper = generate_profile("body_measurements.yaml", front_neck_drop_mm=140.0)
    assert deeper.network.front_neck_drop_mm == 140.0
    assert deeper.network.trim_y("front", 0) == pytest.approx(140.0)
    assert deeper.network.shoulder_tip_y_mm() == pytest.approx(
        basic_result.network.shoulder_tip_y_mm())


# ---- segmentation ----------------------------------------------------------


def test_segmentation_produces_a_front_and_a_back_patch(basic_result):
    assert [p.panel.key for p in basic_result.panels] == ["front", "back"]


def test_each_patch_is_a_disk_with_one_boundary_loop(basic_result):
    for result in basic_result.panels:
        validate_panel(result.panel)  # raises otherwise
        assert len(result.panel.boundary_loops) == 1
        euler = (len(result.panel.vertices_3d) - _edges(result.panel)
                 + len(result.panel.faces))
        assert euler == 1  # a disk


def _edges(panel) -> int:
    seen = set()
    for a, b, c in panel.faces:
        for u, v in ((a, b), (b, c), (c, a)):
            seen.add((min(u, v), max(u, v)))
    return len(seen)


def test_the_boundary_walk_covers_the_semantic_intervals_exactly(basic_result):
    for result in basic_result.panels:
        panel = result.panel
        walked: list[int] = []
        for role in panel.boundary_order:
            ids = panel.semantic_boundaries[role]
            walked.extend(ids if not walked else ids[1:])
        assert walked[-1] == panel.boundary()[0]  # closes
        assert walked[:-1] == panel.boundary()


def test_paired_seams_are_the_same_curve_on_the_surface(basic_result):
    """Front and back share the side seam and mirror the shoulder, so their 3D lengths
    must be identical before anything is flattened."""
    for seam in basic_result.surface_seams:
        assert seam["differenceMm"] < 1e-6, seam["seam"]


def test_the_configured_resolution_is_honoured():
    coarse = generate_profile("body_measurements.yaml", cross_section_resolution=16,
                              longitudinal_segments=2)
    assert coarse.proxy.columns == 16
    assert coarse.network.side_column == 4
    assert len(coarse.proxy.ys) < 20


def test_a_resolution_not_divisible_by_four_is_refused():
    from app.pipeline import GenerationBlocked

    with pytest.raises(GenerationBlocked) as exc:
        generate_profile("body_measurements.yaml", cross_section_resolution=18)
    assert exc.value.code == "proxy_surface_invalid"
    assert exc.value.details[0]["code"] == "RESOLUTION_NOT_DIVISIBLE"


def test_a_panel_grid_too_small_is_refused(basic_result):
    with pytest.raises(SegmentationError) as exc:
        segment(basic_result.garment, basic_result.network, panel_rows=1)
    assert exc.value.errors[0]["code"] == "PANEL_GRID_TOO_COARSE"
