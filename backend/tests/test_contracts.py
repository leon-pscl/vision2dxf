"""Every mock must satisfy its contract, and must be reproducible."""

from __future__ import annotations

import base64

import pytest

from contracts import (
    CaptureInput,
    GarmentType,
    LandmarksSource,
    MeasurementBackend,
    SegmentationInput,
)
from registry import get_adapter

GARMENTS = list(GarmentType)


@pytest.mark.parametrize("garment", GARMENTS)
def test_garment_spec_contract(garment):
    r = get_adapter("garment_spec").run(garment)
    assert r.is_mock is True
    assert r.garment_type is garment
    assert r.required_measurements, "a garment must declare what it needs"
    assert r.ease_cm
    assert r.design_rules
    assert r.warnings


@pytest.mark.parametrize("garment", GARMENTS)
def test_every_required_measurement_has_ease_or_is_known(garment):
    r = get_adapter("garment_spec").run(garment)
    known = {"chest", "waist", "hip", "shoulder_width", "back_length",
             "armhole_depth", "sleeve_length", "outseam", "inseam", "rise", "thigh"}
    assert set(r.required_measurements) <= known


def test_slacks_do_not_require_sleeve_length():
    slacks = get_adapter("garment_spec").run(GarmentType.SLACKS)
    tops = get_adapter("garment_spec").run(GarmentType.LONG_SLEEVE_POLO)
    assert "sleeve_length" not in slacks.required_measurements
    assert "sleeve_length" in tops.required_measurements


def test_capture_contract(front_png_b64, side_png_b64):
    inp = CaptureInput(
        front_image=front_png_b64, side_image=side_png_b64, height_cm=178, weight_kg=74
    )
    a = get_adapter("capture").run(inp)
    b = get_adapter("capture").run(inp)
    for field in ("distance_ok", "pose_ok", "full_body_visible", "clothing_ok"):
        assert isinstance(getattr(a, field), bool)
    assert a.is_mock is True
    # determinism
    assert a.model_dump() == b.model_dump()


def test_segmentation_contract_and_determinism(front_png_b64):
    inp = SegmentationInput(image=front_png_b64)
    a = get_adapter("segmentation").run(inp)
    b = get_adapter("segmentation").run(inp)
    assert 0.0 <= a.boundary_quality <= 1.0
    assert a.model_name
    assert a.mask_png == b.mask_png
    # it must be a real PNG
    assert base64.b64decode(a.mask_png)[:8] == b"\x89PNG\r\n\x1a\n"


def test_segmentation_box_prompt_shifts_the_mask(front_png_b64):
    centred = get_adapter("segmentation").run(SegmentationInput(image=front_png_b64))
    off = get_adapter("segmentation").run(
        SegmentationInput(image=front_png_b64, box=(10, 10, 60, 300))
    )
    assert off.mask_png != centred.mask_png


def test_calibration_contract():
    r = get_adapter("calibration").run({"front": (200, 320), "side": (160, 320)}, 178.0)
    assert set(r.px_per_cm) == {"front", "side"}
    assert all(v > 0 for v in r.px_per_cm.values())
    assert 0 <= r.relative_uncertainty <= 1
    assert r.is_mock is True


def test_landmarks_contract(front_png_b64):
    r = get_adapter("landmarks").run(front_png_b64, {"front_width": 200, "front_height": 320})
    assert r.confirmed_by_user is False
    assert r.points
    sources = {p.source for p in r.points.values()}
    # the canvas colours by source, so more than one must appear
    assert LandmarksSource.MODEL in sources
    assert LandmarksSource.HEURISTIC in sources
    for name, p in r.points.items():
        assert 0 <= p.confidence <= 1
        assert 0 <= p.x <= 200 and 0 <= p.y <= 320, name


def test_landmarks_deterministic(front_png_b64):
    a = get_adapter("landmarks").run(front_png_b64, {"front_width": 200, "front_height": 320})
    b = get_adapter("landmarks").run(front_png_b64, {"front_width": 200, "front_height": 320})
    assert a.model_dump() == b.model_dump()


@pytest.mark.parametrize("backend", list(MeasurementBackend))
def test_measurement_contract_same_type_for_both_backends(backend):
    r = get_adapter(
        "measurement_regression" if backend is MeasurementBackend.REGRESSION else "measurement_body_model"
    ).run(178.0, 74.0)
    assert r.backend is backend
    assert r.measurements
    for name, m in r.measurements.items():
        assert m.value_cm > 0, name
        assert m.lower_cm is not None and m.upper_cm is not None
        assert m.lower_cm <= m.value_cm <= m.upper_cm


def test_both_backends_agree_on_type_and_values(front_png_b64):
    """6a and 6b must return the identical MeasurementResult type, intervals
    included. Only the spread is allowed to differ."""
    reg = get_adapter("measurement_regression").run(178.0, 74.0)
    bm = get_adapter("measurement_body_model").run(178.0, 74.0)
    assert set(reg.measurements) == set(bm.measurements)
    for name in reg.measurements:
        # same estimator, so the point value has to match exactly
        assert reg.measurements[name].value_cm == bm.measurements[name].value_cm
        assert type(reg.measurements[name].upper_cm) is type(bm.measurements[name].upper_cm)


def test_measurement_deterministic():
    a = get_adapter("measurement_regression").run(178.0, 74.0)
    b = get_adapter("measurement_regression").run(178.0, 74.0)
    assert a.model_dump() == b.model_dump()


@pytest.mark.parametrize("garment", GARMENTS)
def test_steps_7_to_10_contract(garment):
    spec = get_adapter("garment_spec").run(garment)
    meas = get_adapter("measurement_regression").run(178.0, 74.0)
    meas.measurements = {k: v for k, v in meas.measurements.items()
                         if k in set(spec.required_measurements)}

    val = get_adapter("validation").run(spec, meas)
    assert val.yaml and "measurements" in val.yaml
    assert val.flags

    for route in ("block", "garmentcode"):
        d = get_adapter("drafting").run(spec, meas, route)
        assert d.route.value == route
        assert d.pieces, route
        for pc in d.pieces:
            assert "<svg" in pc.svg and pc.name

    prod = get_adapter("production").run(spec, meas)
    assert prod.export_formats == ["svg"]
    assert prod.pieces
    for pc in prod.pieces:
        assert pc.seam_allowance_cm is not None
        assert isinstance(pc.notches, list)

    drape = get_adapter("drape").run(spec)
    assert drape.toile_checklist
    assert base64.b64decode(drape.placeholder_image)[:8] == b"\x89PNG\r\n\x1a\n"


def test_measurements_change_the_pattern():
    """Mock or not, a different body must give a different pattern."""
    spec = get_adapter("garment_spec").run(GarmentType.POLO_SHIRT)
    small = get_adapter("measurement_regression").run(160.0, 55.0)
    large = get_adapter("measurement_regression").run(195.0, 100.0)
    small.measurements = {k: v for k, v in small.measurements.items() if k in set(spec.required_measurements)}
    large.measurements = {k: v for k, v in large.measurements.items() if k in set(spec.required_measurements)}
    a = get_adapter("drafting").run(spec, small, "block")
    b = get_adapter("drafting").run(spec, large, "block")
    assert a.pieces[0].svg != b.pieces[0].svg
