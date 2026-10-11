"""Panel geometry: continuity, closure, darts, ease, seam pairing, determinism."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import drafting_config
from app.drafting.ease import Settings
from app.drafting.pipeline import SemanticsBlocked, draft
from app.drafting.semantics import resolve

GOLDEN = Path(__file__).parent / "golden"
TOL = 1e-6


def _front(result):
    return next(p for p in result.document.panels if p.key == "front")


def _back(result):
    return next(p for p in result.document.panels if p.key == "back")


# -- ease --------------------------------------------------------------------

def test_ease_presets_change_the_widths(basic_resolution):
    loose = draft(basic_resolution, Settings(ease_preset="relaxed"))
    tight = draft(basic_resolution, Settings(ease_preset="skin_thin"))
    assert loose.values["der.front_bust_half_width"] > tight.values["der.front_bust_half_width"]
    assert loose.values["der.front_bust_half_width"] - tight.values["der.front_bust_half_width"] == \
        pytest.approx((140.0 - 20.0) / 2)


def test_explicit_ease_overrides_the_preset(basic_resolution):
    result = draft(basic_resolution, Settings(ease_preset="fitted", bust_ease_mm=200.0))
    assert result.values["der.front_bust_half_width"] == pytest.approx(
        (858.0 + 200.0 - 390.0) / 2)


def test_ease_does_not_move_the_vertical_levels(basic_resolution):
    base = draft(basic_resolution, Settings(ease_preset="fitted")).values
    loose = draft(basic_resolution, Settings(ease_preset="relaxed")).values
    for key in ("lm.front.underarm_y", "lm.front.shoulder_tip_y", "lm.front.waist_side_y"):
        assert base[key] == pytest.approx(loose[key])


# -- continuity and closure --------------------------------------------------

def test_adjacent_edges_share_endpoints(basic_result):
    tol = drafting_config()["tolerances"]["endpoint_continuity_mm"]
    for panel in basic_result.document.panels:
        for edge, nxt in zip(panel.edges, panel.edges[1:] + panel.edges[:1]):
            assert edge.end.dist(nxt.start) <= tol


def test_boundary_is_closed(basic_result):
    tol = drafting_config()["tolerances"]["panel_closure_mm"]
    for panel in basic_result.document.panels:
        contour = panel.outer_contour()
        assert len(contour) >= 3
        assert contour[0].dist(panel.edges[-1].end) <= tol


def test_curve_endpoints_match_their_edges(basic_result):
    for panel in basic_result.document.panels:
        for edge in panel.edges:
            if edge.curve is None:
                continue
            assert edge.curve.p0 == edge.start
            assert edge.curve.p3 == edge.end
            assert edge.curve.sample(2.0)[0] == edge.start
            assert edge.curve.sample(2.0)[-1] == edge.end


def test_every_edge_has_a_distinct_id(basic_result):
    ids = [e.ensure_id() for p in basic_result.document.panels for e in p.edges]
    assert len(ids) == len(set(ids))


def test_every_edge_traces_to_a_rule(basic_result):
    for panel in basic_result.document.panels:
        for edge in panel.edges:
            assert edge.rule_id


# -- darts -------------------------------------------------------------------

def test_front_carries_one_bust_dart_and_the_back_none(basic_result):
    assert len(_front(basic_result).darts) == 1
    assert _back(basic_result).darts == []


def test_dart_legs_sit_on_the_waist_and_bust_lines(basic_result):
    """Panel geometry is stored centred on its own bbox, so compare the gap between
    the two leg levels against the gap the rules produced."""
    v = basic_result.values
    dart = _front(basic_result).darts[0]
    top, bottom = dart.legs[0][1].y, dart.legs[1][1].y
    assert bottom - top == pytest.approx(v["lm.front.dart_bottom_y"] - v["lm.front.dart_top_y"])
    assert top < dart.apex.y < bottom


def test_dart_intake_is_symmetric_about_the_apex(basic_result):
    dart = _front(basic_result).darts[0]
    for apex, a, b in dart.legs:
        assert (a.x + b.x) / 2 == pytest.approx(apex.x)
        assert b.x - a.x == pytest.approx(dart.intake_mm)


def test_dart_is_inside_its_panel(basic_result):
    check = next(c for c in basic_result.report.checks if c.id.startswith("dart_containment"))
    assert check.status == "pass"


def test_no_dart_when_the_apex_spacing_is_zero(basic_source, default_confirmations):
    resolution = resolve(basic_source.values_cm, default_confirmations, {"bust_points": 0.0})
    result = draft(resolution, Settings())
    assert result.values["der.dart_intake"] == pytest.approx(0.0)
    assert _front(result).darts == []
    assert result.report.status == "pass"


# -- seams -------------------------------------------------------------------

def test_front_and_back_side_seams_are_paired(basic_result):
    front_edge = _front(basic_result).edge_by_role("side_seam")
    back_edge = _back(basic_result).edge_by_role("side_seam")
    assert front_edge.paired_edge_id == back_edge.ensure_id()
    assert back_edge.paired_edge_id == front_edge.ensure_id()


def test_paired_side_seams_have_the_same_length(basic_result):
    check = next(c for c in basic_result.report.checks if c.id == "seam_mismatch")
    assert check.status == "pass"
    for pair in check.metrics["pairs"]:
        assert pair["deltaMm"] <= pair["allowedMm"] + TOL


def test_every_stitch_parameter_is_normalised(basic_result):
    for pair in basic_result.document.stitches:
        assert len(pair.records) == 2
        for rec in pair.records:
            assert 0.0 <= rec.param_start <= 1.0
            assert 0.0 <= rec.param_end <= 1.0


def test_stitches_reference_existing_edges(basic_result):
    for pair in basic_result.document.stitches:
        for rec in pair.records:
            assert basic_result.document.edge_by_id(rec.edge_id) is not None


# -- semantics reach the geometry --------------------------------------------

def test_changing_an_interpretation_changes_the_pattern(basic_source, default_confirmations):
    """A confirmed representation must actually drive the draft, not just the report."""
    half = draft(resolve(basic_source.values_cm, default_confirmations), Settings())
    full = draft(resolve(basic_source.values_cm,
                         {**default_confirmations, "shoulder_w": "full_across_shoulder_tips"}),
                 Settings())
    assert half.values["der.shoulder_half_width"] == pytest.approx(171.0)
    assert full.values["der.shoulder_half_width"] == pytest.approx(85.5)


def test_measurements_drive_the_geometry(basic_source, basic_resolution, default_confirmations):
    small = draft(basic_resolution, Settings())
    wide = draft(resolve(basic_source.values_cm,
                         {**default_confirmations, "back_width": "centre_back_to_side_seam"}),
                 Settings())
    # a different confirmed representation of the same source value moves the side seam
    assert wide.values["der.front_bust_half_width"] < small.values["der.front_bust_half_width"]


def test_a_profile_outside_the_schema_ranges_never_reaches_the_draft(pascual_source,
                                                                    default_confirmations):
    resolution = resolve(pascual_source.values_cm, default_confirmations)
    assert not resolution.can_generate
    with pytest.raises(SemanticsBlocked):
        draft(resolution, Settings())


def test_no_master_pattern_is_loaded(basic_source, default_confirmations, monkeypatch):
    """The draft may not read the reference fixture."""
    opened: list[str] = []
    real_open = open

    def spy(path, *args, **kwargs):
        opened.append(str(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", spy)
    draft(resolve(basic_source.values_cm, default_confirmations), Settings())
    assert not any("garmageset" in p for p in opened)


def test_repeated_runs_are_geometrically_identical(basic_resolution):
    first = draft(basic_resolution, Settings())
    second = draft(basic_resolution, Settings())
    for a, b in zip(first.document.panels, second.document.panels):
        assert a.ensure_id() == b.ensure_id()
        for ea, eb in zip(a.edges, b.edges):
            assert ea.ensure_id() == eb.ensure_id()
            assert ea.start == eb.start and ea.end == eb.end
            assert ea.length_mm == pytest.approx(eb.length_mm, abs=TOL)


def test_repeated_runs_over_the_file_are_identical(basic_source, default_confirmations):
    a = draft(resolve(basic_source.values_cm, default_confirmations), Settings())
    b = draft(resolve(basic_source.values_cm, default_confirmations), Settings())
    assert json.dumps(_snapshot(a)) == json.dumps(_snapshot(b))


# -- golden files ------------------------------------------------------------

def _snapshot(result) -> dict:
    return {
        "panels": [
            {
                "key": p.key,
                "edges": [
                    {"role": e.role, "lengthMm": round(e.length_mm, 6),
                     "start": [round(e.start.x, 6), round(e.start.y, 6)],
                     "end": [round(e.end.x, 6), round(e.end.y, 6)]}
                    for e in p.edges
                ],
                "darts": [{"intakeMm": round(d.intake_mm, 6),
                           "apex": [round(d.apex.x, 6), round(d.apex.y, 6)]} for d in p.darts],
            }
            for p in result.document.panels
        ],
        "values": {k: round(v, 6) for k, v in sorted(result.values.items()) if k.startswith("der.")},
    }


def _compare(actual, expected, path=""):
    if isinstance(expected, dict):
        assert set(actual) == set(expected), path
        for key in expected:
            _compare(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert len(actual) == len(expected), path
        for i, (a, e) in enumerate(zip(actual, expected)):
            _compare(a, e, f"{path}[{i}]")
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, abs=1e-4), f"{path}: {actual} != {expected}"
    else:
        assert actual == expected, path


def test_golden_snapshot(basic_result):
    expected = json.loads((GOLDEN / "basic_bodice.json").read_text(encoding="utf-8"))
    _compare(_snapshot(basic_result), expected)


def test_golden_derived_values(basic_result):
    expected = json.loads((GOLDEN / "basic_bodice.json").read_text(encoding="utf-8"))["values"]
    assert {k: round(v, 6) for k, v in basic_result.values.items() if k.startswith("der.")} == expected


def test_golden_semantics_for_both_profiles(basic_resolution, pascual_resolution):
    """Golden record of the semantics decision for each supplied profile."""
    expected = json.loads((GOLDEN / "semantics_resolution.json").read_text(encoding="utf-8"))
    for key, resolution in (("basic", basic_resolution), ("pascual", pascual_resolution)):
        assert resolution.schema_version == expected[key]["schemaVersion"]
        assert resolution.can_generate == expected[key]["canGenerate"]
        assert {f.name: f.status for f in resolution.fields.values()} == expected[key]["statuses"]
        assert [(b["field"], b["code"]) for b in resolution.blocking] == \
            [tuple(b) for b in expected[key]["blocking"]]
        if resolution.measurements:
            assert resolution.measurements.model_dump() == expected[key]["draftingMeasurements"]
        else:
            assert expected[key]["draftingMeasurements"] is None
