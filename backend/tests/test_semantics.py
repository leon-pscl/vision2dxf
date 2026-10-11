"""Measurement Semantics Layer: statuses, normalization, blocking rules, adapter."""
from __future__ import annotations

import pytest

from app.drafting.measurements import parse_source
from app.drafting.semantics import (
    AMBIGUOUS,
    ASSUMED,
    BLOCKING_STATUSES,
    CONFIRMED,
    DERIVED,
    INVALID,
    MISSING,
    OPTIONAL,
    DraftingMeasurements,
    SemanticError,
    SemanticsRegistry,
    registry,
    resolve,
)

from conftest import yaml_text


def test_registry_declares_a_schema_and_definition_version(semantics_registry):
    assert semantics_registry.schema_version
    assert semantics_registry.definition_version
    assert semantics_registry.coordinate_system["vertical_origin"]


def test_every_field_is_classified(semantics_registry):
    categories = set(semantics_registry.raw["categories"])
    for name, f in semantics_registry.fields.items():
        assert f.category in categories, name
        assert f.quantity_type
        assert f.status in {"confirmed", "ambiguous", "optional"}, name


def test_ambiguous_fields_offer_choices_and_pinned_ones_do_not(semantics_registry):
    for name, f in semantics_registry.fields.items():
        assert f.candidates, name
        assert len({c.id for c in f.candidates}) == len(f.candidates), name
        if f.status == "ambiguous":
            assert all(c.drafting_variable for c in f.candidates), name


def test_candidates_declare_full_and_half_representations(semantics_registry):
    widths = [f for f in semantics_registry.fields.values() if f.category == "width"]
    reps = {c.representation for f in widths for c in f.candidates}
    assert {"full", "half"} <= reps


def test_drafting_variables_are_descriptive(semantics_registry):
    variables = {c.drafting_variable for f in semantics_registry.fields.values() for c in f.candidates}
    assert {"bust_circumference", "shoulder_width_full", "back_width_full",
            "neck_to_bust_apex_vertical", "bust_apex_spacing_full", "armscye_depth_vertical",
            "shoulder_slope_degrees"} <= variables
    for v in variables:
        assert not v.endswith("_w"), v


# ---- statuses ---------------------------------------------------------------

def test_unconfirmed_required_fields_block_generation(basic_source, semantics_registry):
    resolution = resolve(basic_source.values_cm)
    assert not resolution.can_generate
    assert resolution.measurements is None
    blocked = {b["field"] for b in resolution.blocking}
    expected = {n for n, f in semantics_registry.fields.items() if f.status == "ambiguous"}
    assert blocked == expected


def test_ambiguous_status_is_reported_per_field(basic_source):
    resolution = resolve(basic_source.values_cm)
    for name, f in resolution.fields.items():
        if f.status == AMBIGUOUS:
            assert f.confirmed_by is None
            assert f.candidate_id is None
            assert f.suggested_action == "Confirm interpretation"


def test_pinned_fields_are_confirmed_without_operator_action(basic_source):
    resolution = resolve(basic_source.values_cm, {})
    assert resolution.fields["bust"].status == CONFIRMED
    assert resolution.fields["bust"].confirmed_by == "schema"
    assert resolution.fields["underbust"].status == CONFIRMED  # optional but unambiguous


def test_operator_confirmation_moves_a_field_to_assumed(basic_source):
    resolution = resolve(basic_source.values_cm, {"shoulder_w": "half_neck_to_shoulder_tip"})
    res = resolution.fields["shoulder_w"]
    assert res.status == ASSUMED
    assert res.confirmed_by == "operator"
    assert any(a["field"] == "shoulder_w" for a in resolution.assumptions)


def test_missing_required_field_is_reported_as_missing():
    values = parse_source(yaml_text("body_measurements.yaml")).values_cm
    values.pop("bust")
    resolution = resolve(values, {})
    assert any(b["field"] == "bust" and b["code"] == MISSING for b in resolution.blocking)


def test_out_of_range_value_is_invalid_not_silently_clamped(basic_source, default_confirmations):
    resolution = resolve(basic_source.values_cm, default_confirmations, {"bust": 400.0})
    assert any(b["code"] == INVALID for b in resolution.blocking)
    assert resolution.fields["bust"].source_value == 400.0
    assert resolution.fields["bust"].original_value == basic_source.values_cm["bust"]


def test_correction_keeps_the_original_value(basic_source, default_confirmations):
    resolution = resolve(basic_source.values_cm, default_confirmations, {"bust": 86.0})
    res = resolution.fields["bust"]
    assert res.original_value == 85.8
    assert res.source_value == 86.0
    assert res.corrected is True
    assert any(a["kind"] == "corrected_value" for a in resolution.assumptions)


def test_optional_field_absent_does_not_block(basic_source, default_confirmations):
    values = dict(basic_source.values_cm)
    values.pop("wrist")
    resolution = resolve(values, default_confirmations)
    assert resolution.can_generate
    assert resolution.fields["wrist"].status == OPTIONAL


# ---- normalization ----------------------------------------------------------

def test_units_are_converted_and_recorded(basic_resolution):
    res = basic_resolution.fields["bust"]
    assert res.source_value == 85.8
    assert res.normalized_mm == pytest.approx(858.0)
    assert res.unit_source == "cm" and res.unit_pattern == "mm"
    assert "cm -> mm" in res.conversion


def test_angles_are_not_length_converted(basic_resolution):
    res = basic_resolution.fields["shoulder_incl"]
    assert res.normalized_mm == pytest.approx(45.0)
    assert "identity" in res.conversion


def test_half_representation_candidates_are_scaled(basic_resolution):
    assert basic_resolution.fields["shoulder_w"].normalized_mm == pytest.approx(342.0)  # 17.1 * 2 * 10
    assert basic_resolution.fields["back_width"].normalized_mm == pytest.approx(390.0)  # 39.0 * 10


def test_alternative_candidate_changes_the_normalized_value(basic_source, default_confirmations):
    conf = {**default_confirmations, "shoulder_w": "full_across_shoulder_tips",
            "back_width": "centre_back_to_side_seam"}
    resolution = resolve(basic_source.values_cm, conf)
    assert resolution.fields["shoulder_w"].normalized_mm == pytest.approx(171.0)
    assert resolution.fields["back_width"].normalized_mm == pytest.approx(780.0)
    assert resolution.can_generate


def test_angle_transform_candidate(basic_source, default_confirmations):
    resolution = resolve(basic_source.values_cm, {**default_confirmations,
                                                  "shoulder_incl": "from_vertical"})
    assert resolution.fields["shoulder_incl"].normalized_mm == pytest.approx(45.0)
    assert "transform=subtract_from_ninety" in resolution.fields["shoulder_incl"].conversion


def test_relative_candidate_adds_the_confirmed_bust_level(basic_source, default_confirmations):
    conf = {**default_confirmations, "vert_bust_line": "bust_line_to_bust_apex"}
    resolution = resolve(basic_source.values_cm, conf)
    res = resolution.fields["vert_bust_line"]
    assert res.normalized_mm == pytest.approx(359.0 + 294.0)
    assert "+ neck_to_bust_level" in res.conversion


def test_relative_candidate_subtracts_the_shoulder_drop(basic_source, default_confirmations):
    conf = {**default_confirmations, "armscye_depth": "neck_reference_point_to_underarm"}
    resolution = resolve(basic_source.values_cm, conf)
    res = resolution.fields["armscye_depth"]
    assert res.normalized_mm < 201.0
    assert "shoulder_tip_drop_below_neck_reference" in res.conversion


# ---- blocking consistency rules ---------------------------------------------

def test_mixed_vertical_origins_block_generation(basic_source, default_confirmations):
    conf = {**default_confirmations, "waist_line": "level_from_top_of_head"}
    resolution = resolve(basic_source.values_cm, conf)
    codes = {b["code"] for b in resolution.blocking}
    assert "NECK_REFERENCE_LEVEL_UNAVAILABLE" in codes
    assert not resolution.can_generate


def test_shared_origin_rule_is_a_blocking_schema_message(basic_source, default_confirmations):
    """An inferred origin warning must be blocking, not informational."""
    conf = {**default_confirmations, "bust_line": "level_from_top_of_head",
            "waist_line": "level_from_neck_reference_point"}
    resolution = resolve(basic_source.values_cm, conf)
    assert not resolution.can_generate
    assert any(b["severity"] == "blocking" for b in resolution.blocking)


def test_waist_must_sit_below_the_bust_line(basic_source, default_confirmations):
    resolution = resolve(basic_source.values_cm, default_confirmations, {"waist_line": 30.0})
    assert any(b["code"] == "WAIST_NOT_BELOW_BUST" for b in resolution.blocking)


def test_bust_apex_must_sit_below_the_bust_line(basic_source, default_confirmations):
    resolution = resolve(basic_source.values_cm, default_confirmations, {"vert_bust_line": 20.0})
    assert any(b["code"] == "BUST_APEX_NOT_BELOW_BUST_LINE" for b in resolution.blocking)


def test_pascual_profile_is_rejected_by_the_schema_ranges(pascual_source, default_confirmations):
    resolution = resolve(pascual_source.values_cm, default_confirmations)
    assert not resolution.can_generate
    assert {b["code"] for b in resolution.blocking} == {INVALID}
    assert any(b["field"] == "bust_line" for b in resolution.blocking)


# ---- adapter ----------------------------------------------------------------

def test_adapter_exposes_no_source_field_names(basic_resolution):
    names = set(DraftingMeasurements.model_fields)
    for source_name in ("bust", "shoulder_w", "vert_bust_line", "bust_line", "waist_line"):
        assert source_name not in names


def test_adapter_produces_the_documented_variables(basic_resolution):
    m = basic_resolution.measurements
    assert m.bust_circumference == pytest.approx(858.0)
    assert m.waist_circumference == pytest.approx(856.0)
    assert m.shoulder_width_full == pytest.approx(342.0)
    assert m.back_width_full == pytest.approx(390.0)
    assert m.neck_to_bust_apex_vertical == pytest.approx(359.0)
    assert m.bust_to_waist_vertical == pytest.approx(117.0)
    assert m.armscye_depth_vertical == pytest.approx(201.0)
    assert m.shoulder_slope_degrees == pytest.approx(45.0)


def test_bust_to_waist_is_a_derived_difference_not_a_raw_level(basic_resolution):
    """The spec forbids using waist_line directly as the bodice length."""
    derived = basic_resolution.derived["bust_to_waist_vertical"]
    assert derived["status"] == DERIVED
    assert "neck_to_waist_level" in derived["equation"]
    assert derived["dependencies"] == ["neck_to_bust_level", "neck_to_waist_level"]
    m = basic_resolution.measurements
    assert m.bust_to_waist_vertical == abs(m.neck_to_waist_level - m.neck_to_bust_level)


def test_waist_to_hip_is_reported_as_not_derivable(basic_resolution):
    entry = basic_resolution.derived["waist_to_hip_vertical"]
    assert entry["value"] is None
    assert "crotch" in entry["equation"]


def test_adapter_is_immutable(basic_resolution):
    with pytest.raises(Exception):
        basic_resolution.measurements.bust_circumference = 1.0


def test_resolution_is_deterministic(basic_source, default_confirmations):
    first = resolve(basic_source.values_cm, default_confirmations)
    second = resolve(basic_source.values_cm, default_confirmations)
    assert first.measurements == second.measurements
    assert first.as_dict() == second.as_dict()


# ---- registry integrity -----------------------------------------------------

def test_a_broken_schema_is_a_developer_error():
    raw = dict(registry().raw)
    broken = {
        "schema_version": "0", "definition_version": "0", "coordinate_system": raw["coordinate_system"],
        "categories": raw["categories"], "consistency_rules": [],
        "fields": {"bust": {"label": "Bust", "category": "circumference", "quantity_type": "c",
                            "required": True, "status": "ambiguous", "unit": "cm"}},
    }
    with pytest.raises(SemanticError):
        SemanticsRegistry(broken)


def test_blocking_statuses_are_the_documented_three():
    assert BLOCKING_STATUSES == {MISSING, AMBIGUOUS, INVALID}
