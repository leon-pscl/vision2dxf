"""YAML import: strict typing, field-specific errors, unit conversion, warnings."""
from __future__ import annotations

import pytest

from app.drafting.measurements import (
    CM_TO_MM,
    MeasurementError,
    load_yaml_document,
    parse_measurements,
    review_rows,
)

from conftest import yaml_text


def test_valid_yaml_loads_every_field(basic_body):
    assert len(review_rows(basic_body)) == 26
    assert basic_body.unit == "cm"


def test_only_yaml_extensions_accepted():
    with pytest.raises(MeasurementError) as exc:
        load_yaml_document("body: {}", "measurements.json")
    assert exc.value.errors[0]["code"] == "bad_extension"


@pytest.mark.parametrize("suffix", ["body.yaml", "body.YML", "body.yml"])
def test_accepted_extensions(suffix):
    assert load_yaml_document("body: {}", suffix) == {"body": {}}


def test_missing_body_mapping():
    with pytest.raises(MeasurementError) as exc:
        parse_measurements("height: 167\n")
    assert exc.value.errors[0]["field"] == "body"
    assert exc.value.errors[0]["code"] == "missing_body"


def test_missing_required_field_is_named():
    text = yaml_text("body_measurements.yaml").replace("  bust: 85.8\n", "")
    with pytest.raises(MeasurementError) as exc:
        parse_measurements(text)
    assert any(e["field"] == "body.bust" and e["code"] == "missing_field" for e in exc.value.errors)


@pytest.mark.parametrize(
    "raw,code",
    [
        ("  bust: \"85.8\"\n", "string_value"),
        ("  bust: null\n", "null_value"),
        ("  bust: true\n", "bool_value"),
        ("  bust: -5.0\n", "negative"),
        ("  bust: .nan\n", "non_finite"),
        ("  bust: .inf\n", "non_finite"),
        ("  bust: [85.8]\n", "non_numeric"),
    ],
)
def test_rejected_value_types(raw, code):
    text = yaml_text("body_measurements.yaml").replace("  bust: 85.8\n", raw)
    with pytest.raises(MeasurementError) as exc:
        parse_measurements(text)
    assert any(e["field"] == "body.bust" and e["code"] == code for e in exc.value.errors)


def test_unknown_field_is_rejected():
    text = yaml_text("body_measurements.yaml") + "  surprise: 1.0\n"
    with pytest.raises(MeasurementError) as exc:
        parse_measurements(text)
    assert any(e["field"] == "body.surprise" and e["code"] == "unknown_field" for e in exc.value.errors)


def test_original_values_are_kept_separate_from_derived(basic_body):
    assert basic_body.cm("bust") == pytest.approx(85.8)
    assert basic_body.mm("bust") == pytest.approx(858.0)
    assert basic_body.mm("bust") == pytest.approx(basic_body.cm("bust") * CM_TO_MM)
    # derived values never leak into the measurement model
    assert not any(key.startswith("der.") for key in basic_body.values_mm)


def test_suspicious_values_warn_without_changing_them(pascual_body):
    codes = {w.code for w in pascual_body.warnings}
    assert "underbust_vs_bust" in codes
    assert pascual_body.cm("bust") == pytest.approx(115.0)
    assert pascual_body.cm("hips_line") == pytest.approx(0.1)


def test_angles_stay_in_degrees(basic_body):
    assert basic_body.rule_inputs()["shoulder_incl"] == pytest.approx(45.0)


def test_second_fixture_loads_and_is_reviewable(pascual_body):
    rows = review_rows(pascual_body)
    assert len(rows) == 26
    assert all(row["interpretation"] for row in rows)
    origins = {row["name"]: row["origin"] for row in rows}
    assert origins["hips_line"] == "crotch"
    assert origins["bust_line"] == "neck_point"
