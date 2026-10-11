"""YAML import: syntax validation and strict typing only. No interpretation here."""
from __future__ import annotations

import pytest

from app.drafting.measurements import (
    CM_TO_MM,
    MeasurementError,
    load_yaml_document,
    parse_source,
    source_rows,
)

from conftest import yaml_text


def test_valid_yaml_loads_every_field(basic_source):
    assert len(source_rows(basic_source)) == 26
    assert basic_source.unit == "cm"


def test_only_yaml_extensions_accepted():
    with pytest.raises(MeasurementError) as exc:
        load_yaml_document("body: {}", "measurements.json")
    assert exc.value.errors[0]["code"] == "bad_extension"


@pytest.mark.parametrize("suffix", ["body.yaml", "body.YML", "body.yml"])
def test_accepted_extensions(suffix):
    assert load_yaml_document("body: {}", suffix) == {"body": {}}


def test_missing_body_mapping():
    with pytest.raises(MeasurementError) as exc:
        parse_source("height: 167\n")
    assert exc.value.errors[0]["field"] == "body"
    assert exc.value.errors[0]["code"] == "missing_body"


def test_missing_required_field_is_kept_as_absent_not_rejected():
    """Absence is a semantics MISSING, not a syntax error."""
    text = yaml_text("body_measurements.yaml").replace("  bust: 85.8\n", "")
    profile = parse_source(text)
    assert "bust" not in profile.values_cm
    assert len(source_rows(profile)) == 25


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
        parse_source(text)
    assert any(e["field"] == "body.bust" and e["code"] == code for e in exc.value.errors)


def test_unknown_field_is_rejected():
    text = yaml_text("body_measurements.yaml") + "  surprise: 1.0\n"
    with pytest.raises(MeasurementError) as exc:
        parse_source(text)
    assert any(e["field"] == "body.surprise" and e["code"] == "unknown_field" for e in exc.value.errors)


def test_original_values_are_kept_separate_from_derived(basic_source, basic_resolution):
    assert basic_source.cm("bust") == pytest.approx(85.8)
    assert basic_source.mm("bust") == pytest.approx(858.0)
    assert basic_source.mm("bust") == pytest.approx(basic_source.cm("bust") * CM_TO_MM)
    # the raw layer never carries a derived dimension
    assert not any("." in key for key in basic_source.values_cm)


def test_every_imported_value_is_displayable(basic_source):
    rows = source_rows(basic_source)
    assert all(row["interpretation"] and row["landmark"] for row in rows)
    assert {r["name"]: r["value"] for r in rows}["bust"] == pytest.approx(85.8)


def test_second_profile_loads_for_review(pascual_source):
    rows = source_rows(pascual_source)
    assert len(rows) == 26
    origins = {row["name"]: row["origin"] for row in rows}
    assert origins["hips_line"] == "crotch"
    assert origins["bust_line"] == "neck_point"


def test_parser_does_not_import_fields_as_dict(basic_source):
    """The raw layer is a flat float map; nothing is interpreted at parse time."""
    assert all(isinstance(v, float) for v in basic_source.values_cm.values())
