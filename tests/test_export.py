"""The three export files, and the gate in front of the pattern-ready one (brief 15)."""

from __future__ import annotations

import pytest
import yaml

from bodymeasure.config import load_schema
from bodymeasure.export import (
    UNVALIDATED,
    body_document,
    export_gate,
    extended_document,
    measurements_csv,
)
from bodymeasure.measure import (
    ACQUIRED,
    ESTIMATED_PENDING,
    GEOMETRIC_2D,
    UNAVAILABLE,
    Measurement,
    MeasurementSet,
)


def _set(**overrides: Measurement) -> MeasurementSet:
    """Return a measurement set with every primary key acquired, then the overrides."""
    values = {
        name: Measurement(
            value=10.0 if entry["unit"] == "deg" else 900.0,
            unit="deg" if entry["unit"] == "deg" else "mm",
            source=GEOMETRIC_2D,
            status=ACQUIRED,
            confidence=entry["confidence"],
            method="test",
        )
        for name, entry in load_schema().items()
    }
    values.update(overrides)
    return MeasurementSet(values=values)


def test_the_pattern_file_carries_the_primary_keys_and_nothing_else() -> None:
    measurements = _set(
        chest=Measurement(value=950.0, method="montazerian_ellipse"),
    )
    document = body_document(measurements, "abc")

    assert set(document["body"]) == set(load_schema())
    assert "chest" not in document["body"], "brief 3.3: an extended key would break the load"
    assert document["metadata"]["unit"] == "cm"
    assert document["metadata"]["session_id"] == "abc"
    assert document["metadata"]["validation_status"] == UNVALIDATED
    assert document["metadata"]["garmentcode_commit"]


def test_the_extended_file_carries_only_what_the_pattern_file_cannot() -> None:
    measurements = _set(chest=Measurement(value=950.0, method="montazerian_ellipse"))
    document = extended_document(measurements)

    assert set(document["measurements"]) == {"chest"}
    assert document["measurements"]["chest"]["value"] == 95.0
    assert document["measurements"]["chest"]["unit"] == "cm"
    assert document["metadata"]["validation_status"] == UNVALIDATED


def test_an_unavailable_value_exports_as_null_not_as_a_number() -> None:
    """Brief 5: never replace a failed calculation with zero."""
    measurements = _set(
        waist=Measurement(value=None, status=UNAVAILABLE, warnings=["no side depth"])
    )
    document = body_document(measurements, "abc")

    assert document["body"]["waist"] is None
    text = yaml.safe_dump(document)
    assert "waist: null" in text
    assert "waist: 0" not in text


@pytest.mark.parametrize(
    ("status", "allowed"),
    [
        (ACQUIRED, True),
        ("operator_confirmed", True),
        ("manual_input", True),
        (ESTIMATED_PENDING, False),
        (UNAVAILABLE, False),
        ("invalidated", False),
    ],
)
def test_the_gate_follows_the_status_table(status: str, allowed: bool) -> None:
    measurements = _set(waist=Measurement(value=843.0, status=status))
    gate = export_gate(measurements, "skirt")

    assert gate.ok is allowed
    if not allowed:
        assert "waist" in gate.blocked
        assert gate.reasons[0].startswith("waist: ")


def test_a_profile_that_needs_less_can_export_sooner() -> None:
    """A missing leg measurement blocks trousers but not a shirt."""
    measurements = _set(leg_circ=Measurement(value=None, status=UNAVAILABLE, warnings=["no leg"]))

    assert export_gate(measurements, "trousers").ok is False
    assert export_gate(measurements, "top_shirt").ok is True


def test_a_field_the_pipeline_never_computes_is_named_not_ignored() -> None:
    measurements = MeasurementSet(values={})
    gate = export_gate(measurements, "skirt")

    assert gate.ok is False
    assert gate.blocked["waist"] == "not computed by this pipeline"


def test_an_unknown_profile_is_an_error_not_an_open_gate() -> None:
    with pytest.raises(KeyError):
        export_gate(_set(), "wedding_dress")


def test_the_csv_names_the_method_and_never_claims_validation() -> None:
    measurements = _set(
        waist=Measurement(
            value=843.27,
            method="montazerian_ellipse",
            formula="ellipse_equation_1",
            views=["front", "side_left"],
            warnings=["check the row"],
        )
    )
    lines = measurements_csv(measurements).splitlines()
    header = lines[0].split(",")
    row = next(line for line in lines if line.startswith("waist,"))

    assert header[:3] == ["name", "value", "unit"]
    assert "validated_against_manual" in header
    assert row.startswith("waist,84.3,cm,acquired,geometric_2d,montazerian_ellipse,")
    assert "front side_left" in row
    assert row.rstrip().endswith("false,check the row")


def test_an_unavailable_value_leaves_the_csv_cell_empty() -> None:
    measurements = _set(waist=Measurement(value=None, status=UNAVAILABLE))
    row = next(
        line for line in measurements_csv(measurements).splitlines() if line.startswith("waist,")
    )
    assert row.startswith("waist,,cm,unavailable,")
