"""YAML import: syntax validation and strict typing only.

This layer never interprets a field. It produces the raw source values; meaning is
resolved downstream by the Measurement Semantics Layer (drafting/semantics.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
import yaml

from ..config import measurement_index

CM_TO_MM = 10.0


class MeasurementError(Exception):
    """Field-specific import failure, surfaced by the API as structured errors."""

    def __init__(self, errors: list[dict]) -> None:
        self.errors = errors
        super().__init__("; ".join(f"{e['field']}: {e['message']}" for e in errors))


@dataclass
class SourceProfile:
    """Raw, confirmed-to-parse values exactly as the file reports them."""

    values_cm: dict[str, float]
    unit: str = "cm"
    filename: str = "body_measurements.yaml"

    def cm(self, name: str) -> float:
        return self.values_cm[name]

    def mm(self, name: str) -> float:
        return self.values_cm[name] * CM_TO_MM


def load_yaml_document(text: str, filename: str) -> dict:
    if not filename.lower().endswith((".yaml", ".yml")):
        raise MeasurementError(
            [{"field": "file", "message": "Only .yaml and .yml files are accepted.",
              "code": "bad_extension"}]
        )
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise MeasurementError(
            [{"field": "file", "message": f"YAML parse error: {exc}", "code": "yaml_parse_error"}]
        ) from exc
    if not isinstance(data, dict):
        raise MeasurementError(
            [{"field": "file", "message": "Top level of the document must be a mapping.",
              "code": "not_a_mapping"}]
        )
    return data


def parse_source(text: str, filename: str = "body_measurements.yaml") -> SourceProfile:
    document = load_yaml_document(text, filename)
    index = measurement_index()
    errors: list[dict] = []

    body = document.get("body")
    if body is None:
        raise MeasurementError(
            [{"field": "body", "message": "Top level 'body' mapping is required.",
              "code": "missing_body"}]
        )
    if not isinstance(body, dict):
        raise MeasurementError(
            [{"field": "body", "message": "'body' must be a mapping.", "code": "body_not_mapping"}]
        )

    for key, value in body.items():
        if key not in index:
            errors.append({"field": f"body.{key}", "message": "Unknown measurement field.",
                           "code": "unknown_field"})
            continue
        if value is None:
            errors.append({"field": f"body.{key}", "message": "Value must not be null.",
                           "code": "null_value"})
        elif isinstance(value, bool):
            errors.append({"field": f"body.{key}", "message": "Value must not be a boolean.",
                           "code": "bool_value"})
        elif isinstance(value, str):
            errors.append({"field": f"body.{key}", "message": "Value must not be a string.",
                           "code": "string_value"})
        elif not isinstance(value, (int, float)):
            errors.append({"field": f"body.{key}", "message": "Value must be a number.",
                           "code": "non_numeric"})
        elif not math.isfinite(float(value)):
            errors.append({"field": f"body.{key}", "message": "Value must be finite.",
                           "code": "non_finite"})
        elif float(value) < 0:
            errors.append({"field": f"body.{key}", "message": "Value must not be negative.",
                           "code": "negative"})

    if errors:
        raise MeasurementError(errors)
    return SourceProfile(
        values_cm={name: float(body[name]) for name in index if name in body},
        filename=filename,
    )


def source_rows(profile: SourceProfile) -> list[dict]:
    """Every imported value exactly as it appears in the file, for operator review."""
    index = measurement_index()
    rows = []
    for name, spec in index.items():
        if name not in profile.values_cm:
            continue
        rows.append({
            "name": name,
            "label": spec["label"],
            "value": profile.cm(name),
            "unit": spec.get("unit", "cm"),
            "valueMm": profile.mm(name),
            "kind": spec["kind"],
            "origin": spec["origin"],
            "direction": spec["direction"],
            "landmark": spec["landmark"],
            "interpretation": " ".join(spec["interpretation"].split()),
            "min": spec.get("min"),
            "max": spec.get("max"),
            "usedBy": spec["used_by"],
        })
    return rows


def dictionary_summary() -> dict:
    from ..config import measurement_dictionary
    doc = measurement_dictionary()
    return {"unit": doc["units"], "checks": doc["checks"], "fieldCount": len(doc["fields"])}
