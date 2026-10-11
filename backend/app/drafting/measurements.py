"""Typed BodyMeasurements model and the YAML import pipeline."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import yaml

from ..config import measurement_index, measurement_dictionary

CM_TO_MM = 10.0


class MeasurementError(Exception):
    """Field-specific import failure, surfaced by the API as structured errors."""

    def __init__(self, errors: list[dict]) -> None:
        self.errors = errors
        super().__init__("; ".join(f"{e['field']}: {e['message']}" for e in errors))


@dataclass
class MeasurementWarning:
    code: str
    field: str | None
    message: str
    original_value: Any = None
    action: str = "review"


@dataclass
class BodyMeasurements:
    """Confirmed centimetre values, kept separate from every derived dimension."""

    values_cm: dict[str, float]
    values_mm: dict[str, float]
    unit: str = "cm"
    warnings: list[MeasurementWarning] = field(default_factory=list)

    def mm(self, name: str) -> float:
        return self.values_mm[name]

    def cm(self, name: str) -> float:
        return self.values_cm[name]

    def rule_inputs(self) -> dict[str, float]:
        """Registry inputs: lengths in millimetres, angles kept in degrees so the
        rule registry never sees 45 deg converted to 450 mm."""
        return {
            name: (self.values_cm[name] if spec["kind"] == "angle_deg" else self.values_mm[name])
            for name, spec in measurement_index().items()
        }


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def load_yaml_document(text: str, filename: str) -> dict:
    if not filename.lower().endswith((".yaml", ".yml")):
        raise MeasurementError(
            [{"field": "file", "message": "Only .yaml and .yml files are accepted.", "code": "bad_extension"}]
        )
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise MeasurementError(
            [{"field": "file", "message": f"YAML parse error: {exc}", "code": "yaml_parse_error"}]
        ) from exc
    if not isinstance(data, dict):
        raise MeasurementError(
            [{"field": "file", "message": "Top level of the document must be a mapping.", "code": "not_a_mapping"}]
        )
    return data


def parse_measurements(text: str, filename: str = "body_measurements.yaml") -> BodyMeasurements:
    document = load_yaml_document(text, filename)
    index = measurement_index()
    errors: list[dict] = []

    body = document.get("body")
    if body is None:
        raise MeasurementError(
            [{"field": "body", "message": "Top level 'body' mapping is required.", "code": "missing_body"}]
        )
    if not isinstance(body, dict):
        raise MeasurementError(
            [{"field": "body", "message": "'body' must be a mapping.", "code": "body_not_mapping"}]
        )

    for name in index:
        if name not in body:
            errors.append(
                {"field": f"body.{name}", "message": "Required measurement is missing.", "code": "missing_field"}
            )

    for key, value in body.items():
        if key not in index:
            errors.append(
                {"field": f"body.{key}", "message": "Unknown measurement field.", "code": "unknown_field"}
            )
            continue
        if value is None:
            errors.append({"field": f"body.{key}", "message": "Value must not be null.", "code": "null_value"})
        elif isinstance(value, bool):
            errors.append({"field": f"body.{key}", "message": "Value must not be a boolean.", "code": "bool_value"})
        elif isinstance(value, str):
            errors.append({"field": f"body.{key}", "message": "Value must not be a string.", "code": "string_value"})
        elif not _is_number(value):
            errors.append({"field": f"body.{key}", "message": "Value must be a number.", "code": "non_numeric"})
        elif not math.isfinite(float(value)):
            errors.append({"field": f"body.{key}", "message": "Value must be finite.", "code": "non_finite"})
        elif float(value) < 0:
            errors.append({"field": f"body.{key}", "message": "Value must not be negative.", "code": "negative"})

    if errors:
        raise MeasurementError(errors)

    values_cm = {name: float(body[name]) for name in index}
    warnings = _plausibility_warnings(values_cm, index)
    return BodyMeasurements(
        values_cm=values_cm,
        values_mm={name: v * CM_TO_MM for name, v in values_cm.items()},
        warnings=warnings,
    )


def _plausibility_warnings(values: dict[str, float], index: dict[str, dict]) -> list[MeasurementWarning]:
    warnings: list[MeasurementWarning] = []
    for name, spec in index.items():
        value = values[name]
        lo, hi = spec.get("min"), spec.get("max")
        if lo is not None and value < lo:
            warnings.append(
                MeasurementWarning(
                    "value_below_range",
                    name,
                    f"{spec['label']} is {value} {index[name].get('unit', 'cm')}, below the plausible "
                    f"range {lo}-{hi}. The value is used unchanged.",
                    value,
                )
            )
        if hi is not None and value > hi:
            warnings.append(
                MeasurementWarning(
                    "value_above_range",
                    name,
                    f"{spec['label']} is {value}, above the plausible range {lo}-{hi}. "
                    f"The value is used unchanged.",
                    value,
                )
            )

    def add(code: str, message: str) -> None:
        warnings.append(MeasurementWarning(code, None, message))

    if values["underbust"] > values["bust"] - 10.0:
        add("underbust_vs_bust", "Underbust is not clearly smaller than bust; check the scan.")
    if values["vert_bust_line"] <= values["bust_line"]:
        add(
            "bust_apex_below_bust_line",
            "vert_bust_line is not below bust_line although both use the neck-point origin.",
        )
    if values["waist_line"] - values["bust_line"] < 80.0:
        add("waist_below_bust", "Bust-to-waist distance is under 8 cm; check waist_line.")
    for width_key, girth_key in (("back_width", "bust"), ("waist_back_width", "waist")):
        share = values[width_key] / values[girth_key] if values[girth_key] else 0.0
        if not 0.30 <= share <= 0.60:
            add(
                "back_share_of_bust",
                f"{width_key} takes {share:.0%} of {girth_key}; expected 30-60%.",
            )
    return warnings


def review_rows(body: BodyMeasurements) -> list[dict]:
    """Every imported value, with its dictionary entry, for the confirmation step."""
    index = measurement_index()
    rows = []
    for name, spec in index.items():
        rows.append(
            {
                "name": name,
                "label": spec["label"],
                "value": body.cm(name),
                "unit": spec.get("unit", "cm"),
                "valueMm": body.mm(name),
                "kind": spec["kind"],
                "origin": spec["origin"],
                "direction": spec["direction"],
                "landmark": spec["landmark"],
                "interpretation": " ".join(spec["interpretation"].split()),
                "min": spec.get("min"),
                "max": spec.get("max"),
                "usedBy": spec["used_by"],
            }
        )
    return rows


def dictionary_summary() -> dict:
    doc = measurement_dictionary()
    return {"unit": doc["units"], "checks": doc["checks"], "fieldCount": len(doc["fields"])}
