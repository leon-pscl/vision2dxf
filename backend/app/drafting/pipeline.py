"""Drafting pipeline.

    YAML -> syntax -> semantics resolution -> normalization -> consistency
         -> operator confirmation -> DraftingMeasurements -> rules -> panels -> JSON

The drafting engine only ever sees `DraftingMeasurements`.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from ..config import drafting_config
from ..validation.geometry_checks import ValidationReport, validate
from .cutting import CuttingLine, build_cutting_lines
from .derived_dimensions import build_registry, input_warnings, registry_inputs
from .ease import Ease, Settings, resolve_ease
from .geometry import PatternDocument
from .measurements import SourceProfile, parse_source
from .panels import build_document
from .rules import RuleTrace
from .semantics import Resolution, resolve

SOFTWARE_VERSION = "1.0.0"


@dataclass
class DraftResult:
    document: PatternDocument
    traces: list[RuleTrace]
    values: dict[str, float]
    report: ValidationReport
    cutting: dict[str, CuttingLine]
    warnings: list[dict]
    settings: Settings
    ease: Ease
    resolution: Resolution


_registry = None


def registry():
    global _registry
    if _registry is None:
        _registry = build_registry()
    return _registry


def draft(resolution: Resolution, settings: Settings) -> DraftResult:
    if not resolution.can_generate or resolution.measurements is None:
        raise SemanticsBlocked(resolution)
    ease = resolve_ease(settings)
    inputs = registry_inputs(resolution.measurements, ease, settings)
    values, traces = registry().evaluate(inputs)
    warnings = [w.__dict__ for w in input_warnings(resolution.measurements, values)]
    doc = build_document(values, settings, ease)
    cutting = build_cutting_lines(doc, settings)
    report = validate(doc, cutting)
    return DraftResult(doc, traces, values, report, cutting, warnings, settings, ease, resolution)


class SemanticsBlocked(Exception):
    def __init__(self, resolution: Resolution) -> None:
        self.resolution = resolution
        super().__init__("generation blocked by unresolved measurement semantics")


def resolve_semantics(
    text: str,
    filename: str,
    confirmations: dict[str, str] | None = None,
    corrections: dict[str, float] | None = None,
) -> tuple[SourceProfile, Resolution]:
    profile = parse_source(text, filename)
    return profile, resolve(profile.values_cm, confirmations, corrections)


def manifest(result: DraftResult, profile: SourceProfile, filename: str) -> dict:
    cfg = drafting_config()
    res = result.resolution
    reg = registry()
    return {
        "measurement_schema_version": res.schema_version,
        "definition_version": res.definition_version,
        "software_version": SOFTWARE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": "formula-drafting",
        "rule_registry_version": reg.version,
        "source_file": filename,
        "units": {"source": "cm", "pattern": "mm"},
        "coordinate_convention": {
            "vertical_origin": res.coordinate_origin,
            "vertical_direction": "downward",
            "pattern_frame_y_zero": "neck_reference_point",
        },
        "field_interpretations": {
            name: {
                "meaning": f.drafting_variable,
                "status": f.status.lower(),
                "confirmed_by": f.confirmed_by or "unconfirmed",
                "candidate": f.candidate_id,
                "registry_status": f.registry_status,
                "definition_version": f.definition_version,
            }
            for name, f in res.fields.items()
            if f.status in ("CONFIRMED", "ASSUMED", "DERIVED")
        },
        "field_records": [f.as_dict() for f in res.fields.values()],
        "derived_measurements": res.derived,
        "drafting_measurements_mm": (
            result.resolution.measurements.model_dump() if res.measurements else {}
        ),
        "assumptions": res.assumptions,
        "warnings": res.warnings + result.warnings,
        "settings": result.settings.as_dict(),
        "ease": {
            "preset": result.ease.preset,
            "bustMm": result.ease.bust_mm,
            "waistMm": result.ease.waist_mm,
            "hipMm": result.ease.hip_mm,
        },
        "tolerances": cfg["tolerances"],
        "fixture_defaults": cfg["fixture_defaults"],
        "rules": [t.as_dict() for t in result.traces],
        "validation": result.report.as_dict()["summary"],
    }
