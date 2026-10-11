"""Drafting pipeline: YAML -> validation -> ease -> dimensions -> panels -> JSON."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from ..config import drafting_config
from ..validation.geometry_checks import ValidationReport, validate
from .cutting import CuttingLine, build_cutting_lines
from .derived_dimensions import build_registry, input_warnings, registry_inputs
from .ease import Ease, Settings, resolve_ease
from .geometry import PatternDocument
from .measurements import BodyMeasurements, parse_measurements
from .panels import build_document
from .rules import RuleTrace


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


@lru_cache(maxsize=1)
def _registry():
    return build_registry()


def draft(body: BodyMeasurements, settings: Settings) -> DraftResult:
    ease = resolve_ease(settings)
    inputs = registry_inputs(body, ease, settings)
    values, traces = _registry().evaluate(inputs)
    warnings = [w.__dict__ for w in (list(body.warnings) + input_warnings(body, inputs, values))]
    doc = build_document(values, settings, ease)
    cutting = build_cutting_lines(doc, settings)
    report = validate(doc, cutting)
    return DraftResult(doc, traces, values, report, cutting, warnings, settings, ease)


def generate_from_yaml(text: str, filename: str, settings: Settings | None = None) -> DraftResult:
    body = parse_measurements(text, filename)
    return draft(body, settings or Settings())


def manifest(result: DraftResult, body: BodyMeasurements, filename: str) -> dict:
    cfg = drafting_config()
    return {
        "method": "formula-drafting",
        "ruleRegistryVersion": _registry().version,
        "sourceFile": filename,
        "units": {"input": body.unit, "geometry": "mm"},
        "settings": result.settings.as_dict(),
        "ease": {
            "preset": result.ease.preset,
            "bustMm": result.ease.bust_mm,
            "waistMm": result.ease.waist_mm,
            "hipMm": result.ease.hip_mm,
        },
        "assumptions": result.warnings,
        "tolerances": cfg["tolerances"],
        "fixtureDefaults": cfg["fixture_defaults"],
        "rules": [t.as_dict() for t in result.traces],
        "validation": result.report.as_dict()["summary"],
    }
