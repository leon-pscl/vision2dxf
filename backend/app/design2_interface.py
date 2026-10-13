"""The common Design 2 program interface.

    generate_pattern(measurements: dict, garment: dict) -> dict

`measurements` is the tailor-confirmed source mapping (`{"body": {...}}`, or a bare body
mapping), `garment` the garment configuration. The return is the common pattern object in
millimetres: the same panels, boundaries, cutting lines, grainlines, fold edges, stitch
pairs, validation report and provenance every Design 2 consumer reads.

This module is an adapter, not a second implementation. No geometry, weight or tolerance
is defined here, so `config/transformation.yaml` still governs every number that reaches
the pattern and the frozen configuration stays the only source of truth.

`run_design_2` is the benchmark protocol's timed and memory-measured boundary: generate,
validate with the common validator, export with the common DXF exporter. Doing it in one
call rather than three is not a shortcut - the same three functions run, in the same
order, on the same objects.

Every refusal leaves as a `Design2Failure` carrying one code from `FAILURE_CODES`. The
pipeline's own refusal codes stay untouched, because they are what the HTTP API reports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .base_block.measurements import default_confirmations
from .config import drafting_config, transformation_config
from .drafting.ease import Settings
from .drafting.geometry import PatternDocument, PatternPanel, world_point
from .drafting.measurements import MeasurementError
from .export.dxf import export_dxf
from .schemas.garmage import GarmageSet
from .serialization.garmageset import to_fixture_dict
from .transformation.pipeline import (
    SOFTWARE_VERSION,
    BaseBlockError,
    TransformationBlocked,
    manifest,
    resolve_semantics,
    transform,
)

SCHEMA_NAME = "design2-common-pattern"
SCHEMA_VERSION = "1.0.0"
DESIGN_NAME = "Design 2 - GarmageSet-Based Parametric Base-Block Transformation"

# The failure codes the benchmark protocol accepts. Nothing else escapes as a refusal.
FAILURE_CODES = (
    "MISSING_MEASUREMENT",
    "OUT_OF_DOMAIN",
    "BOUND_VIOLATION",
    "INVALID_GEOMETRY",
    "SOLVER_DID_NOT_CONVERGE",
    "BASE_BLOCK_UNAVAILABLE",
)

# Pipeline refusals carry their own codes; the protocol only needs one it can aggregate on.
_BLOCKED_CODES = {
    # Outside the tested envelope. `envelope_exceeded` is past `stop_factor` and is
    # refused however the operator confirms it; `envelope_confirmation_required` is
    # inside the tested range but past the declared one.
    "envelope_exceeded": "OUT_OF_DOMAIN",
    "envelope_confirmation_required": "OUT_OF_DOMAIN",
    # A difference the fixed base block has no landmark to express.
    "delta_not_mapped": "OUT_OF_DOMAIN",
    "curve_loop": "INVALID_GEOMETRY",
    "curve_spike": "INVALID_GEOMETRY",
    "solver_failed": "SOLVER_DID_NOT_CONVERGE",
    "not_converged": "SOLVER_DID_NOT_CONVERGE",
    # The solve finished but paired seams would not settle inside the seam tolerance, so
    # the achieved geometry violates a declared bound rather than being malformed.
    "seam_balance_settled": "BOUND_VIOLATION",
    "invalid_base_block": "BASE_BLOCK_UNAVAILABLE",
    "base_block_unavailable": "BASE_BLOCK_UNAVAILABLE",
}

# A source file that never carried the measurements, as opposed to one that does but with a
# value outside the declared range.
_UNREADABLE_SOURCE_CODES = {
    "bad_extension", "yaml_parse_error", "not_a_mapping", "missing_body", "body_not_mapping",
}

# Common-validator check id -> protocol code, by prefix. Check ids carry the panel or
# seam key after a colon (`seam_mismatch:66ced50f`), so the match is on the prefix.
_VALIDATION_CODES = (
    ("seam_mismatch", "BOUND_VIOLATION"),
    ("seam_", "BOUND_VIOLATION"),
    ("achieved_dimension", "BOUND_VIOLATION"),
    ("envelope", "OUT_OF_DOMAIN"),
    ("unmapped", "OUT_OF_DOMAIN"),
    ("stitch_", "INVALID_GEOMETRY"),
    ("dart_", "INVALID_GEOMETRY"),
    ("self_intersection", "INVALID_GEOMETRY"),
    ("panel_closure", "INVALID_GEOMETRY"),
    ("cutting_lines", "INVALID_GEOMETRY"),
    ("finite_coordinates", "INVALID_GEOMETRY"),
)


class Design2Failure(Exception):
    """A structured refusal. `code` is always one of `FAILURE_CODES`."""

    def __init__(self, code: str, message: str, details: list[dict] | None = None) -> None:
        self.code = code
        self.message = message
        self.details = details or []
        super().__init__(f"{code}: {message}")

    def as_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "details": self.details}


@dataclass
class Design2Run:
    """One complete generation, validation and export."""

    pattern: dict
    report: dict
    document: PatternDocument
    cutting: dict
    diagnostics: dict = field(default_factory=dict)
    dxf_path: str | None = None


# ---- the interface ---------------------------------------------------------


def generate_pattern(measurements: dict, garment: dict | None = None) -> dict:
    """Transform the fixed base block and return the common pattern object in mm."""
    return run_design_2(measurements, garment).pattern


def run_design_2(
    measurements: dict,
    garment: dict | None = None,
    dxf_output: str | Path | None = None,
) -> Design2Run:
    """The measured boundary: generate, validate with the common validator, export DXF."""
    source_text = _to_source_text(measurements)
    settings = _to_settings(garment)

    profile, resolution = _resolve(source_text)
    result = _transform(resolution, settings)

    pattern = _pattern_object(result, profile)
    report = result.report.as_dict()
    if report["status"] != "pass":
        raise Design2Failure(_validation_code(report), _validation_message(report),
                             [c for c in report["checks"] if c["status"] == "fail"])

    run = Design2Run(
        pattern=pattern,
        report=report,
        document=result.document,
        cutting=result.cutting,
        diagnostics=_transformation_diagnostics(result, profile),
    )
    if dxf_output is not None:
        run.dxf_path = export_common_dxf(run, dxf_output)
    return run


def export_common_dxf(run: Design2Run, output: str | Path) -> str:
    """Export the common pattern with the shared DXF exporter. Returns the path written."""
    path = str(output)
    Path(path).write_bytes(export_dxf(run.document, run.cutting))
    return path


# ---- input adaptation ------------------------------------------------------


def _to_source_text(measurements: dict) -> str:
    """The source YAML text, from either `{"body": {...}}` or a bare body mapping."""
    if not isinstance(measurements, dict) or not measurements:
        raise Design2Failure("MISSING_MEASUREMENT",
                             "The measurements argument is empty or not a mapping.",
                             [_detail("measurements", "The generation interface takes the "
                              "confirmed source mapping, not a file path.")])
    body = measurements.get("body", measurements)
    if not isinstance(body, dict) or not body:
        raise Design2Failure("MISSING_MEASUREMENT", "The 'body' mapping is missing or empty.",
                             [_detail("body", "Load a measurement fixture that carries a "
                              "non-empty 'body' mapping.")])
    return yaml.safe_dump({"body": body}, sort_keys=True, allow_unicode=True)


def _to_settings(garment: dict | None) -> Settings:
    """Garment configuration onto `Settings`.

    Unknown keys are refused rather than dropped: a garment configuration the engine does
    not recognise means the caller and the engine disagree about what is being made, and
    silently ignoring it would benchmark something other than the request.
    """
    unknown = sorted(set(garment or {}) - set(Settings.__dataclass_fields__))
    if unknown:
        raise Design2Failure(
            "OUT_OF_DOMAIN",
            f"Unknown garment configuration key(s): {', '.join(unknown)}.",
            [_detail(name, "Not a garment configuration field. See Settings.as_dict().")
             for name in unknown],
        )
    # `base_block_id` stays as given; None resolves to the catalog default. Which block is
    # frozen is a benchmark decision and lives in benchmark/config-design-2.yaml, not here.
    return Settings(**(garment or {}))


def _detail(field: str, message: str) -> dict:
    """A refusal detail in the same shape the pipeline's own stages emit."""
    return {
        "field": field,
        "code": "MISSING_MEASUREMENT",
        "message": " ".join(message.split()),
        "suggestedAction": message,
        "severity": "blocking",
    }


def _resolve(source_text: str):
    # The benchmark loads an already-approved fixture rather than asking an operator, so
    # it supplies the registry's own default confirmations. A caller that has not actually
    # confirmed those interpretations is bypassing the review step, not satisfying it.
    try:
        return resolve_semantics(source_text, "measurements.yaml", default_confirmations())
    except MeasurementError as exc:
        code = (
            "MISSING_MEASUREMENT"
            if exc.errors[0].get("code") in _UNREADABLE_SOURCE_CODES
            else "OUT_OF_DOMAIN"
        )
        raise Design2Failure(code, "The measurement file could not be read.", exc.errors) from exc


def _transform(resolution, settings: Settings):
    try:
        return transform(resolution, settings)
    except TransformationBlocked as exc:
        raise Design2Failure(_blocked_code(exc), exc.message, exc.details) from exc
    except BaseBlockError as exc:
        raise Design2Failure("BASE_BLOCK_UNAVAILABLE",
                             "The versioned base block could not be loaded.",
                             exc.errors) from exc


def _blocked_code(exc: TransformationBlocked) -> str:
    if exc.code == "semantics_unresolved":
        # `INVALID` means a value is present and outside the declared range; anything else
        # means the confirmed value never arrived, which is a different failure.
        return (
            "OUT_OF_DOMAIN"
            if any(d.get("code") == "INVALID" for d in exc.details)
            else "MISSING_MEASUREMENT"
        )
    return _BLOCKED_CODES.get(exc.code, "OUT_OF_DOMAIN")


def _validation_code(report: dict) -> str:
    for check in report["checks"]:
        if check["status"] != "fail":
            continue
        for prefix, code in _VALIDATION_CODES:
            if check["id"].startswith(prefix):
                return code
    return "INVALID_GEOMETRY"


def _validation_message(report: dict) -> str:
    failed = [c["id"] for c in report["checks"] if c["status"] == "fail"]
    return f"The common validator rejected the pattern: {', '.join(failed)}."


# ---- the common pattern object ---------------------------------------------


def _pattern_object(result, profile) -> dict:
    doc = result.document
    garmageset = to_fixture_dict(doc)
    GarmageSet.model_validate(garmageset)  # never emit output the shared schema rejects
    return {
        "schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION, "units": "mm"},
        "metadata": _metadata(result, profile),
        "panels": [_panel(panel) for panel in doc.panels],
        "cuttingLines": {
            panel.key: _cutting(result.cutting[panel.key], panel)
            for panel in doc.panels if panel.key in result.cutting
        },
        "stitches": [
            {
                "id": f"stitch-{i + 1}",
                "records": [
                    {
                        "panelId": record.panel_id,
                        "startEdgeId": record.start_edge_id,
                        "startParam": record.start_param,
                        "endEdgeId": record.end_edge_id,
                        "endParam": record.end_param,
                        "isCounterClockwise": record.is_counter_clockwise,
                    }
                    for record in pair.records
                ],
            }
            for i, pair in enumerate(doc.stitches)
        ],
        "garmageset": garmageset,
        "validation": result.report.as_dict(),
        "provenance": _provenance(result, profile),
    }


def _metadata(result, profile) -> dict:
    return {
        "design": DESIGN_NAME,
        "softwareVersion": SOFTWARE_VERSION,
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sourceFile": profile.filename,
        "units": {"source": result.settings.unit, "pattern": "mm"},
        "method": "base-block-transformation",
        "coordinateConvention": {
            "verticalOrigin": "neck reference point",
            "verticalDirection": "+y downward",
            "patternFrame": "panel-local millimetres, placed by panel center",
        },
        "garmentType": result.settings.garment_type,
        "baseBlockId": result.block.id,
        "easePreset": result.ease.preset,
    }


def _pt(point) -> list[float]:
    """A point as plain floats, so the common pattern object survives `json.dumps`."""
    return [float(point.x), float(point.y)]


def _panel(panel: PatternPanel) -> dict:
    spacing = drafting_config()["sample_spacing_mm"]
    fold_edges = [edge.ensure_id() for edge in panel.edges if edge.seam_allowance_mm == 0.0]
    return {
        "id": panel.ensure_id(),
        "key": panel.key,
        "label": panel.label,
        "center": _pt(panel.center) if panel.center else [0.0, 0.0],
        "areaMm2": _area(panel),
        "foldEdgeIds": fold_edges,
        "edges": [
            {
                "id": edge.ensure_id(),
                "role": edge.role,
                "label": edge.label,
                "ruleId": edge.rule_id,
                "isFold": edge.seam_allowance_mm == 0.0,
                "seamAllowanceMm": float(edge.seam_allowance_mm),
                "lengthMm": float(edge.length_mm),
                "pairedEdgeId": edge.paired_edge_id,
                "start": _pt(edge.start),
                "end": _pt(edge.end),
                "points": [_pt(p) for p in edge.polyline(spacing)],
                "notches": [
                    {"at": float(notch.at), "kind": notch.kind, "role": notch.role}
                    for notch in edge.notches
                ],
            }
            for edge in panel.edges
        ],
        "grainline": (
            [_pt(panel.grainline[0]), _pt(panel.grainline[1])] if panel.grainline else None
        ),
        "landmarks": {name: _pt(p) for name, p in panel.landmarks.items()},
        # Internal reference lines, each carried as a (start, end) segment pair.
        "construction": [[_pt(a), _pt(b)] for a, b in panel.construction],
        "darts": [
            {
                "id": dart.id,
                "ruleId": dart.rule_id,
                "intakeMm": float(dart.intake_mm),
                "apex": _pt(dart.apex),
                "contours": [[_pt(p) for p in contour] for contour in dart.contours],
            }
            for dart in panel.darts
        ],
    }


def _cutting(line, panel: PatternPanel) -> dict:
    return {
        "panelId": panel.ensure_id(),
        "join": line.join,
        "closed": line.closed,
        "contour": [_pt(p) for p in line.contour],
        "dartContours": [[_pt(p) for p in c] for c in line.dart_contours],
        "seamAllowancesMm": {edge.role: float(edge.seam_allowance_mm) for edge in panel.edges},
    }


def _area(panel: PatternPanel) -> float:
    """Shoelace area of the sewing-line boundary, in square millimetres."""
    points = [world_point(panel, edge.start) for edge in panel.edges]
    total = 0.0
    for i, a in enumerate(points):
        b = points[(i + 1) % len(points)]
        total += a.x * b.y - b.x * a.y
    return abs(total) / 2.0


def _transformation_diagnostics(result, profile) -> dict:
    """Per-panel solver outcome, as the benchmark reports it per trial."""
    diagnostics = result.diagnostics()
    solver = diagnostics.get("solver", {})
    return {
        "configVersion": diagnostics.get("transformationConfigVersion"),
        "influenceMapVersion": diagnostics.get("influenceMapVersion"),
        "method": diagnostics.get("method"),
        "baseBlock": diagnostics.get("baseBlock"),
        "seamBalanceRounds": diagnostics.get("seamBalanceRounds"),
        "seamMismatchAfterMm": diagnostics.get("seamMismatchAfterMm"),
        "panels": [
            {
                "panel": name,
                "method": entry.get("method"),
                "landmarks": entry.get("landmarkCount"),
                "iterations": entry.get("iterations"),
                "converged": entry.get("converged"),
                "lambda": entry.get("lambda"),
                "maxResidualMm": entry.get("maxResidualMm"),
                "meanResidualMm": entry.get("meanResidualMm"),
                "conditionNumber": entry.get("conditionNumber"),
                "constraintViolations": entry.get("constraintViolations", []),
                "mergedLandmarks": entry.get("mergedLandmarks", []),
            }
            for name, entry in sorted(solver.items())
        ],
        "unmappedDeltas": [
            {"measurement": d["name"], "deltaMm": d["deltaValue"]}
            for d in diagnostics.get("deltas", []) if not d.get("mapped") and d["deltaValue"]
        ],
        "warnings": diagnostics.get("warnings", []),
    }


def _provenance(result, profile) -> dict:
    """Where every number came from: schema version, frozen config, the base block, progress."""
    return {
        "measurementSchemaVersion": result.resolution.schema_version,
        "definitionVersion": result.resolution.definition_version,
        "transformationConfigVersion": transformation_config()["version"],
        "draftingConfigVersion": drafting_config()["version"],
        "baseBlock": {
            "id": result.block.id,
            "version": result.block.version,
            "label": result.block.label,
            "reviewed": result.block.reviewed,
        },
        "settings": result.settings.as_dict(),
        "ease": dict(result.ease.__dict__),
        "semanticAssumptions": result.resolution.assumptions,
        "derivedMeasurements": result.resolution.derived,
        "generationManifest": manifest(result, profile, profile.filename),
    }