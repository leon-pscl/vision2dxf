"""The common Design 3 program interface.

    generate_pattern(measurements: dict, garment: dict) -> dict

`measurements` is the tailor-confirmed source mapping (`{"body": {...}}`, or a bare body
mapping), `garment` the garment configuration. The return is the common pattern object in
millimetres: the same panels, boundaries, cutting lines, grainlines, fold edges, stitch
pairs, validation report and provenance every Design 3 consumer reads.

This module is an adapter, not a second implementation. No geometry, solver, weight or
tolerance is defined here, so `config/reconstruction.yaml` still governs every number that
reaches the pattern and the frozen configuration stays the only source of truth.

`run_design_3` is the benchmark protocol's timed and memory-measured boundary: generate,
validate with the common validator, export with the common DXF exporter. Doing it in one
call rather than three is not a shortcut - the same three functions run, in the same
order, on the same objects.

Every refusal leaves as a `Design3Failure` carrying one code from `FAILURE_CODES`. The
pipeline's own stage codes stay untouched, because they are what the HTTP API reports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .config import drafting_config, reconstruction_config
from .drafting.ease import Settings
from .drafting.geometry import PatternDocument, PatternPanel, world_point
from .drafting.measurements import MeasurementError
from .drafting.semantics import default_confirmations
from .export.dxf import export_dxf
from .pipeline import (
    SOFTWARE_VERSION,
    GenerationBlocked,
    GenerationResult,
    generate,
    resolve_semantics,
)
from .schemas.garmage import GarmageSet
from .serialization.garmageset import to_fixture_dict

SCHEMA_NAME = "design3-common-pattern"
SCHEMA_VERSION = "1.0.0"
DESIGN_NAME = "Design 3 - Anthropometric Surface Reconstruction and Constrained Mesh Flattening"

# The failure codes the benchmark protocol accepts. Nothing else escapes as a refusal.
FAILURE_CODES = (
    "MISSING_MEASUREMENT",
    "INVALID_SECTION_ORDER",
    "MESH_INVALID",
    "NONDISK_PANEL",
    "SOLVER_DID_NOT_CONVERGE",
    "FLIPPED_TRIANGLES",
    "SEAM_MISMATCH",
    "BOUNDARY_INVALID",
    "OUT_OF_DOMAIN",
)

# Stage refusals carry their own codes; the protocol only needs one it can aggregate on.
_BLOCKED_CODES = {
    "units_unconfirmed": "OUT_OF_DOMAIN",
    "semantics_unresolved": "MISSING_MEASUREMENT",
    "anatomical_levels_inconsistent": "INVALID_SECTION_ORDER",
    "cross_section_unsolvable": "MESH_INVALID",
    "proxy_surface_invalid": "MESH_INVALID",
    "garment_surface_invalid": "MESH_INVALID",
    "seam_network_invalid": "OUT_OF_DOMAIN",
    "panel_segmentation_invalid": "NONDISK_PANEL",
    "unknown_flattening_method": "OUT_OF_DOMAIN",
    "boundary_refit_failed": "BOUNDARY_INVALID",
}

# A flattening refusal arrives as per-panel detail codes, so the first one decides.
_FLATTENING_DETAIL_CODES = {
    "FLIPPED_TRIANGLES": "FLIPPED_TRIANGLES",
    "FLATTENING_NOT_CONVERGED": "SOLVER_DID_NOT_CONVERGE",
    "BOUNDARY_SELF_INTERSECTION": "BOUNDARY_INVALID",
}

# Common-validator check id -> protocol code, by prefix. Anything unrecognised is a
# boundary problem, which is what an unresolved reference or a broken contour both are.
_VALIDATION_CODES = (
    ("seam_mismatch", "SEAM_MISMATCH"),
    ("stitch_", "SEAM_MISMATCH"),
    ("dart_containment", "BOUNDARY_INVALID"),
    ("self_intersection", "BOUNDARY_INVALID"),
    ("panel_closure", "BOUNDARY_INVALID"),
    ("cutting_lines", "BOUNDARY_INVALID"),
    ("finite_coordinates", "MESH_INVALID"),
)


class Design3Failure(Exception):
    """A structured refusal. `code` is always one of `FAILURE_CODES`."""

    def __init__(self, code: str, message: str, details: list[dict] | None = None) -> None:
        self.code = code
        self.message = message
        self.details = details or []
        super().__init__(f"{code}: {message}")

    def as_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "details": self.details}


@dataclass
class Design3Run:
    """One complete generation, validation and export."""

    pattern: dict
    report: dict
    document: PatternDocument
    cutting: dict
    diagnostics: dict = field(default_factory=dict)
    dxf_path: str | None = None


# ---- the interface ---------------------------------------------------------


def generate_pattern(measurements: dict, garment: dict | None = None) -> dict:
    """Reconstruct, segment, flatten, and return the common pattern object in mm."""
    return run_design_3(measurements, garment).pattern


def run_design_3(
    measurements: dict,
    garment: dict | None = None,
    dxf_output: str | Path | None = None,
) -> Design3Run:
    """The measured boundary: generate, validate with the common validator, export DXF."""
    source_text = _to_source_text(measurements)
    settings = _to_settings(garment)

    profile, resolution = _resolve(source_text)
    result = _generate(resolution, profile, settings)

    pattern = _pattern_object(result)
    report = result.report.as_dict()
    if report["status"] != "pass":
        raise Design3Failure(_validation_code(report), _validation_message(report),
                             [c for c in report["checks"] if c["status"] == "fail"])

    run = Design3Run(
        pattern=pattern,
        report=report,
        document=result.document,
        cutting=result.cutting,
        diagnostics=_mesh_diagnostics(result),
    )
    if dxf_output is not None:
        run.dxf_path = export_common_dxf(run, dxf_output)
    return run


def export_common_dxf(run: Design3Run, output: str | Path) -> str:
    """Export the common pattern with the shared DXF exporter. Returns the path written."""
    path = str(output)
    Path(path).write_bytes(export_dxf(run.document, run.cutting))
    return path


# ---- input adaptation ------------------------------------------------------


def _to_source_text(measurements: dict) -> str:
    """The source YAML text, from either `{"body": {...}}` or a bare body mapping."""
    if not isinstance(measurements, dict) or not measurements:
        raise Design3Failure("MISSING_MEASUREMENT",
                             "The measurements argument is empty or not a mapping.",
                             [_detail("measurements", "The generation interface takes the "
                              "confirmed source mapping, not a file path.")])
    body = measurements.get("body", measurements)
    if not isinstance(body, dict) or not body:
        raise Design3Failure("MISSING_MEASUREMENT", "The 'body' mapping is missing or empty.",
                             [_detail("body", "Load a measurement fixture that carries a "
                              "non-empty 'body' mapping.")])
    return yaml.safe_dump({"body": body}, sort_keys=True, allow_unicode=True)


_SETTINGS_FIELDS = frozenset(Settings.__dataclass_fields__) - {"confirm_units", "unit"}


def _to_settings(garment: dict | None) -> Settings:
    """Garment configuration onto `Settings`, with the unit already confirmed.

    The unit is confirmed here because the harness loads an already-approved fixture
    rather than asking an operator; a caller who has not actually confirmed it is
    bypassing the review step, not satisfying it.
    """
    unknown = sorted(set(garment or {}) - _SETTINGS_FIELDS)
    if unknown:
        raise Design3Failure(
            "OUT_OF_DOMAIN",
            f"Unknown garment configuration key(s): {', '.join(unknown)}.",
            [_detail(name, "Not a garment configuration field. See Settings.as_dict().")
             for name in unknown],
        )
    return Settings(confirm_units=True, **{k: v for k, v in (garment or {}).items()})


def _detail(field: str, message: str) -> dict:
    """A refusal detail in the same shape the pipeline's own stages emit."""
    return {
        "field": field,
        "code": "MISSING_MEASUREMENT",
        "message": " ".join(message.split()),
        "suggestedAction": message,
        "severity": "blocking",
    }


# A parse failure is either a file that never carried the measurements, or a value that is
# present and unusable. The second group is out of the supported domain, not missing.
_UNREADABLE_SOURCE_CODES = {
    "bad_extension", "yaml_parse_error", "not_a_mapping", "missing_body", "body_not_mapping",
}


def _resolve(source_text: str):
    try:
        return resolve_semantics(source_text, "measurements.yaml", default_confirmations())
    except MeasurementError as exc:
        code = (
            "MISSING_MEASUREMENT"
            if exc.errors[0].get("code") in _UNREADABLE_SOURCE_CODES
            else "OUT_OF_DOMAIN"
        )
        raise Design3Failure(code, "The measurement file could not be read.", exc.errors) from exc


def _generate(resolution, profile, settings: Settings) -> GenerationResult:
    try:
        return generate(resolution, profile, settings)
    except GenerationBlocked as exc:
        raise Design3Failure(_blocked_code(exc), exc.message, exc.details) from exc


def _blocked_code(exc: GenerationBlocked) -> str:
    if exc.code == "flattening_failed":
        for detail in exc.details:
            code = _FLATTENING_DETAIL_CODES.get(detail.get("code", ""))
            if code:
                return code
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
    return "BOUNDARY_INVALID"


def _validation_message(report: dict) -> str:
    failed = [c["id"] for c in report["checks"] if c["status"] == "fail"]
    return f"The common validator rejected the pattern: {', '.join(failed)}."


# ---- the common pattern object ---------------------------------------------


def _pattern_object(result: GenerationResult) -> dict:
    doc = result.document
    garmageset = to_fixture_dict(doc)
    GarmageSet.model_validate(garmageset)  # never emit output the shared schema rejects
    return {
        "schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION, "units": "mm"},
        "metadata": _metadata(result),
        "panels": [_panel(panel, result) for panel in doc.panels],
        "cuttingLines": {
            key: _cutting(result.cutting[key], panel) for key, panel in _by_key(doc).items()
            if key in result.cutting
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
        "provenance": _provenance(result),
    }


def _by_key(doc: PatternDocument) -> dict[str, PatternPanel]:
    return {panel.key: panel for panel in doc.panels}


def _metadata(result: GenerationResult) -> dict:
    return {
        "design": DESIGN_NAME,
        "softwareVersion": SOFTWARE_VERSION,
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sourceFile": result.profile.filename,
        "units": {"source": result.settings.unit, "pattern": "mm"},
        "coordinateConvention": {
            "verticalOrigin": "neck reference point",
            "verticalDirection": "+y downward",
            "ringAngleZero": "centre front",
            "patternFrame": "panel-local millimetres, placed by panel center",
        },
        "garmentType": result.settings.garment_type,
        "seamPreset": result.network.preset,
        "sectionModel": result.settings.section_model,
        "flatteningMethod": result.settings.flattening_method,
    }


def _pt(point) -> list[float]:
    """A panel-local point as plain floats. The flattened layout is a NumPy array, so
    without this the common pattern object would not survive `json.dumps`."""
    return [float(point.x), float(point.y)]


def _panel(panel: PatternPanel, result: GenerationResult) -> dict:
    spacing = drafting_config()["sample_spacing_mm"]
    fold_edges = [edge.ensure_id() for edge in panel.edges if edge.seam_allowance_mm == 0.0]
    return {
        "id": panel.ensure_id(),
        "key": panel.key,
        "label": panel.label,
        "center": _pt(panel.center) if panel.center else [0.0, 0.0],
        "areaMm2": float(_area(panel)),
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
            {"start": _pt(panel.grainline[0]), "end": _pt(panel.grainline[1])}
            if panel.grainline else None
        ),
        "landmarks": {name: _pt(p) for name, p in panel.landmarks.items()},
        "construction": [_pt(p) for p in panel.construction],
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
        "surfaceRows": next(
            (p.quality for p in result.panels if p.panel.key == panel.key), {}
        ),
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


def _mesh_diagnostics(result: GenerationResult) -> dict:
    """Per-panel mesh and flattening diagnostics, as the benchmark reports them."""
    panels = []
    for entry in result.panels:
        d = entry.distortion
        panels.append({
            "panel": entry.panel.key,
            "vertices": entry.quality.get("vertices"),
            "triangles": entry.quality.get("triangles"),
            "degenerateTriangles": entry.quality.get("degenerateTriangles"),
            "boundaryLoops": len(entry.panel.boundary_loops),
            "diskTopology": len(entry.panel.boundary_loops) == 1,
            "solver": entry.flattening.method,
            "converged": entry.flattening.converged,
            "iterations": entry.flattening.iterations,
            "finalEnergy": round(entry.flattening.energy, 9),
            "flippedTriangles": d["flippedTriangles"],
            "flippedPct": round(100.0 * d["flippedTriangles"] / max(d["triangles"], 1), 6),
            "meanEdgeDistortion": d["meanEdgeLengthDistortion"],
            "maxEdgeDistortion": d["maxEdgeLengthDistortion"],
            "meanAreaDistortion": d["meanAreaDistortion"],
            "maxAreaDistortion": d["maxAreaDistortion"],
            "boundaryIntersections": d["boundarySelfIntersections"],
            "surfaceAreaMm2": d["surfaceAreaMm2"],
            "flattenedAreaMm2": d["flattenedAreaMm2"],
        })
    return {
        "configVersion": reconstruction_config()["version"],
        "weightTotals": {
            "totalVertices": sum(p["vertices"] or 0 for p in panels),
            "totalTriangles": sum(p["triangles"] or 0 for p in panels),
        },
        "panels": panels,
    }


def _provenance(result: GenerationResult) -> dict:
    """Where every number came from: schema version, frozen config, assumptions, progress."""
    cfg = reconstruction_config()
    return {
        "measurementSchemaVersion": result.resolution.schema_version,
        "definitionVersion": result.resolution.definition_version,
        "reconstructionConfigVersion": cfg["version"],
        "draftingConfigVersion": drafting_config()["version"],
        "surfaceDescription": (
            "Anthropometric surrogate surface reconstructed from scalar measurements, "
            "not a complete body scan. Recorded depths, section shapes and landmark "
            "positions are assumptions, listed below and in generation-manifest.json."
        ),
        "settings": result.settings.as_dict(),
        "ease": result.ease.as_dict(),
        "assumptions": result.assumptions.as_list(),
        "semanticAssumptions": result.resolution.assumptions,
        "derivedMeasurements": result.resolution.derived,
        "seamBalance": result.seam_balance,
        "progress": result.progress,
    }