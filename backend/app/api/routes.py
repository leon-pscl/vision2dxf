"""HTTP API. Structured errors only, never a raw stack trace.

Generation is gated: a request that leaves any required field MISSING, AMBIGUOUS or
INVALID is refused with the blocking schema messages, not an informational note.
"""
from __future__ import annotations

import io
import json
import zipfile

from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response

from ..config import drafting_config
from ..drafting.ease import Settings
from ..drafting.measurements import (
    MeasurementError,
    dictionary_summary,
    parse_source,
    source_rows,
)
from ..drafting.pipeline import draft, manifest, resolve_semantics
from ..drafting.semantics import registry as semantics_registry
from ..export.dxf import export_dxf
from ..export.pdf import export_pdf
from ..rendering.png import render_png
from ..rendering.svg import VIEW_LAYERS, RenderOptions, document_extent, render_svg
from ..schemas.api import ExportRequest, GenerateRequest, RenderRequest, SemanticsRequest, UploadRequest
from ..schemas.garmage import GarmageSet
from ..serialization.garmageset import to_fixture_dict

router = APIRouter(prefix="/api")

BLOKED_STATUS = 409


def _settings(payload) -> Settings:
    return Settings(**payload.model_dump())


def _semantics(req):
    try:
        return resolve_semantics(req.yamlText, req.filename, req.confirmations, req.corrections), None
    except MeasurementError as exc:
        return None, JSONResponse(
            status_code=422,
            content={"error": {"code": "invalid_yaml", "message": "Measurement import failed.",
                               "details": exc.errors}},
        )


def _blocked(resolution, res_json: JSONResponse | None) -> JSONResponse:
    if res_json is not None:
        return res_json
    return JSONResponse(
        status_code=BLOKED_STATUS,
        content={
            "error": {
                "code": "semantics_unresolved",
                "message": "Pattern generation is blocked until every required field has a "
                           "confirmed definition.",
                "details": resolution.blocking,
            },
            "semantics": resolution.as_dict(),
        },
    )


def _generate(req):
    (profile, resolution), res_json = _semantics(req)
    if res_json is not None:
        return None, None, None, None, res_json
    if not resolution.can_generate or resolution.measurements is None:
        return None, None, None, None, _blocked(resolution, None)
    settings = _settings(req.settings)
    result = draft(resolution, settings)
    payload = to_fixture_dict(result.document)
    GarmageSet.model_validate(payload)  # never emit output the schema rejects
    return profile, resolution, settings, (result, payload), None


def _preview(result, dark: bool = False) -> dict:
    options = RenderOptions()
    return {
        "svg": render_svg(result.document, result.cutting, options),
        "extent": list(document_extent(result.document, options)),
        "layers": list(VIEW_LAYERS),
        "panels": [
            {
                "id": p.ensure_id(), "label": p.label, "center": p.center.as_triple(),
                "edges": [
                    {"id": e.ensure_id(), "label": e.label, "role": e.role,
                     "lengthMm": round(e.length_mm, 4)}
                    for e in p.edges
                ],
                "darts": [{"id": d.id, "intakeMm": d.intake_mm} for d in p.darts],
            }
            for p in result.document.panels
        ],
    }


@router.post("/measurements/validate")
def validate_measurements(req: UploadRequest):
    """Step 1: syntax validation and strict typing, no interpretation."""
    try:
        profile = parse_source(req.content, req.filename)
    except MeasurementError as exc:
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "invalid_yaml", "message": "Measurement import failed.",
                               "details": exc.errors}},
        )
    return {
        "ok": True,
        "filename": req.filename,
        "unit": profile.unit,
        "rows": source_rows(profile),
        "dictionary": dictionary_summary(),
    }


@router.post("/measurements/semantics")
def resolve_field_semantics(req: SemanticsRequest):
    """Step 2-4: semantics resolution, normalization, consistency and confirmation."""
    (profile, resolution), res_json = _semantics(req)
    if res_json is not None:
        return res_json
    return {
        "filename": req.filename,
        "source": resolution.as_dict(),
        "registry": semantics_registry().as_dict(),
        "draftingMeasurements": (
            resolution.measurements.model_dump() if resolution.measurements else None
        ),
    }


@router.get("/measurements/semantics/schema")
def semantics_schema():
    return semantics_registry().as_dict()


@router.post("/measurements/report")
def measurements_report(req: SemanticsRequest):
    """Validation report for the definitions screen, before any geometry exists."""
    (profile, resolution), res_json = _semantics(req)
    if res_json is not None:
        return res_json
    return Response(
        content=json.dumps(
            {
                "report": "measurement-semantics",
                "generated_from": req.filename,
                "schema_version": resolution.schema_version,
                "definition_version": resolution.definition_version,
                "can_generate": resolution.can_generate,
                "fields": [f.as_dict() for f in resolution.fields.values()],
                "derived_measurements": resolution.derived,
                "assumptions": resolution.assumptions,
                "warnings": resolution.warnings,
                "blocking": resolution.blocking,
            },
            indent=2,
        ).encode("utf-8"),
        media_type="application/json",
        headers={"Content-Disposition": 'attachment; filename="measurement-report.json"'},
    )


@router.post("/patterns/generate")
def generate(req: GenerateRequest):
    profile, resolution, settings, generated, error = _generate(req)
    if error is not None:
        return error
    result, payload = generated
    return {
        "pattern": payload,
        "trace": manifest(result, profile, req.filename),
        "validation": result.report.as_dict(),
        "preview": _preview(result),
        "settings": result.settings.as_dict(),
    }


@router.post("/patterns/render")
def render(req: GenerateRequest):
    _profile, _resolution, _settings, generated, error = _generate(req)
    if error is not None:
        return error
    return _preview(generated[0])


@router.post("/patterns/validate")
def validate_pattern(req: RenderRequest):
    """Schema and closure checks on a supplied GarmageSet document."""
    try:
        model = GarmageSet.model_validate(req.pattern)
    except Exception as exc:
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "invalid_pattern_schema",
                               "message": "Pattern does not match the schema.",
                               "details": str(exc)}},
        )
    issues = _validate_payload(model)
    return {"ok": not issues, "panelCount": len(model.panels),
            "stitchCount": len(model.stitches), "issues": issues}


def _validate_payload(model: GarmageSet) -> list[dict]:
    tol = drafting_config()["tolerances"]
    issues: list[dict] = []
    ids: set[str] = set()
    for panel in model.panels:
        if panel.id in ids:
            issues.append({"check": "unique_identifiers", "panel": panel.id,
                           "message": "Duplicate panel id."})
        ids.add(panel.id)
        for seq in panel.seqEdges:
            for i, edge in enumerate(seq.edges):
                if edge.id in ids:
                    issues.append({"check": "unique_identifiers", "edge": edge.id,
                                   "message": "Duplicate edge id."})
                ids.add(edge.id)
                if len(edge.controlPoints) < 2:
                    issues.append({"check": "missing_boundary_edge", "edge": edge.id,
                                   "message": "Edge needs at least two control points."})
                nxt = seq.edges[(i + 1) % len(seq.edges)]
                gap = max(abs(a - b) for a, b in zip(edge.controlPoints[-1], nxt.controlPoints[0]))
                if gap > tol["endpoint_continuity_mm"]:
                    issues.append({"check": "endpoint_continuity", "edge": edge.id,
                                   "message": f"Endpoint gap {gap:.3f} mm."})
                if all(abs(a - b) < tol["duplicate_point_mm"]
                       for a, b in zip(edge.controlPoints[0], edge.controlPoints[-1])):
                    issues.append({"check": "zero_length_edge", "edge": edge.id,
                                   "message": "Edge start and end coincide."})
            if len(seq.vertices) < 3:
                issues.append({"check": "panel_closure", "panel": panel.id,
                               "message": "Fewer than three boundary vertices."})
    known = {e.id for p in model.panels for s in p.seqEdges for e in s.edges}
    for pair in model.stitches:
        for record in pair.records:
            for ref in (record.start, record.end):
                if ref.edgeId not in known:
                    issues.append({"check": "invalid_stitch_reference", "edge": ref.edgeId,
                                   "message": "Stitch references an unknown edge."})
                if not (0.0 <= ref.param <= 1.0):
                    issues.append({"check": "stitch_parameters", "edge": ref.edgeId,
                                   "message": f"param {ref.param} outside [0, 1]."})
    return issues


@router.post("/patterns/export/pdf")
def export_pdf_route(req: ExportRequest):
    _p, _r, settings, generated, error = _generate(req)
    if error is not None:
        return error
    result, _payload = generated
    data = export_pdf(result.document, result.cutting, settings.paper_size, settings.paper_orientation)
    return Response(content=data, media_type="application/pdf",
                    headers={"Content-Disposition": 'attachment; filename="pattern-a0.pdf"'})


@router.post("/patterns/export/dxf")
def export_dxf_route(req: ExportRequest):
    _p, _r, _settings, generated, error = _generate(req)
    if error is not None:
        return error
    data = export_dxf(generated[0].document, generated[0].cutting)
    return Response(content=data, media_type="application/dxf",
                    headers={"Content-Disposition": 'attachment; filename="pattern.dxf"'})


@router.post("/patterns/export/svg")
def export_svg_route(req: ExportRequest):
    _p, _r, _settings, generated, error = _generate(req)
    if error is not None:
        return error
    data = render_svg(generated[0].document, generated[0].cutting).encode("utf-8")
    return Response(content=data, media_type="image/svg+xml",
                    headers={"Content-Disposition": 'attachment; filename="pattern-preview.svg"'})


@router.post("/patterns/export/bundle")
def export_bundle(req: ExportRequest):
    profile, _resolution, settings, generated, error = _generate(req)
    if error is not None:
        return error
    result, payload = generated
    files = {
        "pattern.json": json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
        "generation-manifest.json": json.dumps(manifest(result, profile, req.filename),
                                              ensure_ascii=False, indent=2).encode("utf-8"),
        "validation-report.json": json.dumps(result.report.as_dict(),
                                             ensure_ascii=False, indent=2).encode("utf-8"),
        "pattern-preview.png": render_png(result.document, result.cutting),
        "pattern-a0.pdf": export_pdf(result.document, result.cutting,
                                     settings.paper_size, settings.paper_orientation),
        "pattern.dxf": export_dxf(result.document, result.cutting),
        "pattern-preview.svg": render_svg(result.document, result.cutting).encode("utf-8"),
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return Response(content=buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": 'attachment; filename="pattern-bundle.zip"'})


@router.get("/config")
def get_config():
    cfg = drafting_config()
    reg = semantics_registry()
    return {
        "easePresets": cfg["ease_presets"],
        "defaultEasePreset": cfg["default_ease_preset"],
        "seamAllowances": cfg["seam_allowance_mm"],
        "paper": cfg["paper"],
        "tolerances": cfg["tolerances"],
        "viewLayers": list(VIEW_LAYERS),
        "fixtureDefaults": cfg["fixture_defaults"],
        "measurementSchema": {
            "schemaVersion": reg.schema_version,
            "definitionVersion": reg.definition_version,
            "coordinateSystem": reg.as_dict()["coordinateSystem"],
        },
    }
