"""HTTP API. Structured errors only, never a raw stack trace."""
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
    parse_measurements,
    review_rows,
)
from ..drafting.pipeline import draft, manifest
from ..export.dxf import export_dxf
from ..export.pdf import export_pdf
from ..rendering.png import render_png
from ..rendering.svg import VIEW_LAYERS, RenderOptions, document_extent, render_svg
from ..schemas.api import ExportRequest, GenerateRequest, RenderRequest, UploadRequest
from ..schemas.garmage import GarmageSet
from ..serialization.garmageset import to_fixture_dict

router = APIRouter(prefix="/api")


def _settings(payload) -> Settings:
    return Settings(**payload.model_dump())


def _generate_from(req: ExportRequest | GenerateRequest):
    try:
        body = parse_measurements(req.yamlText, req.filename)
    except MeasurementError as exc:
        return None, None, None, None, JSONResponse(
            status_code=422,
            content={"error": {"code": "invalid_measurements",
                               "message": "Measurement import failed.",
                               "details": exc.errors}},
        )
    settings = _settings(req.settings)
    result = draft(body, settings)
    payload = to_fixture_dict(result.document)
    GarmageSet.model_validate(payload)  # never emit output the schema rejects
    return body, settings, result, payload, None


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
    try:
        body = parse_measurements(req.content, req.filename)
    except MeasurementError as exc:
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "invalid_measurements",
                               "message": "Measurement import failed.",
                               "details": exc.errors}},
        )
    return {
        "ok": True,
        "filename": req.filename,
        "unit": body.unit,
        "rows": review_rows(body),
        "warnings": [w.__dict__ for w in body.warnings],
        "dictionary": dictionary_summary(),
    }


@router.post("/patterns/generate")
def generate(req: GenerateRequest):
    body, settings, result, payload, error = _generate_from(req)
    if error is not None:
        return error
    return {
        "pattern": payload,
        "trace": manifest(result, body, req.filename),
        "validation": result.report.as_dict(),
        "preview": _preview(result),
        "settings": result.settings.as_dict(),
    }


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


@router.post("/patterns/render")
def render(req: GenerateRequest):
    """Server-side SVG render of the same document the exports are built from."""
    body, settings, result, payload, error = _generate_from(req)
    if error is not None:
        return error
    return _preview(result)


@router.post("/patterns/export/pdf")
def export_pdf_route(req: ExportRequest):
    body, settings, result, payload, error = _generate_from(req)
    if error is not None:
        return error
    data = export_pdf(result.document, result.cutting, settings.paper_size, settings.paper_orientation)
    return Response(content=data, media_type="application/pdf",
                    headers={"Content-Disposition": 'attachment; filename="pattern-a0.pdf"'})


@router.post("/patterns/export/dxf")
def export_dxf_route(req: ExportRequest):
    body, settings, result, payload, error = _generate_from(req)
    if error is not None:
        return error
    data = export_dxf(result.document, result.cutting)
    return Response(content=data, media_type="application/dxf",
                    headers={"Content-Disposition": 'attachment; filename="pattern.dxf"'})


@router.post("/patterns/export/svg")
def export_svg_route(req: ExportRequest):
    _body, _settings, result, _payload, error = _generate_from(req)
    if error is not None:
        return error
    data = render_svg(result.document, result.cutting).encode("utf-8")
    return Response(content=data, media_type="image/svg+xml",
                    headers={"Content-Disposition": 'attachment; filename="pattern-preview.svg"'})


@router.post("/patterns/export/bundle")
def export_bundle(req: ExportRequest):
    body, settings, result, payload, error = _generate_from(req)
    if error is not None:
        return error
    manifest_data = manifest(result, body, req.filename)
    report = result.report.as_dict()
    files = {
        "pattern.json": json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
        "generation-manifest.json": json.dumps(manifest_data, ensure_ascii=False, indent=2).encode("utf-8"),
        "validation-report.json": json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8"),
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
    return {
        "easePresets": cfg["ease_presets"],
        "defaultEasePreset": cfg["default_ease_preset"],
        "seamAllowances": cfg["seam_allowance_mm"],
        "paper": cfg["paper"],
        "tolerances": cfg["tolerances"],
        "viewLayers": list(VIEW_LAYERS),
        "fixtureDefaults": cfg["fixture_defaults"],
    }
