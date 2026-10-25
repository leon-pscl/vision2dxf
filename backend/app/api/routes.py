"""HTTP API. Structured errors only, never a raw stack trace.

Transformation is gated twice: a request that leaves any required field MISSING,
AMBIGUOUS or INVALID is refused with the blocking schema messages, and a target that
falls outside the tested transformation envelope is refused or held for confirmation.
"""
from __future__ import annotations

import io
import json
import zipfile

from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response

from ..base_block.loader import (
    BaseBlockError,
    catalog,
    catalog_summary,
    resolve_block,
    validate_base_block,
)
from ..config import drafting_config, transformation_config
from ..drafting.cutting import build_cutting_lines
from ..drafting.ease import Settings
from ..drafting.measurements import (
    MeasurementError,
    dictionary_summary,
    parse_source,
    source_rows,
)
from ..drafting.semantics import registry as semantics_registry
from ..export.dxf import export_dxf
from ..export.pdf import export_pdf
from ..rendering.png import render_png
from ..rendering.svg import COMPARISON_LAYERS, VIEW_LAYERS, RenderOptions, document_extent, render_svg
from ..schemas.api import ExportRequest, SemanticsRequest, TransformRequest, RenderRequest, UploadRequest
from ..schemas.garmage import GarmageSet
from ..serialization.garmageset import to_fixture_dict
from ..transformation.deltas import compute_deltas, influence_map
from ..transformation.pipeline import (
    TransformationBlocked,
    manifest,
    resolve_semantics,
    transform,
)

router = APIRouter(prefix="/api")

BLOCKED_STATUS = 409


def _settings(payload) -> Settings:
    return Settings(**payload.model_dump())


def _semantics(req) -> tuple[tuple, JSONResponse | None]:
    """Never returns None: an unresolved import becomes a structured 422, not a crash."""
    try:
        return resolve_semantics(req.yamlText, req.filename, req.confirmations, req.corrections), None
    except MeasurementError as exc:
        return (None, None), JSONResponse(
            status_code=422,
            content={"error": {"code": "invalid_yaml", "message": "Measurement import failed.",
                               "details": exc.errors}},
        )


def _blocked(code: str, message: str, details: list[dict], extra: dict | None = None) -> JSONResponse:
    content = {"error": {"code": code, "message": message, "details": details}}
    content.update(extra or {})
    return JSONResponse(status_code=BLOCKED_STATUS, content=content)


def _transform(req):
    """Resolve, transform, serialise. Returns (profile, resolution, settings, bundle, error)."""
    (profile, resolution), res_json = _semantics(req)
    if res_json is not None:
        return None, None, None, None, res_json
    settings = _settings(req.settings)
    try:
        result = transform(resolution, settings)
    except TransformationBlocked as exc:
        return None, None, None, None, _blocked(
            exc.code, exc.message, exc.details,
            {"semantics": resolution.as_dict()} if exc.code == "semantics_unresolved" else None,
        )
    except BaseBlockError as exc:
        return None, None, None, None, JSONResponse(
            status_code=500,
            content={"error": {"code": "base_block_unavailable",
                               "message": "The versioned base block could not be loaded.",
                               "details": exc.errors}},
        )
    payload = to_fixture_dict(result.document)
    GarmageSet.model_validate(payload)  # never emit output the schema rejects
    return profile, resolution, settings, (result, payload), None


def _panel_summary(doc) -> list[dict]:
    return [
        {
            "id": p.ensure_id(), "key": p.key, "label": p.label, "center": p.center.as_triple(),
            "edges": [
                {"id": e.ensure_id(), "label": e.label, "role": e.role,
                 "lengthMm": round(e.length_mm, 4)}
                for e in p.edges
            ],
            "darts": [{"id": d.id, "intakeMm": round(d.intake_mm, 4)} for d in p.darts],
            "landmarks": {role: point.as_triple() for role, point in p.landmarks.items()},
        }
        for p in doc.panels
    ]


def _preview(result, dark: bool = False) -> dict:
    """One render that carries both blocks, so the comparison cannot drift apart."""
    options = RenderOptions(dark=dark)
    base = result.block.document
    return {
        "svg": render_svg(result.document, result.cutting, options, base=base),
        "extent": list(document_extent(result.document, options)),
        "layers": list(VIEW_LAYERS),
        "comparisonLayers": list(COMPARISON_LAYERS),
        "panels": _panel_summary(result.document),
        "basePanels": _panel_summary(base),
        "displacements": result.diagnostics()["landmarkTargets"],
    }


# ---- step 1: syntax only ---------------------------------------------------

@router.post("/measurements/validate")
def validate_measurements(req: UploadRequest):
    """Syntax validation and strict typing, no interpretation."""
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
    """Semantics resolution, normalization, consistency and confirmation."""
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


# ---- the base blocks -------------------------------------------------------

def _block_or_error(block_id: str | None):
    try:
        return resolve_block(block_id), None
    except BaseBlockError as exc:
        return None, JSONResponse(
            status_code=404 if any(e["code"] == "unknown_base_block" for e in exc.errors) else 500,
            content={"error": {"code": "base_block_unavailable",
                               "message": "The base block could not be loaded.",
                               "details": exc.errors}},
        )


@router.get("/base-blocks")
def list_base_blocks():
    """The blocks on offer, so the operator can choose one before transforming."""
    return {
        "catalogVersion": catalog()["version"],
        "default": catalog()["default"],
        "reference": catalog().get("reference", {}),
        "blocks": catalog_summary(),
        "templateSelection": {
            "supported": True,
            "idFormat": "template:<file>.json",
            "note": "Any template in the library can be loaded this way. It has not been "
                    "reviewed as a base block and says so.",
        },
    }


@router.get("/base-blocks/{block_id:path}")
def get_base_block(block_id: str):
    """One block in full: semantics, derived maps, measured dimensions and a preview."""
    block, error = _block_or_error(block_id)
    if error is not None:
        return error
    report = validate_base_block(block)
    cutting = build_cutting_lines(block.document, Settings())
    return {
        **block.summary(),
        "maps": {
            "landmarks": block.landmark_map(),
            "edges": block.edge_map(),
            "seams": block.seam_map(),
            "darts": block.dart_map(),
            "grainlines": block.grainline_map(),
        },
        "previewSvg": render_svg(block.document, cutting, RenderOptions()),
        "panelSummary": _panel_summary(block.document),
        "validation": report.as_dict(),
        "validationStatus": report.status,
        "referenceProfile": block.reference_profile.as_dict(),
        "referenceEasePreset": block.reference_ease_preset,
        "referenceMeasurementsMm": block.reference_measurements_mm,
        "garmentDimensionsMm": block.garment_dimensions_mm,
        "measurementDefinitions": block.measurement_definitions,
        "provenance": block.provenance,
        "influenceMapVersion": influence_map()["version"],
    }


@router.post("/base-blocks/{block_id:path}/validate")
def validate_base_block_route(block_id: str):
    """The same geometry validator that runs on transformed patterns."""
    block, error = _block_or_error(block_id)
    if error is not None:
        return error
    report = validate_base_block(block)
    return {"ok": report.status == "pass", "blockId": block.id, **report.as_dict()}


@router.post("/base-block/deltas")
def base_block_deltas(req: TransformRequest):
    """Target-versus-base differences and the envelope, before any geometry is built."""
    (profile, resolution), res_json = _semantics(req)
    if res_json is not None:
        return res_json
    settings = _settings(req.settings)
    block, error = _block_or_error(settings.base_block_id)
    if error is not None:
        return error
    if resolution.measurements is None:
        return _blocked(
            "semantics_unresolved",
            "Measurement differences cannot be computed until every required field has a "
            "confirmed definition.",
            resolution.blocking,
            {"semantics": resolution.as_dict()},
        )

    from ..drafting.ease import resolve_ease
    from ..transformation.constraints import evaluate_envelope

    ease = resolve_ease(settings)
    reference_ease = resolve_ease(Settings(ease_preset=block.reference_ease_preset))
    deltas = compute_deltas(
        resolution.measurements, block.reference_measurements_mm, ease,
        {"bust_mm": reference_ease.bust_mm, "waist_mm": reference_ease.waist_mm,
         "hip_mm": reference_ease.hip_mm},
    )
    envelope = evaluate_envelope(deltas, confirmed=settings.confirm_envelope)
    return {
        "baseBlock": {"id": block.id, "label": block.label, "version": block.version,
                      "reviewed": block.reviewed},
        "referenceProfile": block.reference_profile.as_dict(),
        "referenceEasePreset": block.reference_ease_preset,
        "baseMeasurementsMm": block.reference_measurements_mm,
        "garmentDimensionsMm": block.garment_dimensions_mm,
        "measurementDefinitions": block.measurement_definitions,
        "targetEaseMm": {"bust_mm": ease.bust_mm, "waist_mm": ease.waist_mm,
                         "hip_mm": ease.hip_mm, "preset": ease.preset},
        "deltas": [d.as_dict() for d in sorted(deltas.values(), key=lambda d: d.name)],
        "envelope": envelope.as_dict(),
        "influenceMap": influence_map(),
    }


# ---- transformation --------------------------------------------------------

@router.post("/patterns/transform")
def transform_pattern(req: TransformRequest):
    profile, resolution, settings, bundle, error = _transform(req)
    if error is not None:
        return error
    result, payload = bundle
    return {
        "pattern": payload,
        "trace": manifest(result, profile, req.filename),
        "diagnostics": result.diagnostics(),
        "validation": result.report.as_dict(),
        "preview": _preview(result),
        "settings": result.settings.as_dict(),
    }


@router.post("/patterns/render")
def render(req: TransformRequest):
    _profile, _resolution, _settings, bundle, error = _transform(req)
    if error is not None:
        return error
    return _preview(bundle[0], dark=req.dark)


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


# ---- exports ---------------------------------------------------------------

@router.post("/patterns/export/pdf")
def export_pdf_route(req: ExportRequest):
    _p, _r, settings, bundle, error = _transform(req)
    if error is not None:
        return error
    result, _payload = bundle
    data = export_pdf(result.document, result.cutting, settings.paper_size,
                      settings.paper_orientation)
    return Response(content=data, media_type="application/pdf",
                    headers={"Content-Disposition": 'attachment; filename="pattern-a0.pdf"'})


@router.post("/patterns/export/dxf")
def export_dxf_route(req: ExportRequest):
    _p, _r, _settings, bundle, error = _transform(req)
    if error is not None:
        return error
    data = export_dxf(bundle[0].document, bundle[0].cutting)
    return Response(content=data, media_type="application/dxf",
                    headers={"Content-Disposition": 'attachment; filename="pattern.dxf"'})


@router.post("/patterns/export/svg")
def export_svg_route(req: ExportRequest):
    _p, _r, _settings, bundle, error = _transform(req)
    if error is not None:
        return error
    result = bundle[0]
    data = render_svg(result.document, result.cutting, RenderOptions(),
                      base=result.block.document).encode("utf-8")
    return Response(content=data, media_type="image/svg+xml",
                    headers={"Content-Disposition": 'attachment; filename="base-vs-target-preview.svg"'})


@router.post("/patterns/export/bundle")
def export_bundle(req: ExportRequest):
    profile, _resolution, settings, bundle, error = _transform(req)
    if error is not None:
        return error
    result, payload = bundle
    base_cutting = build_cutting_lines(result.block.document, settings)
    files = {
        "pattern.json": _json_bytes(payload),
        "generation-manifest.json": _json_bytes(manifest(result, profile, req.filename)),
        "transformation-diagnostics.json": _json_bytes(result.diagnostics()),
        "validation-report.json": _json_bytes(result.report.as_dict()),
        "base-vs-target-preview.png": render_png(result.document, result.cutting,
                                                 base=result.block.document),
        "base-vs-target-preview.svg": render_svg(result.document, result.cutting,
                                                 RenderOptions(),
                                                 base=result.block.document).encode("utf-8"),
        "base-block.json": _json_bytes(result.block.summary()),
        "base-block-landmark-map.json": _json_bytes(result.block.landmark_map()),
        "base-block-edge-map.json": _json_bytes(result.block.edge_map()),
        "base-block-seam-map.json": _json_bytes(result.block.seam_map()),
        "base-block-dart-map.json": _json_bytes(result.block.dart_map()),
        "base-block-grainline-map.json": _json_bytes(result.block.grainline_map()),
        "base-block-preview.svg": render_svg(result.block.document, base_cutting).encode("utf-8"),
        "pattern-a0.pdf": export_pdf(result.document, result.cutting, settings.paper_size,
                                     settings.paper_orientation),
        "pattern.dxf": export_dxf(result.document, result.cutting),
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return Response(content=buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": 'attachment; filename="pattern-bundle.zip"'})


def _json_bytes(data: dict) -> bytes:
    return json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")


@router.get("/config")
def get_config():
    cfg = drafting_config()
    tcfg = transformation_config()
    reg = semantics_registry()
    return {
        "easePresets": cfg["ease_presets"],
        "defaultEasePreset": cfg["default_ease_preset"],
        "seamAllowances": cfg["seam_allowance_mm"],
        "paper": cfg["paper"],
        "tolerances": cfg["tolerances"],
        "viewLayers": list(VIEW_LAYERS),
        "comparisonLayers": list(COMPARISON_LAYERS),
        "fixtureDefaults": cfg["fixture_defaults"],
        "transformation": {
            "version": tcfg["version"],
            "method": tcfg["solver"]["method"],
            "objective": tcfg["objective"],
            "solver": tcfg["solver"],
            "curveRefit": tcfg["curve_refit"],
            "seamBalance": tcfg["seam_balance"],
            "envelope": tcfg["envelope"],
            "influenceMapVersion": tcfg["influence"]["version"],
        },
        "measurementSchema": {
            "schemaVersion": reg.schema_version,
            "definitionVersion": reg.definition_version,
            "coordinateSystem": reg.as_dict()["coordinateSystem"],
        },
    }
