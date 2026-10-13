"""HTTP API. Structured errors only, never a raw stack trace.

One pipeline run serves every geometry endpoint. The stages are cheap enough - the whole
run is a couple of seconds, almost all of it the two flattening solves - that a job queue
would be more moving parts than the problem needs. Each response still carries the stage
timings under `progress`, so a client can show where the time went, and a stage that
refuses returns its reason with the stage that raised it.

`ponytail:` if the mesh resolution is ever raised far enough for a run to exceed a few
seconds, these endpoints need a job id, a poll route and cancellation. The `progress`
payload is already shaped for it.
"""
from __future__ import annotations

import io
import json
import zipfile

import numpy as np
from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response

from ..config import drafting_config, reconstruction_config
from ..drafting.ease import Settings
from ..drafting.measurements import (
    MeasurementError,
    dictionary_summary,
    parse_source,
    source_rows,
)
from ..drafting.semantics import default_confirmations
from ..drafting.semantics import registry as semantics_registry
from ..export.dxf import export_dxf
from ..export.pdf import export_pdf
from ..pipeline import (
    STITCH_PAIRS,
    UNMAPPED_FIELDS,
    GenerationBlocked,
    GenerationResult,
    generate,
    resolve_semantics,
)
from ..rendering.png import render_png
from ..rendering.svg import VIEW_LAYERS, RenderOptions, document_extent, render_svg
from ..schemas.api import ExportRequest, GenerateRequest, PatternRequest, SemanticsRequest, UploadRequest
from ..schemas.garmage import GarmageSet
from ..serialization.garmageset import to_fixture_dict

router = APIRouter(prefix="/api")

BLOCKED_STATUS = 409


# ---- shared plumbing -------------------------------------------------------


def _settings(payload) -> Settings:
    return Settings(**payload.model_dump())


def _semantics(req) -> tuple[tuple, JSONResponse | None]:
    """Never returns None: an unresolved import becomes a structured 422, not a crash."""
    try:
        return resolve_semantics(req.yamlText, req.filename, req.confirmations,
                                 req.corrections), None
    except MeasurementError as exc:
        return (None, None), JSONResponse(
            status_code=422,
            content={"error": {"code": "invalid_yaml", "message": "Measurement import failed.",
                               "details": exc.errors}},
        )


def _run(req) -> tuple[GenerationResult | None, JSONResponse | None]:
    (profile, resolution), error = _semantics(req)
    if error is not None:
        return None, error
    try:
        return generate(resolution, profile, _settings(req.settings)), None
    except GenerationBlocked as exc:
        content = {
            "error": {"code": exc.code, "message": exc.message, "details": exc.details},
            "stage": exc.stage,
        }
        if exc.code == "semantics_unresolved":
            content["semantics"] = resolution.as_dict()
        return None, JSONResponse(status_code=BLOCKED_STATUS, content=content)


def _round(array, digits: int = 2) -> list:
    return np.round(np.asarray(array, dtype=float), digits).tolist()


def _mesh_payload(surface) -> dict:
    """A ring-stack surface as the viewer wants it: flat vertices and triangles."""
    verts = surface.vertices().reshape(-1, 3)
    return {
        "label": surface.label,
        "columns": surface.columns,
        "rows": len(surface.ys),
        "vertices": _round(verts, 2),
        "faces": [[int(v) for v in f] for f in surface.faces()],
        "levelRows": surface.level_rows,
        "rings": surface.as_dict()["rings"],
    }


def _seam_paths(result: GenerationResult) -> list[dict]:
    """The seam network as 3D polylines, for the surface viewer."""
    out = []
    for panel in result.panels:
        for role, ids in panel.panel.semantic_boundaries.items():
            out.append({
                "panel": panel.panel.key,
                "role": role,
                "points": _round(panel.panel.vertices_3d[ids], 2),
            })
    return out


def _panel_summary(doc) -> list[dict]:
    return [
        {
            "id": p.ensure_id(), "key": p.key, "label": p.label,
            "center": (p.center.as_triple() if p.center else [0.0, 0.0, 0.0]),
            "edges": [
                {"id": e.ensure_id(), "label": e.label, "role": e.role,
                 "lengthMm": round(e.length_mm, 4),
                 "allowanceMm": e.seam_allowance_mm,
                 "pairedEdgeId": e.paired_edge_id}
                for e in p.edges
            ],
            "darts": [],
            "landmarks": {role: point.as_triple() for role, point in p.landmarks.items()},
        }
        for p in doc.panels
    ]


def _preview(result: GenerationResult, dark: bool = False) -> dict:
    options = RenderOptions(dark=dark)
    return {
        "svg": render_svg(result.document, result.cutting, options),
        "extent": [float(v) for v in document_extent(result.document, options)],
        "layers": [layer for layer in VIEW_LAYERS
                   if layer not in {"base", "displacement", "heatmap"}],
        "comparisonLayers": [],
        "panels": _panel_summary(result.document),
        "basePanels": [],
        "displacements": [],
    }


def _result_payload(result: GenerationResult, req) -> dict:
    payload = to_fixture_dict(result.document)
    GarmageSet.model_validate(payload)  # never emit output the schema rejects
    out = {
        "pattern": payload,
        "trace": result.manifest(req.filename),
        "diagnostics": result.flattening_diagnostics(),
        "validation": result.report.as_dict(),
        "preview": _preview(result, req.dark),
        "heatMaps": result.heat_maps(),
        "settings": result.settings.as_dict(),
        "assumptions": result.assumptions.as_list(),
        "progress": result.progress,
    }
    if getattr(req, "include_mesh", False):
        out["flattenedMesh"] = result.flattened_mesh()
    if getattr(req, "include_surfaces", False):
        out["crossSections"] = result.cross_sections_json()
        out["surfaceStages"] = {
            "proxy": _mesh_payload(result.proxy),
            "garment": _mesh_payload(result.garment),
            "seamNetwork": result.network.as_dict(),
            "seamPaths": _seam_paths(result),
            "surfaceSeamLengths": result.surface_seams,
            "panels": [
                {
                    **p.panel.as_dict(),
                    "vertices": _round(p.panel.vertices_3d, 2),
                    "faces": [[int(v) for v in f] for f in p.panel.faces],
                    "boundary": [int(i) for i in p.panel.boundary()],
                    "semanticBoundaries": {
                        role: [int(i) for i in ids]
                        for role, ids in p.panel.semantic_boundaries.items()
                    },
                }
                for p in result.panels
            ],
        }
    return out


def _production_export_block(result: GenerationResult) -> JSONResponse | None:
    report = result.report.as_dict()
    if report["summary"]["criticalFailures"] == 0:
        return None
    failed = [
        check for check in report["checks"]
        if check["status"] == "fail" and check["severity"] == "critical"
    ]
    return JSONResponse(
        status_code=409,
        content={
            "error": {
                "code": "validation_failed",
                "message": "Production export is blocked by critical geometry checks.",
                "details": [
                    {
                        "field": check["id"],
                        "code": "CRITICAL_GEOMETRY_CHECK_FAILED",
                        "message": check["message"],
                        "suggestedAction": check["suggestedAction"],
                    }
                    for check in failed
                ],
            },
            "stage": "validation",
            "validation": report,
        },
    )


# ---- step 1: the measurements ---------------------------------------------


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
        "warnings": [],
        "dictionary": dictionary_summary(),
    }


@router.post("/measurements/semantics")
def resolve_field_semantics(req: SemanticsRequest):
    """Semantics resolution, normalisation, consistency and confirmation."""
    (profile, resolution), error = _semantics(req)
    if error is not None:
        return error
    return {
        "filename": req.filename,
        "source": resolution.as_dict(),
        "registry": semantics_registry().as_dict(),
        "defaultConfirmations": default_confirmations(),
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
    (profile, resolution), error = _semantics(req)
    if error is not None:
        return error
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
                "unmapped_fields": UNMAPPED_FIELDS,
            },
            indent=2,
        ).encode("utf-8"),
        media_type="application/json",
        headers={"Content-Disposition": 'attachment; filename="measurement-report.json"'},
    )


# ---- reconstruction --------------------------------------------------------


@router.post("/reconstruction/cross-sections")
def reconstruct_cross_sections(req: GenerateRequest):
    """Anatomical levels and the solved sections, with every residual."""
    result, error = _run(req)
    if error is not None:
        return error
    return {
        **result.cross_sections_json(),
        "assumptions": result.assumptions.as_list(),
        "progress": result.progress,
    }


@router.post("/reconstruction/proxy-surface")
def reconstruct_proxy_surface(req: GenerateRequest):
    """The lofted, smoothed proxy torso."""
    result, error = _run(req)
    if error is not None:
        return error
    return {
        "mesh": _mesh_payload(result.proxy),
        "smoothing": result.smoothing,
        "quality": {p.panel.key: p.quality for p in result.panels},
        "assumptions": result.assumptions.as_list(),
        "progress": result.progress,
    }


@router.post("/garment/surface")
def garment_surface(req: GenerateRequest):
    """The ease-adjusted garment surface and the circumferences it achieved."""
    result, error = _run(req)
    if error is not None:
        return error
    return {
        "mesh": _mesh_payload(result.garment),
        "proxyMesh": _mesh_payload(result.proxy),
        "ease": result.ease.as_dict(),
        "diagnostics": result.garment_diagnostics,
        "progress": result.progress,
    }


@router.post("/garment/segment")
def garment_segment(req: GenerateRequest):
    """The seam network and the segmented panels."""
    result, error = _run(req)
    if error is not None:
        return error
    return {
        "network": result.network.as_dict(),
        "seamPaths": _seam_paths(result),
        "surfaceSeamLengths": result.surface_seams,
        "panels": [
            {
                **p.panel.as_dict(),
                "vertices": _round(p.panel.vertices_3d, 2),
                "faces": [[int(v) for v in f] for f in p.panel.faces],
                "boundary": [int(i) for i in p.panel.boundary()],
                "semanticBoundaries": {
                    role: [int(i) for i in ids]
                    for role, ids in p.panel.semantic_boundaries.items()
                },
            }
            for p in result.panels
        ],
        "stitchPairs": [
            {"a": {"panel": a[0], "role": a[1]}, "b": {"panel": b[0], "role": b[1]}}
            for a, b in STITCH_PAIRS
        ],
        "progress": result.progress,
    }


# ---- flattening and the pattern -------------------------------------------


@router.post("/patterns/flatten")
def flatten_pattern(req: GenerateRequest):
    """The whole pipeline: pattern, manifest, diagnostics, validation and preview."""
    result, error = _run(req)
    if error is not None:
        return error
    return _result_payload(result, req)


@router.post("/patterns/render")
def render(req: GenerateRequest):
    result, error = _run(req)
    if error is not None:
        return error
    return {**_preview(result, req.dark), "heatMaps": result.heat_maps()}


@router.post("/patterns/validate")
def validate_pattern(req: PatternRequest):
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
    result, error = _run(req)
    if error is not None:
        return error
    blocked = _production_export_block(result)
    if blocked is not None:
        return blocked
    data = export_pdf(result.document, result.cutting, result.settings.paper_size,
                      result.settings.paper_orientation)
    return Response(content=data, media_type="application/pdf",
                    headers={"Content-Disposition": 'attachment; filename="pattern-a0.pdf"'})


@router.post("/patterns/export/dxf")
def export_dxf_route(req: ExportRequest):
    result, error = _run(req)
    if error is not None:
        return error
    blocked = _production_export_block(result)
    if blocked is not None:
        return blocked
    data = export_dxf(result.document, result.cutting)
    return Response(content=data, media_type="application/dxf",
                    headers={"Content-Disposition": 'attachment; filename="pattern.dxf"'})


@router.post("/patterns/export/svg")
def export_svg_route(req: ExportRequest):
    result, error = _run(req)
    if error is not None:
        return error
    blocked = _production_export_block(result)
    if blocked is not None:
        return blocked
    data = render_svg(result.document, result.cutting, RenderOptions()).encode("utf-8")
    return Response(content=data, media_type="image/svg+xml",
                    headers={"Content-Disposition": 'attachment; filename="pattern-preview.svg"'})


# The debugging artefacts, by name. They never replace the required outputs.
DEBUG_ARTIFACTS = (
    "cross-sections.json", "proxy-mesh.obj", "garment-mesh.obj", "segmented-panels.obj",
    "flattened-mesh.json", "flattening-diagnostics.json",
)


@router.post("/patterns/debug/{artifact}")
def export_debug(artifact: str, req: ExportRequest):
    if artifact not in DEBUG_ARTIFACTS:
        return JSONResponse(
            status_code=404,
            content={"error": {"code": "unknown_artifact",
                               "message": f"No debugging artefact named {artifact}.",
                               "details": [{"field": "artifact",
                                            "message": f"One of {', '.join(DEBUG_ARTIFACTS)}."}]}},
        )
    result, error = _run(req)
    if error is not None:
        return error
    data = _debug_files(result)[artifact]
    media = "application/json" if artifact.endswith(".json") else "text/plain"
    return Response(content=data, media_type=media,
                    headers={"Content-Disposition": f'attachment; filename="{artifact}"'})


def _debug_files(result: GenerationResult) -> dict[str, bytes]:
    return {
        "cross-sections.json": _json_bytes(result.cross_sections_json()),
        "proxy-mesh.obj": result.proxy.obj(),
        "garment-mesh.obj": result.garment.obj(),
        "segmented-panels.obj": b"".join(p.panel.obj() for p in result.panels),
        "flattened-mesh.json": _json_bytes(result.flattened_mesh()),
        "flattening-diagnostics.json": _json_bytes(result.flattening_diagnostics()),
    }


@router.post("/patterns/export/bundle")
def export_bundle(req: ExportRequest):
    result, error = _run(req)
    if error is not None:
        return error
    blocked = _production_export_block(result)
    if blocked is not None:
        return blocked
    payload = to_fixture_dict(result.document)
    GarmageSet.model_validate(payload)
    files = {
        "pattern.json": _json_bytes(payload),
        "generation-manifest.json": _json_bytes(result.manifest(req.filename)),
        "validation-report.json": _json_bytes(result.report.as_dict()),
        "flattening-diagnostics.json": _json_bytes(result.flattening_diagnostics()),
        "pattern-preview.png": render_png(result.document, result.cutting),
        "pattern-preview.svg": render_svg(result.document, result.cutting,
                                          RenderOptions()).encode("utf-8"),
        "pattern-a0.pdf": export_pdf(result.document, result.cutting,
                                     result.settings.paper_size,
                                     result.settings.paper_orientation),
        "pattern.dxf": export_dxf(result.document, result.cutting),
        **{f"debug/{name}": data for name, data in _debug_files(result).items()},
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
    rcfg = reconstruction_config()
    reg = semantics_registry()
    return {
        "version": rcfg["version"],
        "crossSection": rcfg["cross_section"],
        "levels": rcfg["levels"],
        "mesh": rcfg["mesh"],
        "garment": rcfg["garment"],
        "seams": rcfg["seams"],
        "flattening": rcfg["flattening"],
        "boundaryRefit": rcfg["boundary_refit"],
        "easePresets": rcfg["garment"]["ease_presets"],
        "defaultEasePreset": rcfg["garment"]["default_ease_preset"],
        "seamAllowances": cfg["seam_allowance_mm"],
        "paper": cfg["paper"],
        "tolerances": {**cfg["tolerances"], **rcfg["tolerances"]},
        "viewLayers": list(VIEW_LAYERS),
        "fixtureDefaults": cfg["fixture_defaults"],
        "debugArtifacts": list(DEBUG_ARTIFACTS),
        "unmappedFields": UNMAPPED_FIELDS,
        "measurementSchema": {
            "schemaVersion": reg.schema_version,
            "definitionVersion": reg.definition_version,
            "coordinateSystem": reg.as_dict()["coordinateSystem"],
        },
    }
