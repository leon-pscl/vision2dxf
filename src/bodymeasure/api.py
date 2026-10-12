"""FastAPI application: the JSON API of brief 17 plus the pages in ``web/``.

Two rules shape everything here. Nothing user-supplied ever reaches a path: a
view and a stage id are checked against the pipeline's own lists before a file
name is built from them, so there is no traversal to prevent further down.
And an error is answered with a code and a sentence, never with a traceback.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from bodymeasure import __version__, artifacts, store
from bodymeasure.config import ROOT, load_app_config, load_calibration, load_schema
from bodymeasure.export import export_gate, measurements_csv
from bodymeasure.landmarks import Landmark, load_landmarks
from bodymeasure.measure import MeasurementSet
from bodymeasure.preprocess import decode_image
from bodymeasure.session import (
    HEIGHT_HINT,
    PROFILES,
    SCALE_MODES,
    VIEWS,
    Subject,
    captured_views,
    create_session,
    export,
    find_landmarks,
    gate_of,
    load_subject,
    measure_session,
    process,
    process_status,
    reset_landmarks,
    save_overrides,
    save_view,
    session_dir,
)
from bodymeasure.vision import load_models, weights_available


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Prepare the store and the models before the first request needs them.

    Weights are loaded from disk if they are there and never downloaded: a
    station with no model files starts and says so (brief 5).
    """
    store.initialise()
    if load_app_config().storage.purge_on_startup:
        store.purge_images()
    if weights_available():
        load_models()
    yield


app = FastAPI(title="Body measurement", version=__version__, lifespan=lifespan)


class ValueEdit(BaseModel):
    """One measurement the operator typed in, with the note explaining it."""

    value: float
    note: str = ""


class LandmarkIn(BaseModel):
    """One landmark as the review UI sends it back."""

    name: str
    view: str
    x: float
    y: float
    auto: bool = True


def _check_view(view: str) -> str:
    """Return ``view`` if the pipeline knows it.

    Raises:
        HTTPException: for any other string, before it can reach a path.

    """
    if view not in VIEWS:
        raise HTTPException(status_code=404, detail=f"unknown view: must be one of {VIEWS}")
    return view


def _check_stage(stage_id: str) -> str:
    """Return ``stage_id`` if it is a stage of the pipeline.

    Raises:
        HTTPException: for any other string, before it can reach a path.

    """
    if stage_id not in artifacts.STAGE_IDS:
        raise HTTPException(status_code=404, detail=f"unknown stage: {stage_id}")
    return stage_id


def _measurement_payload(
    marks: list[Landmark], measurements: MeasurementSet, session_id: str
) -> dict[str, object]:
    """Shape landmarks, measurements and the export gate for the review page."""
    gate = export_gate(measurements, load_subject(session_id).profile)
    return {
        "landmarks": [asdict(mark) for mark in marks],
        "measurements": {
            name: {
                "value": entry.value_cm,
                "unit": "cm" if entry.unit == "mm" else "deg",
                "status": entry.status,
                "source": entry.source,
                "method": entry.method,
                "formula": entry.formula,
                "inputs": {key: round(value, 2) for key, value in entry.inputs.items()},
                "confidence": entry.confidence,
                "validated_against_manual": False,
                "edited": entry.edited,
                "auto": entry.auto,
                "views": entry.views,
                "warnings": entry.warnings,
                "note": entry.note,
            }
            for name, entry in sorted(measurements.values.items())
        },
        "primary_keys": sorted(load_schema()),
        "warnings": measurements.warnings,
        "export": {"ok": gate.ok, "profile": gate.profile, "blocked": gate.blocked},
    }


def _measured(session_id: str) -> dict[str, object]:
    """Measure the session and return the review payload.

    Raises:
        HTTPException: 404 for a missing session, 422 for a missing view.

    """
    try:
        measurements = process(session_id)
        marks = load_landmarks(session_dir(session_id) / "landmarks.json")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _measurement_payload(marks, measurements, session_id)


def _export_file(session_id: str, name: str) -> FileResponse:
    """Serve one exported YAML file.

    Raises:
        HTTPException: if that file has not been written for this session.

    """
    path = session_dir(session_id) / name
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"{name} has not been written: process the session, and check the export gate",
        )
    return FileResponse(path, media_type="application/x-yaml", filename=name)


@app.get("/api/health")
def health() -> dict[str, object]:
    """Report the version, the station's state and what the UI needs to offer."""
    config = load_app_config()
    calibration = load_calibration()
    return {
        "status": "ok",
        "version": __version__,
        "calibrated": calibration is not None and calibration.has_lens_model,
        "weights": weights_available(),
        "camera_rotation": config.camera.rotation,
        "consent_version": config.storage.consent_version,
        "height_hint": HEIGHT_HINT,
        "views": list(VIEWS),
        "profiles": list(PROFILES),
        "scale_modes": list(SCALE_MODES),
        "garmentcode_commit": config.garmentcode.commit,
        "validation_status": "unvalidated_against_manual_anthropometry",
        "stages": [asdict(stage) for stage in artifacts.STAGES],
        "measurement_keys": sorted(load_schema()),
    }


@app.post("/api/sessions", status_code=201)
async def create_session_route(
    subject_code: Annotated[str, Form()],
    sex: Annotated[str, Form()],
    consent: Annotated[bool, Form()],
    height_cm: Annotated[float, Form()],
    notes: Annotated[str, Form()] = "",
    profile: Annotated[str, Form()] = "full",
    scale_mode: Annotated[str, Form()] = "stature",
) -> dict[str, str]:
    """Open a session. Consent must be ticked before any capture."""
    try:
        session_id = create_session(
            Subject(
                code=subject_code,
                sex=sex,
                consent=consent,
                height_cm=height_cm,
                notes=notes,
                profile=profile,
                scale_mode=scale_mode,
            )
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"id": session_id}


@app.get("/api/sessions")
async def list_sessions_route(status: str | None = None) -> list[dict[str, object]]:
    """List sessions from the index, newest first, optionally by status."""
    return store.list_sessions(status)


@app.get("/api/sessions/{session_id}")
async def get_session_route(session_id: str) -> dict[str, object]:
    """Return one session: its subject, its views and where it has got to.

    Raises:
        HTTPException: 404 if there is no such session.

    """
    try:
        subject = load_subject(session_id)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=f"no such session: {session_id}") from exc
    row = store.get_session(session_id) or {}
    return {
        "id": session_id,
        "status": row.get("status"),
        "subject_code": subject.code,
        "sex": subject.sex,
        "height_cm": subject.height_cm,
        "profile": subject.profile,
        "scale_mode": subject.scale_mode,
        "required_fields": load_app_config().profiles[subject.profile],
        "views_captured": captured_views(session_id),
        "gates": {view: gate_of(session_id, view) for view in VIEWS},
        "images_purge_after": row.get("images_purge_after"),
    }


@app.delete("/api/sessions/{session_id}")
async def delete_session_route(session_id: str) -> dict[str, bool]:
    """Delete the session folder, its row and every image in it."""
    try:
        deleted = store.delete_session(session_id, reason="user_request")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail=f"no such session: {session_id}")
    return {"deleted": True}


@app.post("/api/sessions/{session_id}/views/{view}")
async def capture_view(
    session_id: str,
    view: str,
    image: Annotated[UploadFile, File()],
    upright: Annotated[bool, Form()] = False,
) -> dict[str, object]:
    """Store one captured view and return the pose gate result.

    ``upright`` marks a frame that is already the right way up. A live capture
    comes from a camera mounted portrait and arrives sideways; an upload or a
    stored sample is a photograph and must not be turned again.
    """
    _check_view(view)
    limits = load_app_config().uploads
    if image.content_type and image.content_type not in limits.allowed_types:
        raise HTTPException(
            status_code=415,
            detail=f"{image.content_type} is not accepted: use {', '.join(limits.allowed_types)}",
        )
    try:
        result = save_view(session_id, view, decode_image(await image.read()), upright)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {**result.as_dict(), "views": captured_views(session_id)}


@app.get("/api/sessions/{session_id}/views/{view}/gate")
async def view_gate(session_id: str, view: str) -> dict[str, object]:
    """Return the gate result stored when this view was captured."""
    return gate_of(session_id, _check_view(view))


@app.post("/api/sessions/{session_id}/process")
async def process_route(session_id: str) -> dict[str, object]:
    """Find landmarks, measure, write the overlays and the export files."""
    return _measured(session_id)


@app.get("/api/sessions/{session_id}/process/status")
async def process_status_route(session_id: str) -> dict[str, object]:
    """Return which stage each view has reached, and what is still missing."""
    try:
        return process_status(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/sessions/{session_id}/artifacts")
async def list_artifacts(session_id: str) -> dict[str, object]:
    """Return the manifest and the stage definitions the filmstrip is built from."""
    return {
        "stages": [asdict(stage) for stage in artifacts.STAGES],
        "views": list(VIEWS),
        "manifest": artifacts.manifest(session_id),
    }


@app.get("/api/sessions/{session_id}/artifacts/{view}/{stage_id}")
async def artifact_image(session_id: str, view: str, stage_id: str) -> FileResponse:
    """Serve one stage image of one view.

    Raises:
        HTTPException: 404 if that stage was never produced for that view.

    """
    path = artifacts.stage_path(session_id, _check_view(view), _check_stage(stage_id))
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{view} has no {stage_id} yet")
    return FileResponse(path, media_type="image/jpeg" if path.suffix == ".jpg" else "image/png")


@app.get("/api/sessions/{session_id}/landmarks")
async def get_landmarks(session_id: str) -> dict[str, object]:
    """Return the stored landmarks, finding them on first call."""
    path = session_dir(session_id) / "landmarks.json"
    if path.exists():
        return {"landmarks": [asdict(mark) for mark in load_landmarks(path)]}
    try:
        return {"landmarks": [asdict(mark) for mark in find_landmarks(session_id)]}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.put("/api/sessions/{session_id}/landmarks")
async def update_landmarks(session_id: str, landmarks: list[LandmarkIn]) -> dict[str, object]:
    """Replace the landmarks with the operator's edits and measure again."""
    for mark in landmarks:
        _check_view(mark.view)
    marks = [
        Landmark(name=mark.name, view=mark.view, x=mark.x, y=mark.y, auto=mark.auto)
        for mark in landmarks
    ]
    try:
        measurements = measure_session(session_id, marks)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    export(session_id, measurements)
    return _measurement_payload(marks, measurements, session_id)


@app.post("/api/sessions/{session_id}/landmarks/reset")
async def reset_landmarks_route(session_id: str) -> dict[str, object]:
    """Throw away every edit and go back to what the pipeline proposed."""
    try:
        reset_landmarks(session_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _measured(session_id)


@app.get("/api/sessions/{session_id}/measurements")
async def get_measurements(session_id: str) -> dict[str, object]:
    """Return the measurements, landmarks and export gate of a session."""
    return _measured(session_id)


@app.put("/api/sessions/{session_id}/measurements/{name}")
async def update_measurement(
    session_id: str, name: str, edit: ValueEdit
) -> dict[str, object]:
    """Record one value the operator typed, with the note that is required for it.

    Raises:
        HTTPException: 422 if the note is missing: an override without a reason
            is not reviewable (brief 16.3).

    """
    if not edit.note.strip():
        raise HTTPException(
            status_code=422, detail="a note is required when you type a value over a measurement"
        )
    try:
        save_overrides(session_id, {name: {"value": edit.value, "note": edit.note}})
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _measured(session_id)


@app.post("/api/sessions/{session_id}/measurements/recompute")
async def recompute(session_id: str) -> dict[str, object]:
    """Measure again from the stored landmarks and edits."""
    return _measured(session_id)


@app.post("/api/sessions/{session_id}/confirm")
async def confirm_session(session_id: str) -> dict[str, object]:
    """Mark the measurements checked by an operator, which starts the retention clock."""
    try:
        store.set_status(session_id, "confirmed")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    store.drop_debug(session_id)
    session = store.get_session(session_id) or {}
    return {
        "status": session.get("status"),
        "images_purge_after": session.get("images_purge_after"),
    }


@app.get("/api/sessions/{session_id}/export/body")
async def export_body(session_id: str) -> FileResponse:
    """Serve ``body_measurements.yaml``, which exists only if the gate passed."""
    return _export_file(session_id, "body_measurements.yaml")


@app.get("/api/sessions/{session_id}/export/extended")
async def export_extended(session_id: str) -> FileResponse:
    """Serve ``extended_measurements.yaml``: everything not a primary key."""
    return _export_file(session_id, "extended_measurements.yaml")


@app.get("/api/sessions/{session_id}/export/report")
async def export_report(session_id: str) -> FileResponse:
    """Serve the provenance report, which is written for a failed session too."""
    return _export_file(session_id, "measurement_report.yaml")


@app.get("/api/sessions/{session_id}/export/csv")
async def export_session_csv(session_id: str) -> PlainTextResponse:
    """Serve this session's measurements as one row per measurement."""
    try:
        measurements = measure_session(session_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return PlainTextResponse(
        measurements_csv(measurements),
        media_type="text/csv",
        headers={"content-disposition": 'attachment; filename="measurements.csv"'},
    )


@app.get("/api/export.csv")
async def export_csv_route() -> FileResponse:
    """Serve one row per confirmed session, for later analysis."""
    path = store.data_dir() / "export.csv"
    store.export_csv(path)
    return FileResponse(path, media_type="text/csv", filename="export.csv")


@app.get("/review/{session_id}")
async def review_page(session_id: str) -> FileResponse:
    """Serve the review page; the script reads the id from the path."""
    return FileResponse(ROOT / "web" / "review.html", media_type="text/html")


@app.get("/process/{session_id}")
async def processing_page(session_id: str) -> FileResponse:
    """Serve the processing page; the script reads the id from the path."""
    return FileResponse(ROOT / "web" / "process.html", media_type="text/html")


# Mounted last so the /api routes above win.
app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
