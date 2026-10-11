"""FastAPI application: JSON API plus the static pages in ``web/``."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from bodymeasure import __version__, store
from bodymeasure.calibrate import (
    compute_intrinsics,
    compute_plane_scale,
    load_intrinsics,
    save_calibration,
    save_intrinsics,
)
from bodymeasure.config import ROOT, load_app_config, load_calibration, load_schema
from bodymeasure.landmarks import Landmark, load_landmarks
from bodymeasure.measure import MeasurementSet
from bodymeasure.preprocess import decode_image, rotate
from bodymeasure.session import (
    Subject,
    captured_views,
    create_session,
    export,
    measure_session,
    save_overrides,
    save_view,
    session_dir,
)
from bodymeasure.vision import load_models, weights_available


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Prepare the store and the models before the first request needs them."""
    store.initialise()
    if load_app_config().storage.purge_on_startup:
        store.purge_images()
    if weights_available():
        load_models()
    yield


app = FastAPI(title="Body Measurement Demo", version=__version__, lifespan=lifespan)


class ValueEdit(BaseModel):
    """One measurement the tailor typed in, with the note explaining it."""

    value: float
    note: str = ""


class LandmarkIn(BaseModel):
    """One landmark as the review UI sends it back."""

    name: str
    view: str
    x: float
    y: float
    auto: bool = True


def _measurement_payload(
    marks: list[Landmark], measurements: MeasurementSet
) -> dict[str, object]:
    """Shape landmarks and measurements the way the review page consumes them."""
    return {
        "landmarks": [asdict(mark) for mark in marks],
        "measurements": {
            name: {
                "value": entry.value_cm,
                "source": entry.source,
                "confidence": entry.confidence,
                "edited": entry.edited,
                "views": entry.views,
                "note": entry.note,
            }
            for name, entry in sorted(measurements.values.items())
        },
        "warnings": measurements.warnings,
    }


def _yaml_file(session_id: str, name: str) -> FileResponse:
    """Serve one exported YAML file.

    Raises:
        HTTPException: if the session has not been measured yet.

    """
    path = session_dir(session_id) / name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{name} not written yet: measure first")
    return FileResponse(path, media_type="application/x-yaml", filename=name)


@app.get("/api/health")
def health() -> dict[str, object]:
    """Report version, whether the station is calibrated, and the measurement keys."""
    config = load_app_config()
    calibration = load_calibration()
    return {
        "status": "ok",
        "version": __version__,
        "calibrated": calibration is not None,
        "weights": weights_available(),
        "calibration": None
        if calibration is None
        else {
            "floor_row": calibration.floor_row,
            "image_size": calibration.image_size,
            "rotation": calibration.rotation,
            "mm_per_px_y": calibration.mm_per_px_y,
        },
        "camera_rotation": config.camera.rotation,
        "consent_version": config.storage.consent_version,
        "measurement_keys": sorted(load_schema()),
    }


@app.post("/api/calibration/checkerboard")
async def calibration_checkerboard(
    frames: Annotated[list[UploadFile], File()],
    pattern_cols: Annotated[int, Form()],
    pattern_rows: Annotated[int, Form()],
    square_mm: Annotated[float, Form()],
) -> dict[str, object]:
    """Compute and park the camera intrinsics from several checkerboard views."""
    rotation = load_app_config().camera.rotation
    try:
        images = [rotate(decode_image(await frame.read()), rotation) for frame in frames]
        intrinsics = compute_intrinsics(images, (pattern_cols, pattern_rows), square_mm)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    save_intrinsics(intrinsics)
    return asdict(intrinsics)


@app.post("/api/calibration/scale")
async def calibration_scale(
    frame: Annotated[UploadFile, File()],
    pattern_cols: Annotated[int, Form()],
    pattern_rows: Annotated[int, Form()],
    square_mm: Annotated[float, Form()],
    bottom_margin_mm: Annotated[float, Form()] = 0.0,
    save: Annotated[bool, Form()] = False,
) -> dict[str, object]:
    """Measure the body-plane scale, and with ``save`` write the station calibration.

    Called twice by the page: once to show the numbers, once to commit them.
    """
    rotation = load_app_config().camera.rotation
    intrinsics = load_intrinsics()
    try:
        image = rotate(decode_image(await frame.read()), rotation)
        scale = compute_plane_scale(
            image, (pattern_cols, pattern_rows), square_mm, intrinsics, bottom_margin_mm
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    result: dict[str, object] = {**asdict(scale), "saved": False}
    if save:
        if intrinsics is None:
            raise HTTPException(status_code=409, detail="run the checkerboard step first")
        save_calibration(intrinsics, scale, rotation)
        result["saved"] = True
    return result


@app.post("/api/sessions", status_code=201)
async def create_session_route(
    subject_code: Annotated[str, Form()],
    sex: Annotated[str, Form()],
    consent: Annotated[bool, Form()],
    notes: Annotated[str, Form()] = "",
) -> dict[str, str]:
    """Open a session. Consent must be ticked before any capture."""
    try:
        session_id = create_session(
            Subject(code=subject_code, sex=sex, consent=consent, notes=notes)
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"id": session_id}


@app.post("/api/sessions/{session_id}/views/{view}")
async def capture_view(
    session_id: str, view: str, image: Annotated[UploadFile, File()]
) -> dict[str, object]:
    """Store one captured view and return the pose gate result."""
    try:
        result = save_view(session_id, view, decode_image(await image.read()))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"ok": result.ok, "reasons": result.reasons, "views": captured_views(session_id)}


@app.post("/api/sessions/{session_id}/measure")
async def measure_route(session_id: str) -> dict[str, object]:
    """Measure the session, finding landmarks only if none are stored yet.

    Safe to call again when the review page reloads: edited landmarks survive,
    because a capture is what clears them.
    """
    try:
        measurements = measure_session(session_id)
        marks = load_landmarks(session_dir(session_id) / "landmarks.json")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    export(session_id, measurements)
    return _measurement_payload(marks, measurements)


@app.put("/api/sessions/{session_id}/landmarks")
async def update_landmarks(session_id: str, landmarks: list[LandmarkIn]) -> dict[str, object]:
    """Replace the landmarks with the tailor's edits and measure again."""
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
    return _measurement_payload(marks, measurements)


@app.put("/api/sessions/{session_id}/values")
async def update_values(session_id: str, values: dict[str, ValueEdit]) -> dict[str, object]:
    """Record values the tailor typed, with their notes, and measure again."""
    try:
        save_overrides(
            session_id,
            {name: {"value": edit.value, "note": edit.note} for name, edit in values.items()},
        )
        measurements = measure_session(session_id)
        marks = load_landmarks(session_dir(session_id) / "landmarks.json")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    export(session_id, measurements)
    return _measurement_payload(marks, measurements)


@app.get("/review/{session_id}")
async def review_page(session_id: str) -> FileResponse:
    """Serve the review page; the script reads the id from the path."""
    return FileResponse(ROOT / "web" / "review.html", media_type="text/html")


@app.get("/api/sessions/{session_id}/overlay/{name}")
async def overlay_png(session_id: str, name: str) -> FileResponse:
    """Serve one debug overlay PNG."""
    path = session_dir(session_id) / "debug" / f"{name}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"no overlay called {name}")
    return FileResponse(path, media_type="image/png")


@app.get("/api/sessions/{session_id}/export")
async def export_body(session_id: str) -> FileResponse:
    """Serve ``body_measurements.yaml`` for Objective 1.2."""
    return _yaml_file(session_id, "body_measurements.yaml")


@app.get("/api/sessions/{session_id}/report")
async def export_report(session_id: str) -> FileResponse:
    """Serve the provenance sidecar."""
    return _yaml_file(session_id, "measurement_report.yaml")


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


@app.get("/api/sessions")
async def list_sessions_route(status: str | None = None) -> list[dict[str, object]]:
    """List sessions from the index, newest first, optionally by status."""
    return store.list_sessions(status)


@app.post("/api/sessions/{session_id}/confirm")
async def confirm_session(session_id: str) -> dict[str, object]:
    """Mark the measurements checked by a tailor, which starts the retention clock."""
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


@app.get("/api/export.csv")
async def export_csv_route() -> FileResponse:
    """Serve one row per confirmed session, for Objective 2 analysis."""
    path = store.data_dir() / "export.csv"
    store.export_csv(path)
    return FileResponse(path, media_type="text/csv", filename="export.csv")


# Mounted last so the /api routes above win.
app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
