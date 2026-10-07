"""FastAPI app. Session state is an in-memory dict; there is no database.

User images live only as base64 strings inside those dicts. Nothing here writes
an uploaded image to disk.
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from contracts import (  # noqa: E402
    CaptureInput,
    DraftRoute,
    GarmentSpec,
    GarmentType,
    LandmarkResult,
    LandmarkUpdate,
    LandmarksSource,
    MeasurementBackend,
    MeasurementResult,
    ModelsResponse,
    SegmentationInput,
    SessionState,
    StepRequest,
    StepStatus,
)
from adapters.mock._common import image_size  # noqa: E402
from registry import get_adapter, load_adapters_yaml, load_config  # noqa: E402

app = FastAPI(title="vision2dxf", version="1.0.0")

_SERVER = load_config().get("server", {})
# An allow-list, not "*": there is no auth on any endpoint, so a wildcard would
# let any page the user visits call a locally running instance.
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(_SERVER.get("cors_origins", [])),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# session_id -> SessionState. No database and no expiry in phase 1, but bounded
# so a long-running process does not accumulate base64 photos forever.
SESSIONS: dict[str, SessionState] = {}

# session_id -> raw step inputs (images, height, weight). Later steps read from
# here instead of the browser re-uploading. In memory only, never on disk.
INPUTS: dict[str, dict[str, dict]] = {}

MAX_SESSIONS = int(_SERVER.get("max_sessions", 50))

# step number -> adapter key. Step 6 resolves per backend instead.
STEP_KEYS = {
    2: "capture",
    3: "segmentation",
    4: "calibration",
    7: "validation",
    8: "drafting",
    9: "production",
    10: "drape",
}


# --------------------------------------------------------------------------
# session plumbing
# --------------------------------------------------------------------------


def _session(sid: str) -> SessionState:
    s = SESSIONS.get(sid)
    if s is None:
        raise HTTPException(404, f"unknown session {sid}")
    return s


def _store(sid: str, s: SessionState, step: int, payload: dict, result: Any, key: str = "") -> None:
    """Record the input and the result, then update status and the mock flag."""
    store_key = str(step) if not key else key
    INPUTS.setdefault(sid, {})[store_key] = payload
    stored = result.model_dump()
    s.results[store_key] = stored
    s.status[str(step)] = StepStatus(
        status="warning" if stored.get("warnings") else "done",
        is_mock=bool(stored.get("is_mock")),
    )
    s.any_mock = any(st.is_mock for st in s.status.values())


def _need(s: SessionState, step: int) -> dict:
    got = s.results.get(str(step))
    if got is None:
        raise HTTPException(400, f"step {step} has not been run")
    return got


def _input(sid: str, step: int) -> dict:
    got = INPUTS.get(sid, {}).get(str(step))
    if got is None:
        raise HTTPException(400, f"step {step} has not been run")
    return got


def _check_unlocked(s: SessionState, step: int) -> None:
    """Step 6 onward is gated on a confirmed step 5."""
    if step >= 6 and not s.landmarks_confirmed:
        raise HTTPException(400, "confirm the landmarks in step 5 before continuing")


# --------------------------------------------------------------------------
# routes
# --------------------------------------------------------------------------


@app.post("/api/session")
def create_session() -> dict:
    sid = uuid.uuid4().hex
    SESSIONS[sid] = SessionState(
        session_id=sid,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    INPUTS[sid] = {}
    _evict_old_sessions()
    return {"session_id": sid}


def _evict_old_sessions() -> None:
    """Drop the oldest sessions past the cap. dicts keep insertion order."""
    while len(SESSIONS) > MAX_SESSIONS:
        oldest = next(iter(SESSIONS))
        SESSIONS.pop(oldest, None)
        INPUTS.pop(oldest, None)


@app.get("/api/session/{sid}")
def get_session(sid: str) -> SessionState:
    return _session(sid)


@app.get("/api/models")
def list_models() -> ModelsResponse:
    return ModelsResponse(
        adapters=dict(load_adapters_yaml()),
        device=str(load_config().get("device", "cpu")),
    )


@app.post("/api/session/{sid}/step/{step}")
def run_step(sid: str, step: int, body: StepRequest | None = None) -> dict:
    s = _session(sid)
    body = body or StepRequest()
    t0 = time.perf_counter()
    result = _dispatch(sid, s, step, body)
    out = result.model_dump()
    out["_elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return out


@app.get("/api/session/{sid}/export.yaml")
def export_yaml(sid: str):
    from fastapi.responses import Response

    s = _session(sid)
    step7 = s.results.get("7")
    if step7 is None:
        raise HTTPException(400, "run step 7 before exporting")
    return Response(
        content=step7["yaml"],
        media_type="application/x-yaml",
        headers={"Content-Disposition": 'attachment; filename="measurements.yaml"'},
    )


# --------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------


def _dispatch(sid: str, s: SessionState, step: int, body: StepRequest):
    handlers = {
        1: lambda: _step1(sid, s, body),
        2: lambda: _step2(sid, s, body),
        3: lambda: _step3(sid, s, body),
        4: lambda: _step4(sid, s, body),
        5: lambda: _step5(sid, s, body),
        6: lambda: _step6(sid, s, body),
        7: lambda: _step7(sid, s, body),
        8: lambda: _step8(sid, s, body),
        9: lambda: _step9(sid, s, body),
        10: lambda: _step10(sid, s, body),
    }
    handler = handlers.get(step)
    if handler is None:
        raise HTTPException(404, f"no such step {step}")
    if step >= 6:
        _check_unlocked(s, step)
    return handler()


# step 1 - garment type


def _step1(sid: str, s: SessionState, body: StepRequest) -> GarmentSpec:
    raw = body.payload.get("garment_type")
    if raw is None:
        raise HTTPException(400, "payload.garment_type is required")
    try:
        garment = GarmentType(raw)
    except ValueError:
        raise HTTPException(400, f"unknown garment_type {raw!r}")
    result = get_adapter("garment_spec").run(garment)
    _store(sid, s, 1, body.payload, result)
    return result


# step 2 - capture


def _step2(sid: str, s: SessionState, body: StepRequest) -> Any:
    p = body.payload
    front, side = p.get("front_image", ""), p.get("side_image", "")
    if not front or not side:
        raise HTTPException(400, "payload needs front_image and side_image")
    if not p.get("height_cm"):
        raise HTTPException(400, "payload.height_cm is required")

    inp = CaptureInput(
        front_image=front,
        side_image=side,
        height_cm=float(p["height_cm"]),
        weight_kg=float(p["weight_kg"]) if p.get("weight_kg") else None,
    )
    result = get_adapter("capture").run(inp)
    _store(sid, s, 2, p, result)
    return result


# step 3 - segmentation


def _step3(sid: str, s: SessionState, body: StepRequest) -> Any:
    p = body.payload
    if not p.get("image"):
        raise HTTPException(400, "payload.image is required")
    inp = SegmentationInput(
        image=p["image"],
        points=[tuple(pt) for pt in p.get("points", [])],
        box=tuple(p["box"]) if p.get("box") else None,
        text=p.get("text"),
    )
    result = get_adapter("segmentation").run(inp)
    _store(sid, s, 3, p, result)
    return result


# step 4 - calibration. Reads the step 2 images back out of memory for their size.


def _step4(sid: str, s: SessionState, body: StepRequest) -> Any:
    _need(s, 2)
    cap = _input(sid, 2)
    fw, fh = image_size(cap["front_image"])
    sw, sh = image_size(cap["side_image"])
    result = get_adapter("calibration").run(
        {"front": (fw, fh), "side": (sw, sh)}, float(cap["height_cm"])
    )
    _store(sid, s, 4, body.payload, result)
    return result


# step 5 - landmarks. Browser drags arrive as updates and override the adapter.


def _step5(sid: str, s: SessionState, body: StepRequest) -> LandmarkResult:
    p = body.payload
    _need(s, 2)
    cap = _input(sid, 2)
    fw, fh = image_size(cap["front_image"])

    if p.get("image"):
        image = p["image"]
        size = {**p.get("size", {}), "front_width": fw, "front_height": fh}
    else:
        image = cap["front_image"]
        size = {"front_width": fw, "front_height": fh}

    result = get_adapter("landmarks").run(image, size)

    for raw in p.get("updates", []) or []:
        u = LandmarkUpdate(**raw)
        if u.name in result.points:
            pt = result.points[u.name]
            pt.x, pt.y = u.x, u.y
            pt.source = LandmarksSource.USER
            pt.confidence = 1.0

    result.confirmed_by_user = bool(p.get("confirm", False)) or s.landmarks_confirmed
    _store(sid, s, 5, p, result)
    s.landmarks_confirmed = s.landmarks_confirmed or result.confirmed_by_user
    return result


# step 6 - measurement, backend 6a or 6b


def _step6(sid: str, s: SessionState, body: StepRequest) -> MeasurementResult:
    backend = body.backend or MeasurementBackend.REGRESSION
    cap = _input(sid, 2)
    key = (
        "measurement_regression"
        if backend is MeasurementBackend.REGRESSION
        else "measurement_body_model"
    )
    result = get_adapter(key).run(
        float(cap["height_cm"]),
        float(cap["weight_kg"]) if cap.get("weight_kg") else None,
    )

    # a garment only cares about the measurements it declares
    spec = _need(s, 1)
    required = set(spec["required_measurements"])
    result.measurements = {k: v for k, v in result.measurements.items() if k in required}

    _store(sid, s, 6, body.payload, result, key=f"6:{backend.value}")
    s.results["6"] = result.model_dump()
    return result


# steps 7 to 10 - all read the step 1 and step 6 results


def _step7(sid: str, s: SessionState, body: StepRequest) -> Any:
    result = get_adapter("validation").run(
        GarmentSpec(**_need(s, 1)), MeasurementResult(**_need(s, 6))
    )
    _store(sid, s, 7, body.payload, result)
    return result


def _step8(sid: str, s: SessionState, body: StepRequest) -> Any:
    route = body.payload.get("route", DraftRoute.BLOCK.value)
    try:
        DraftRoute(route)
    except ValueError:
        raise HTTPException(400, f"unknown route {route!r}")
    result = get_adapter("drafting").run(
        GarmentSpec(**_need(s, 1)), MeasurementResult(**_need(s, 6)), route
    )
    _store(sid, s, 8, body.payload, result)
    return result


def _step9(sid: str, s: SessionState, body: StepRequest) -> Any:
    result = get_adapter("production").run(
        GarmentSpec(**_need(s, 1)), MeasurementResult(**_need(s, 6))
    )
    _store(sid, s, 9, body.payload, result)
    return result


def _step10(sid: str, s: SessionState, body: StepRequest) -> Any:
    result = get_adapter("drape").run(GarmentSpec(**_need(s, 1)))
    _store(sid, s, 10, body.payload, result)
    return result
