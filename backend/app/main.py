"""Anthropometric surface reconstruction and mesh flattening: FastAPI serving the
API and the built frontend."""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .api.routes import router
from .drafting.measurements import MeasurementError
from .drafting.semantics import SemanticError
from .pipeline import GenerationBlocked

logger = logging.getLogger("app")
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Anthropometric surface reconstruction and mesh flattening",
              version="3.0.0")
app.include_router(router)


@app.exception_handler(MeasurementError)
async def measurement_error_handler(_request: Request, exc: MeasurementError):
    return JSONResponse(
        status_code=422,
        content={"error": {"code": "invalid_yaml", "message": "Measurement import failed.",
                           "details": exc.errors}},
    )


@app.exception_handler(GenerationBlocked)
async def generation_blocked_handler(_request: Request, exc: GenerationBlocked):
    """A stage refused. The reason is already structured; nothing is invented here."""
    return JSONResponse(
        status_code=409,
        content={"error": {"code": exc.code, "message": exc.message, "details": exc.details},
                 "stage": exc.stage},
    )


@app.exception_handler(SemanticError)
async def semantic_error_handler(_request: Request, exc: SemanticError):
    """A broken schema configuration is a developer error, surfaced without a stack trace."""
    return JSONResponse(
        status_code=500,
        content={"error": {"code": "schema_configuration_error",
                           "message": str(exc),
                           "details": [{"field": exc.field_name or "schema",
                                        "message": "Fix config/measurement-semantics.yaml"}]}},
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    """The client never receives a raw traceback; the operator still gets one in the log."""
    logger.exception("unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": {"code": "internal_error",
                           "message": "The request could not be completed.",
                           "details": [{"field": type(exc).__name__, "message": str(exc)[:400]}]}},
    )


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


if STATIC_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/{full_path:path}")
    def static_file(full_path: str):
        """Serve a built asset, or 404.

        Anything else is a 404 on purpose: probes such as /health or /v1/models must not
        be answered with 200 and a page of HTML.
        """
        candidate = (STATIC_DIR / full_path).resolve()
        if full_path and candidate.is_file() and STATIC_DIR.resolve() in candidate.parents:
            return FileResponse(candidate)
        return JSONResponse(
            status_code=404,
            content={"error": {"code": "not_found", "message": f"No route for /{full_path}", "details": []}},
        )
