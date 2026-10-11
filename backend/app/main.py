"""Formula-based drafting app: FastAPI serving the API and the built frontend."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .api.routes import router
from .drafting.measurements import MeasurementError
from .drafting.semantics import SemanticError

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Formula drafting", version="1.0.0")
app.include_router(router)


@app.exception_handler(MeasurementError)
async def measurement_error_handler(_request, exc: MeasurementError):
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=422,
        content={"error": {"code": "invalid_yaml", "message": "Measurement import failed.",
                           "details": exc.errors}},
    )


@app.exception_handler(SemanticError)
async def semantic_error_handler(_request, exc: SemanticError):
    """A broken schema configuration is a developer error, surfaced without a stack trace."""
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=500,
        content={"error": {"code": "schema_configuration_error",
                           "message": str(exc),
                           "details": [{"field": exc.field_name or "schema",
                                        "message": "Fix config/measurement-semantics.yaml"}]}},
    )


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


if STATIC_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def spa(full_path: str):
        candidate = STATIC_DIR / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(STATIC_DIR / "index.html")
