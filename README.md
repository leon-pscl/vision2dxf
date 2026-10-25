# Anthropometric Surface Reconstruction and Mesh Flattening

A web application that accepts `body_measurements.yaml`, builds a measurement-constrained
proxy torso, applies garment ease, segments the garment surface along a predetermined seam
network, flattens the resulting panels, validates their geometry, and exports a
GarmageSet-like pattern with production files.

**The pattern is produced from the reconstructed 3D surface.** The generation pipeline does
not load or deform a 2D master block, use sequential drafting formulas, retrieve a pattern,
or use a learned pattern generator. Measurements are scalar inputs, not a full body scan:
the result is an assumption-recorded proxy anthropometric surface, not an exact reconstruction of an
individual's body.

## Scope and workflow

The first release generates a basic torso garment with a fixed seam network. Its workflow is:

1. Upload and validate a `.yaml` or `.yml` file containing a top-level `body` mapping.
2. Review the source values, units, anatomical definitions, and any inferred values.
3. Confirm ambiguous field interpretations and the measurement unit.
4. Configure the cross-section model, mesh resolution, garment ease, seam preset,
   flattening weights, seam allowances, and paper output.
5. Generate and inspect the solved cross-sections, proxy torso, ease-adjusted surface,
   seam-segmented panels, flattened pattern, and geometry diagnostics.
6. Export the compatibility JSON, previews, and production files.

Anatomical origins and ordering are not silently repaired. Missing or inconsistent
definitions block generation and are reported for operator review. Every geometric
assumption is written to `generation-manifest.json`; it is kept out of the compatibility
pattern JSON.

## Generation architecture

```text
body_measurements.yaml
  → strict parsing and measurement semantics
  → confirmed anatomical levels
  → numerically solved ellipse/superellipse sections
  → proxy torso loft and constraint-preserving smoothing
  → interpolated circumference ease
  → predetermined semantic seam network
  → panel segmentation and constrained triangulation
  → LSCM initialization and ARAP flattening
  → boundary fitting and seam balancing
  → geometric validation
  → GarmageSet-like JSON, previews, and production exports
```

The surface and flattening implementation lives in `backend/app`. Reconstruction,
garment-surface, flattening, validation, serialization, rendering, and export are separate
modules. Computation runs on the backend; the frontend presents measurement review,
configuration, intermediate geometry, diagnostics, and exports.

## Run locally

### Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

The API and built frontend are served from `http://localhost:8000`. The backend API is
under `/api`; `/healthz` is the health check.

### Frontend

```powershell
cd frontend
npm install
npm run dev
```

For a production frontend build, run `npm run build`; Vite emits the static application
into `backend/app/static`, which FastAPI serves when present.

### Docker

```powershell
docker compose up --build
```

## API

Generation endpoints accept `yamlText`, `filename`, optional semantics `confirmations` and
numeric `corrections`, plus optional `settings`. Geometry is not generated until the unit is
confirmed in settings.

| Endpoint | Purpose |
|---|---|
| `POST /api/measurements/validate` | Strict YAML parsing and source-value review |
| `GET /api/measurements/semantics/schema` | Measurement definitions and candidate meanings |
| `POST /api/measurements/semantics` | Resolve meanings, units, and consistency |
| `POST /api/measurements/report` | Downloadable measurement-semantics report |
| `GET /api/config` | Versioned reconstruction, flattening, ease, and output settings |
| `POST /api/reconstruction/cross-sections` | Solved anatomical sections and residuals |
| `POST /api/reconstruction/proxy-surface` | Lofted and smoothed proxy torso mesh |
| `POST /api/garment/surface` | Ease-adjusted garment mesh and achieved circumferences |
| `POST /api/garment/segment` | Predetermined seam network and segmented surface panels |
| `POST /api/patterns/flatten` | Full generation result, diagnostics, preview, and pattern |
| `POST /api/patterns/validate` | Validate a supplied GarmageSet-like pattern document |
| `POST /api/patterns/render` | SVG preview and per-panel distortion heat maps |
| `POST /api/patterns/export/pdf` | Full-scale PDF |
| `POST /api/patterns/export/dxf` | Full-scale DXF in millimetres |
| `POST /api/patterns/export/svg` | Pattern preview SVG |
| `POST /api/patterns/export/bundle` | ZIP containing pattern, manifest, diagnostics, previews, and production files |
| `POST /api/patterns/debug/{artifact}` | Optional mesh and intermediate diagnostic downloads |

The flatten endpoint accepts `include_mesh: true` to return the dense flattened triangulation
and `include_surfaces: true` to return cross-sections, proxy and garment meshes, seam paths,
and segmented panels. These larger intermediate payloads are opt-in.

Generation refusals return structured errors with an HTTP status and the stage that blocked.
Unexpected server exceptions are logged server-side; raw tracebacks are not returned to
clients. Geometry endpoints include stage timings in their `progress` response. For the
current configured mesh resolution, generation is synchronous.

## Outputs

The ZIP bundle contains:

- `pattern.json` — GarmageSet-like panel and stitch data in millimetres.
- `generation-manifest.json` — source, measurement meanings, reconstruction settings,
  anatomical levels, assumptions, and validation summary.
- `flattening-diagnostics.json` and `validation-report.json`.
- `pattern-preview.svg` and `pattern-preview.png`.
- `pattern-a0.pdf` and `pattern.dxf`.
- `debug/` cross-sections, proxy/garment/segmented meshes, and flattened mesh.

The reference fixture in `fixtures/garmageset-reference.json` is validated by the typed
schema in `backend/app/schemas/garmage.py`.

## Configuration and assumptions

| File | Contents |
|---|---|
| `backend/app/config/measurement-semantics.yaml` | Field meanings, alternatives, origins, and consistency rules |
| `backend/app/config/measurements.yaml` | Input field dictionary |
| `backend/app/config/drafting.yaml` | Pattern tolerances, cutting output, paper, and fixture defaults |
| `backend/app/config/reconstruction.yaml` | Sections, levels, mesh density, ease, seams, flattening, and boundary-fit limits |

Assumed section depth ratios, the superellipse exponent, underbust placement, neckline
shape, ease conversion, and other inferred values are versioned in configuration and
reported in the manifest. A circumference ease is converted to a section scale at each
anatomical level; it is not applied as a uniform normal-distance offset.

## Tests

Backend tests:

```powershell
cd backend
python -m pytest
```

Frontend tests and production build:

```powershell
cd frontend
npm test
npm run build
```

The geometry tests include analytic plane, cylinder, truncated-cone, and sphere fixtures,
as well as reconstruction, seam, segmentation, flattening, serializer, validation, and
full-scale export checks.
