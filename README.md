# Anthropometric Surface Reconstruction and Mesh Flattening

A web application that accepts `body_measurements.yaml`, builds a measurement-constrained
proxy torso, applies garment ease, segments the garment surface along a predetermined seam
network, flattens the resulting panels, validates their geometry, and exports a
GarmageSet-like pattern with production files.

**The pattern is produced from the reconstructed 3D surface.** The generation pipeline does
not load or deform a 2D master block, use sequential drafting formulas, retrieve a pattern,
or use a learned pattern generator. Measurements are scalar inputs, not a full body scan:
the result is an assumption-recorded proxy surface, not an exact reconstruction of an
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

## Benchmark

`benchmark/` measures the Chapter 3 metrics for this design. It wraps the production
pipeline rather than reimplementing any of it.

```powershell
python benchmark/run_design_3.py --smoke   # 1 case, 3 + 3 trials: proves the harness
python benchmark/run_design_3.py           # the compliant run
python benchmark/maintainability_check.py  # Metric 5, on its own
python benchmark/assemble_metrics.py       # final metrics, from the raw result files
python benchmark/environment.py            # environment record, on its own
```

| File | Purpose |
|---|---|
| `benchmark/config-design-3.yaml` | Frozen cases, trial counts, thread counts, Metric 5 allow-list, output paths |
| `benchmark/_child.py` | One trial in one fresh process: memory pass or time pass |
| `benchmark/run_design_3.py` | Spawns every trial and writes the trial, diagnostics, failure and summary files |
| `benchmark/maintainability_check.py` | Metric 5, Radon 6.0.1, source-line-weighted |
| `benchmark/assemble_metrics.py` | Generates `design_3_final_metrics.json`; never hand-edited |
| `benchmark/environment.py` | Environment record and configuration fingerprint |
| `fixtures/measurements/` | Frozen case fixtures, pseudonymous, approved |
| `fixtures/garments/` | Frozen garment configuration, one for every case |

Every trial runs in a fresh child process and the memory pass and the time pass are
separate passes, so no trial inherits another's allocator state and tracemalloc never
inflates a timing. A trial that fails stays in `design_3_trials.csv` with its error code
and is repeated in `design_3_failures.jsonl`; nothing is retried or replaced.

## The common program interface

`backend/app/design3_interface.py` is the one function every Design 3 consumer calls:

```python
from app.design3_interface import generate_pattern

pattern = generate_pattern(measurements_dict, garment_dict)  # millimetres
```

It is an adapter over `pipeline.generate`, not a second implementation: no geometry,
solver, weight or tolerance is defined there, so `config/reconstruction.yaml` still governs
every number that reaches the pattern. The returned object carries the schema and units,
panels and identifiers, ordered seam-line boundary entities, cutting lines and allowances,
grainlines, fold edges, notches, labels, darts, stitch-pair identifiers, the validated
report and provenance. Every refusal raises `Design3Failure` with a code from
`FAILURE_CODES`.

`run_design_3(measurements, garment, dxf_output)` is the same work plus the common
validator and the common DXF exporter, which is the benchmark's measured boundary.

## Measured metrics

`results/design_3_final_metrics.json` is generated from the raw files. Two of the five
Chapter 3 metrics are null, deliberately:

- **Metric 2, feature completeness** — not measured. It needs a frozen, approved
  `feature_manifest.csv`, and this repository does not have one. Darts and notches are
  modelled throughout the backend but never populated by the pipeline, so a manifest
  written today would either overstate completeness or have to declare them
  non-applicable; that is a scoping decision, not a measurement.
- **Metric 4, pattern-dimension MAE** — not measured. It needs reference seam-line
  dimensions from independently approved patterns for the garment this design generates.
  The only reference in the repository, `fixtures/garmageset-reference.json`, is a
  19-panel skirt-and-dress set, not a torso bodice, and a reference synthesised from a
  drafting oracle would not be an independent approval. Reporting a number the protocol
  cannot support is worse than reporting none.

Metrics 1, 3 and 5 are measured by the harness. See
`results/design_3_case_summary.csv` for per-case medians and interquartile ranges, and
`results/design_3_environment.json` for the interpreter, CPU, libraries and configuration
hash the numbers were produced on.
