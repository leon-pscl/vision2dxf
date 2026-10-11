# Formula-based drafting

A containerised web app that reads camera-derived anthropometric measurements
(`body_measurements.yaml`), resolves what each field *means*, drafts a basic bodice from
explicit equations, validates the geometry, and exports a GarmageSet-like pattern plus
full-scale production files.

No master pattern is loaded or transformed. No field meaning is inferred from its name.

## Run it

```bash
docker compose up --build          # http://localhost:8000
```

```bash
docker build --target test -t formula-drafting:test . && docker run --rm formula-drafting:test
cd frontend && npm install && npm test        # viewer maths + generation gate (vitest)
```

Local development without Docker:

```bash
cd frontend && npm install && npm run build       # emits backend/app/static
cd backend && pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

## The Measurement Semantics Layer

Raw YAML values never reach a drafting equation. The pipeline is:

```text
body_measurements.yaml
      ↓  YAML syntax validation + strict typing
Measurement Semantics Resolution      config/measurement-semantics.yaml
      ↓  Unit and representation normalization
Range and Consistency Validation      blocking schema rules
      ↓  Operator review and confirmation
DraftingMeasurements (adapter)        descriptive names, millimetres, degrees
      ↓
Formula-Based Drafting Engine
      ↓
Pattern Geometry → GarmageSet-Like JSON
```

### Two levels of settings

| Level | File | Who | Changes meaning? |
|---|---|---|---|
| Schema configuration | `backend/app/config/measurement-semantics.yaml` | developers / researchers | yes — it *is* the meaning |
| Measurement review | the UI | operator | no — values and confirmations only |

An operator may pick one of the interpretations the schema offers, or correct a numeric
value. They cannot invent a definition, so two uploads of identically structured YAML are
interpreted identically, or `schema_version` says why not.

### Field statuses

| Status | Meaning | Blocks generation |
|---|---|---|
| `CONFIRMED` | pinned by the schema | no |
| `DERIVED` | computed by a recorded equation | no |
| `ASSUMED` | operator confirmed an unpinned definition | no |
| `OPTIONAL` | not required by the selected garment | no |
| `MISSING` | required and absent | **yes** |
| `AMBIGUOUS` | definition not confirmed | **yes** |
| `INVALID` | rejected by the schema | **yes** |

`POST /api/patterns/generate` and every export endpoint answer `409 semantics_unresolved`
with the blocking list while a required field is unresolved. The Generate button is
disabled under the same condition.

### Field categories

Circumference, width, direct length, vertical level, relative vertical distance, landmark
spacing and angle. Each category declares required metadata; a field missing it is a
developer error surfaced as `schema_configuration_error`, not a silent default.

Three vertical distinctions are enforced, not assumed:

1. **Absolute level** — `bust_line`, `waist_line`, `waist_over_bust_line` (origin declared).
2. **Relative vertical distance** — `vert_bust_line`, `armscye_depth` (start/end landmarks).
3. **Garment drafting depth** — produced by the adapter, e.g.
   `bust_to_waist_vertical = |neck_to_waist_level − neck_to_bust_level|`.

`waist_line` is never used directly as the bodice length. If two vertical fields declare
different origins, generation stops: an inferred-origin problem is a **blocking
schema-validation message**, not a warning.

### The drafting input adapter

Drafting equations only ever address these (`backend/app/drafting/semantics.py`):

```python
class DraftingMeasurements(BaseModel):
    bust_circumference, waist_circumference, hip_circumference, underbust_circumference
    shoulder_width_full, back_width_full, waist_back_width_full, hip_back_width_full
    neck_width_half
    neck_to_bust_level, neck_to_waist_level, neck_to_bust_apex_vertical
    bust_to_waist_vertical, waist_to_hip_vertical, armscye_depth_vertical
    bust_apex_spacing_full, buttock_spacing_full
    shoulder_slope_degrees
```

`body["vert_bust_line"]` has no route into an equation; `neck_to_bust_apex_vertical` does.
A test asserts the adapter model exposes no source field name.

## Workflow

Upload YAML → **Field definitions** → Review measurements → Garment & ease → Generate →
Inspect → Validate → Export.

The **Field definitions** screen is the gate: field, value, unit, interpreted meaning,
status, plus view definition / confirm interpretation / correct value / return to
measurement capture / download validation report. There is no override that hides an
ambiguity — confirming an unpinned definition records it as `ASSUMED` with
`confirmed_by: operator`.

## Configuration

| File | Holds |
|---|---|
| `backend/app/config/measurement-semantics.yaml` | what every field means, candidates, origins, ranges, cross-field blocking rules |
| `backend/app/config/measurements.yaml` | the camera system's own field dictionary |
| `backend/app/config/drafting.yaml` | drafting constants, ease presets, seam allowances, tolerances, paper sizes |

## The drafting rules that shape the panels

Frame: origin at the centre neck point, `+x` toward the side seam, `+y` down, unit mm.

```
front_bust_half  = (bust + ease - back_width_full) / 2      back_bust_half  = back_width_full / 2
front_waist_half = (waist + ease - waist_back_width_full) / 2 back_waist_half = waist_back_width_full / 2
neck_shoulder_x  = neck_width_half * neck_shoulder_share
shoulder_seam    = shoulder_width_full / 2 - shoulder_neck_offset
shoulder_tip     = neck_shoulder + shoulder_seam * (cos slope, sin slope)
underarm_y       = shoulder_tip_y + armscye_depth_vertical
bust_apex_x      = front_bust_half - bust_apex_inset_ratio * bust_apex_spacing_full
dart_intake      = dart_intake_ratio * bust_apex_spacing_full
```

`der.shoulder_slope_deg` is the only clamped input and it always reports the original
confirmed value as an assumption. Everything else is reported, never silently changed.

## Fixture-compatible defaults

`sample/000c79de6519526691ff7b988eb6b92a.json` is the schema fixture. Keys whose meaning
cannot be derived from the fixture alone are preserved verbatim and listed in
`drafting.yaml → fixture_defaults` (echoed into `generation-manifest.json`):

| Key | Value | Why |
|---|---|---|
| `panel.particleDistance` | `6` | constant in every fixture panel |
| `panel.collisionLayer` | `0` | constant in every fixture panel |
| `seqEdge.type` | `3` | constant in every fixture contour |
| `seqEdge.circleType` | `0` | constant in every fixture contour |
| `edge.bezierPoints` | `[[0,0,0],[0,0,0]]` | constant in every fixture edge |

Two representations were confirmed by inspecting the fixture rather than guessed:

* `edge.controlPoints` is the **flattened polyline** of the edge — two endpoints for a
  straight edge, near-uniform samples for a curve (fixture edges carry 33 or 51 evenly
  spaced points). Sampling spacing is `sample_spacing_mm`.
* `panel.center` is the **workspace placement** of a panel whose local geometry is centred
  on its own bounding box; world position = local + `center`.

Stitches are stored as the fixture stores them: a pair of records, one forward along its
edge and one backward, with `param` normalised to `[0, 1]`. A dart is emitted as its own
closed contour inside `seqEdges`, next to the panel's outer contour.

## Generation manifest

`generation-manifest.json` records `measurement_schema_version`, `definition_version`,
`software_version`, `generated_at`, units, the coordinate convention, per-field
interpretations (meaning, status, confirmed_by, candidate), the full field records
(original value, normalized value, conversion, corrections), derived measurements with
their equations, drafting measurements, assumptions, warnings, settings, tolerances and the
full rule trace.

## API

```
POST /api/measurements/validate            { filename, content }            -> rows (syntax only)
GET  /api/measurements/semantics/schema                                   -> registry + statuses
POST /api/measurements/semantics           { yamlText, confirmations, corrections }
POST /api/measurements/report              -> measurement-report.json
POST /api/patterns/generate                -> pattern + trace + validation + preview
POST /api/patterns/validate                { pattern }                     -> closure issues
POST /api/patterns/render                  -> SVG scene
POST /api/patterns/export/pdf | dxf | svg  -> full-scale production files
POST /api/patterns/export/bundle           -> every file as a zip
GET  /api/config                           -> presets, allowances, paper, tolerances, schema version
```

Errors are structured: `{"error": {"code", "message", "details": [...]}}`.

## Validation

Geometry checks block export: non-finite coordinates, missing edges, fewer than three
boundary vertices, open boundaries, endpoint discontinuity, zero-length edges,
self-intersecting boundaries, duplicate ids, invalid stitch references, stitch parameters
outside `[0,1]`, darts outside their panel, paired-seam mismatch, invalid cutting
contours. Tolerances live in `drafting.yaml → tolerances`.

## Pattern viewer

One server-rendered SVG with a `<g data-layer>` group per layer, so the UI toggles
visibility without a round trip and the preview is by construction the same geometry as
the PDF and DXF. Zoom, pan, fit-to-view and a click-two-points ruler; edge lengths come
from the internal model, not from screen pixels.

## Known limits

* `offset_polylines` builds concave joins with the same construction as convex ones and
  does not repair self-intersecting offsets; validation reports them instead.
* Ten of the 26 fields ship as `ambiguous` with no authoritative definition, because the
  samples do not determine their convention (`shoulder_w` is 17.1 in one profile and 35.1
  in the other). Each upload therefore starts blocked until an operator confirms. This is
  the intended behaviour, not a defect — pin a field to `confirmed` in the schema once the
  camera subsystem's convention is known.
* `fixtures/body_measurements_pascual.yaml` is rejected as `INVALID` (`bust_line` 64.0 cm,
  `waist_line` 76.9 cm fall outside the declared schema ranges). It is a validation
  fixture, not a body; it is not a geometry golden.
* Sleeves are out of scope until the bodice passes acceptance, as the design sheet says.
