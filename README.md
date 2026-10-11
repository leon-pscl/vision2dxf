# Formula-based drafting

A containerised web app that reads camera-derived anthropometric measurements
(`body_measurements.yaml`), drafts a basic bodice from explicit equations, validates the
geometry, and exports a GarmageSet-like pattern plus full-scale production files.

No master pattern is loaded or transformed. Every principal landmark is produced by a
registered drafting rule and carries its rule id, inputs and source note.

## Run it

```bash
docker compose up --build          # http://localhost:8000
```

```bash
docker build --target test -t formula-drafting:test . && docker run --rm formula-drafting:test   # 112 backend tests
cd frontend && npm install && npm test        # viewer maths (vitest)
```

Local development without Docker:

```bash
cd frontend && npm install && npm run build       # emits backend/app/static
cd backend && pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

## Workflow

Upload YAML → review measurements → garment and ease → generate → inspect → validate → export.

## Drafting pipeline

```
YAML → parse+validate → measurement confirmation → ease → derived dimensions
     → construction lines → named landmarks → boundary curves → panel assembly
     → stitch pairs → geometry validation → GarmageSet JSON → preview + exports
```

| Area | Where |
|---|---|
| typed measurement model | `backend/app/drafting/measurements.py` |
| ease presets / overrides | `backend/app/drafting/ease.py` |
| rule registry | `backend/app/drafting/rules.py`, `derived_dimensions.py`, `landmarks.py` |
| geometry model | `backend/app/drafting/geometry.py`, `curves.py` |
| panels and darts | `backend/app/drafting/panels.py`, `stitches.py` |
| cutting lines | `backend/app/drafting/cutting.py` |
| validation | `backend/app/validation/geometry_checks.py` |
| serialisation | `backend/app/serialization/garmageset.py` |
| rendering | `backend/app/rendering/svg.py`, `png.py` |
| exports | `backend/app/export/pdf.py`, `dxf.py` |

## Configuration

No unexplained constant lives inside a geometry function.

* `backend/app/config/measurements.yaml` — the measurement dictionary: unit, origin,
  direction, landmark, interpretation, plausible range and the rules each value feeds.
  Similarly named vertical fields do **not** share an origin (`bust_line` and
  `vert_bust_line` are both measured down from the neck point but sit at different
  levels; `hips_line` is measured up from the crotch and is never subtracted from them).
* `backend/app/config/drafting.yaml` — drafting constants, ease presets, seam allowances,
  tolerances and paper sizes.

## The drafting rules that shape the panels

Frame: origin at the centre neck point, `+x` toward the side seam, `+y` down, unit mm.

```
front_bust_half  = (bust + ease - back_width) / 2          back_bust_half  = back_width / 2
front_waist_half = (waist + ease - waist_back_width) / 2  back_waist_half = waist_back_width / 2
neck_shoulder_x  = neck_w * neck_shoulder_share
shoulder_seam    = shoulder_w - shoulder_neck_offset
shoulder_tip     = neck_shoulder + shoulder_seam * (cos slope, sin slope)
underarm_y       = shoulder_tip_y + armscye_depth
bust_apex_x      = front_bust_half - bust_apex_inset_ratio * bust_points
dart_intake      = dart_intake_ratio * bust_points
```

`shoulder_slope_deg` is the only clamped input: a steep measured slope drives the
underarm point through the bust line, so the draft clamps it into
`ranges.shoulder_slope_deg` and reports the original value as an assumption. Everything
else is reported, never silently changed.

## Fixture-compatible defaults

`sample/000c79de6519526691ff7b988eb6b92a.json` is the schema fixture. Keys whose meaning
cannot be derived from the fixture alone are preserved verbatim and listed in
`drafting.yaml → fixture_defaults` (they are echoed in `generation-manifest.json`):

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
* `panel.center` is the **workspace placement** of a panel whose local geometry is
  centred on its own bounding box; world position = local + `center`.

Stitches are stored as the fixture stores them: a pair of records, one running forward
along its edge and one backward, with `param` normalised to `[0, 1]`.

A dart is emitted as its own closed contour inside `seqEdges`, next to the panel's outer
contour — the only closed-boundary representation the schema offers.

## Validation

Critical checks block export: non-finite coordinates, missing edges, fewer than three
boundary vertices, open boundaries, endpoint discontinuity, zero-length edges,
self-intersecting boundaries, duplicate ids, invalid stitch references, stitch parameters
outside `[0,1]`, darts outside their panel, paired-seam mismatch and invalid cutting
contours.

Tolerances live in `drafting.yaml → tolerances`: endpoint continuity 0.1 mm, panel
closure 0.1 mm, duplicate point 0.01 mm, zero self-intersections, seam mismatch
`max(2 mm, 0.5 %)`.

## API

```
POST /api/measurements/validate   { filename, content }            -> review rows + warnings
POST /api/patterns/generate       { yamlText, filename, settings } -> pattern + trace + validation + preview
POST /api/patterns/validate       { pattern }                     -> schema + closure issues
POST /api/patterns/render         { yamlText, filename, settings } -> SVG scene
POST /api/patterns/export/pdf     -> pattern-a0.pdf (1:1)
POST /api/patterns/export/dxf     -> pattern.dxf (1:1, mm)
POST /api/patterns/export/svg     -> pattern-preview.svg
POST /api/patterns/export/bundle  -> every file as a zip
GET  /api/config                  -> presets, allowances, paper, tolerances, view layers
```

Errors are structured: `{"error": {"code", "message", "details": [{"field", "message", "code"}]}}`.

## Pattern viewer

One server-rendered SVG with a `<g data-layer>` group per layer, so the UI toggles
visibility without a round trip and the preview is by construction the same geometry as
the PDF and DXF. Zoom, pan, fit-to-view and a click-two-points ruler are implemented in
the frontend; edge lengths come from the internal model, not from screen pixels.

## Known limits

* `offset_polylines` builds concave joins with the same construction as convex ones and
  does not repair self-intersecting offsets; the validation check reports them instead.
* Sleeves are out of scope until the bodice passes acceptance, as the design sheet says.
* `sample/body_measurements_pascual.yaml` contains implausible values (bust == waist ==
  hips, `hips_line: 0.1`). It imports cleanly and reports warnings; it is a validation
  fixture, not a body.
