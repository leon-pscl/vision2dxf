# Parametric base-block transformation

A containerised web app that reads camera-derived anthropometric measurements
(`body_measurements.yaml`), resolves what each field *means*, deforms a **validated
two-dimensional pattern block** onto those measurements with constrained computational
geometry, verifies the result, and exports a GarmageSet-like pattern plus full-scale
production files.

The defining rule: an existing block is **preserved and transformed**. Nothing is
redrafted from a sequence of drafting formulas, no model is trained, and no
three-dimensional surface is created and flattened. No field meaning is inferred from its
name.

## Before you run it

**Download the GarmageSet pattern library and put it in `pattern_templates/local/` in this
repository.**
It is 2.9 GB of pattern JSON, one file per garment set, named after the hash of its own
contents. It is a developer prerequisite, not part of this repository: `.gitignore`
excludes `/pattern_templates/`, and a `pre-commit` hook refuses any file over 5 MB so it cannot be
committed by accident. Enable the hook once per clone:

```bash
git config core.hooksPath .githooks
```

The library ships as a Hugging Face bucket, [`leonpscl/GarmageSet-bucket`][bucket], and is
fetched with the `hf` CLI — on the bucket page, **Sync this bucket** gives the same two
commands:

[bucket]: https://huggingface.co/buckets/leonpscl/GarmageSet-bucket

```powershell
# once, from any shell: install the hf CLI
powershell -ExecutionPolicy ByPass -c "irm https://hf.co/cli/install.ps1 | iex"
```

```bash
cd pattern_templates
hf sync hf://buckets/leonpscl/GarmageSet-bucket ./local
```

`hf sync` writes into `pattern_templates/local/` and is re-runnable: it copies what is
missing or changed and leaves the rest alone.

The three catalogued base blocks are committed under `assets/base-blocks/templates/`, so
the application itself runs without the library. You need it to select a template the
catalog does not list (`template:<file>.json`), and `docker-compose.yml` mounts it
read-only at `/srv/templates` for exactly that.

> Parts of this repository were lost and rebuilt rather than recovered, including the
> whole of `transformation.yaml`. See [RECONSTRUCTION.md](RECONSTRUCTION.md) for every
> value that was chosen rather than carried over, before relying on a generated pattern.

## Run it

```bash
docker compose up --build -d       # http://localhost:8000
```

```bash
docker build --target test -t base-block-transform:test . && docker run --rm base-block-transform:test
cd frontend && npm install && npm test        # viewer maths + definitions gate (vitest)
```

Local development without Docker:

```bash
cd frontend && npm install && npm run build       # emits backend/app/static
cd backend && pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

## The base blocks

A base block is a **GarmageSet pattern file plus a semantic layer derived from it when it
is loaded**. The template file is the versioned asset: immutable, named after the hash of
its own contents, and pinned in `assets/base-blocks/catalog.yaml` by `sha256`. If the
bytes change the block is refused rather than silently becoming a different block.

Three blocks ship, and the operator chooses between them:

| Block | Boundary | Seams | Darts | Notes |
|---|---|---|---|---|
| `bodice-ref-a` | 10 edges per panel | 6 | none | carries internal waist, bust and chest reference lines |
| `bodice-ref-b` | 12 edges per panel | 4 | none | the simplest; shoulder seam split across two edges |
| `bodice-ref-c` | 16 front / 10 back | 8 | two front darts | denser curves, side seam split across two edges |

Any other template in the library can also be loaded at runtime as
`template:<file>.json`. Such a block is reported as **not reviewed** and carries the same
validation report, so an unusable template is refused rather than silently transformed.

The three catalogued templates are committed under `assets/base-blocks/templates/`, so the
catalog works on a fresh clone and its pinned hashes mean something. The full 2.9 GB
GarmageSet library is **not** versioned and **not** copied into the image; mount it at
`/srv/templates` (see `docker-compose.yml`) only if ad-hoc template selection is wanted.

### How the semantics are derived

A template states geometry, identifiers and seams. It does not state semantics: panel
labels are garment part names in Chinese and edge labels are mostly just numbers. Roles
are recovered from what the file *does* state — never from a label:

1. **Seams** come from the file's own stitch records. A seam joining two panels is a seam;
   one whose two sides are on the same panel is a dart. A seam spanning more than a quarter
   of the panel height is a side seam, the rest are shoulders.
2. **Stretches between seams** follow from the boundary order. Walking from the centre hem
   up one side: waist, side seam, armhole, shoulder, neckline; after the centre neck the
   same roles mirror back.
3. **Landmarks** are where one stretch hands over to the next, taken from the whole run of
   a role, so a seam split across several edges still yields one underarm per side.
4. **The centre line** is the axis of mirror symmetry. A mirrored torso panel crosses it at
   exactly two vertices — the centre neck and the centre hem — and the end the shoulder
   seams sit at is the neck. Nothing assumes which way up a template was drawn.
5. **The grainline** runs down that centre line.

A template that does not yield a complete, mirror-consistent torso block is refused with a
structured reason. See `assets/base-blocks/provenance.md`.

### A template panel is a whole panel

Template panels are a whole front and a whole back, mirrored about their own centre line —
not half patterns on a fold. So most landmarks come in a left/right pair, the centre line
is an interior axis rather than an edge, and a girth difference has to move the left side
left and the right side right. The influence map is therefore written in **semantic
directions** (`outward`, `down`, `up`) resolved against each panel's own frame, never raw
axes.

### The reference body is declared, not measured

A template carries no body measurements, and it cannot: it is a garment, and a garment is
longer, wider and lower-necked than the body it fits by whatever the pattern maker chose.

So the body each block is considered to fit is **declared** in `catalog.yaml` and resolved
through the same semantics layer as a target profile. Differences are then body-to-body —
which is what grading a block means — and the block keeps its own length, neckline and
design ease instead of having them read as measurement errors. The declared ease preset is
the ease the block already carries, so keeping that preset leaves its fit alone.

Separately, the block's **garment dimensions** are measured off its geometry, and those are
what achieved dimensions are compared against. Each states the definition it implements,
and the definitions are published with the block.

## The Measurement Semantics Layer

Raw YAML values never reach the transformation. The pipeline is:

```text
body_measurements.yaml
      ↓  YAML syntax validation + strict typing
Measurement Semantics Resolution      config/measurement-semantics.yaml
      ↓  Unit and representation normalization
Range and Consistency Validation      blocking schema rules
      ↓  Operator review and confirmation
DraftingMeasurements (adapter)        descriptive names, millimetres, degrees
      ↓
Target-versus-base differences        against the block's declared reference body
      ↓
Constrained 2D deformation            thin-plate spline, per panel
      ↓
Pattern Geometry → GarmageSet-Like JSON
```

### Two levels of settings

| Level | File | Who | Changes meaning? |
|---|---|---|---|
| Schema configuration | `backend/app/config/measurement-semantics.yaml` | developers / researchers | yes — it *is* the meaning |
| Transformation configuration | `backend/app/config/transformation.yaml` | developers / researchers | yes — weights, limits, influence map |
| Base-block catalog | `assets/base-blocks/catalog.yaml` | developers / researchers | yes — which blocks, which reference body |
| Measurement review | the UI | operator | no — values, confirmations, ease, limits |

### Field statuses

| Status | Meaning | Blocks transformation |
|---|---|---|
| `CONFIRMED` | pinned by the schema | no |
| `DERIVED` | computed by a recorded equation | no |
| `ASSUMED` | operator confirmed an unpinned definition | no |
| `OPTIONAL` | not required by the selected garment | no |
| `MISSING` | required and absent | **yes** |
| `AMBIGUOUS` | definition not confirmed | **yes** |
| `INVALID` | rejected by the schema | **yes** |

`POST /api/patterns/transform` and every export endpoint answer `409
semantics_unresolved` with the blocking list while a required field is unresolved.

## The transformation

```text
target YAML → validation and confirmation → base-block loading and validation
  → target-versus-base differences → target landmark calculation
  → constrained two-dimensional deformation → curve reconstruction
  → dart adjustment → seam balancing → geometric validation
  → GarmageSet-like serialisation → preview and production export
```

### Influence map

Every millimetre of movement is attributable to one measurement difference. A measurement
can only move the landmarks `transformation.yaml → influence` authorises, with a
configurable weight; a measurement with an empty op list records that absence
deliberately and is reported to the operator as unmapped.

Modes: `translate` (a semantic direction), `along` (away from a pivot), `rotate_about`,
and `set_angle`. `set_angle` runs last and keeps the landmark's horizontal distance, which
is how the across-shoulder width and the shoulder slope — two measurements that both claim
the shoulder tip — can each land exactly instead of spoiling each other.

### Solver

One documented method: a **thin-plate spline** per panel, solved in closed form. It
minimises `w_landmark · E_landmark + w_smoothness · E_smoothness` with
`lambda = w_smoothness / w_landmark`; `lambda = 0` interpolates the targets exactly. The
spline is solved in a frame normalised to the panel's extent, because `r² log r` on raw
millimetres is ill-conditioned for reasons that have nothing to do with the landmarks
being dependent.

The remaining terms of the design objective are enforced as **hard constraints** by a
named stage rather than traded off: seam by the balancing stage, symmetry and the centre
line by the constraints on the targets, area by the orientation guard, darts by the
topology checks. A solve that does not converge raises; the last iterate is never returned
as a pattern.

### Curve reconstruction

A template edge is a control polyline, so transforming it moves its control points and
pins the two endpoints to the landmark-resolved boundary vertices. Nothing is resampled,
which is why an unchanged target reproduces the source edge to the bit. An edge carried as
a single cubic is refitted against its own parameterisation instead, so a notch at 0.25
still sits a quarter of the way along the same stretch of seam. Loops, spikes and runaway
curvature are rejected, never smoothed over.

### Seam balancing

Balancing works on the length a seam actually covers, walked along the panel boundary from
one stitch endpoint to the other — a straight line between two landmarks says little about
a seam split across several edges. That needs the solved geometry, so the pipeline solves,
balances and solves again until the seams settle (`seam_balance.max_iterations`), and
reports when they do not.

What is corrected is the gap the transformation opened, **not** the gap the block arrived
with. Block C's source seams already differ by up to 4.7 mm; that difference is inherited,
recorded against the block, and the transformation is held to not widening it.

### Topology contract

Preserved: panel count, boundary-loop count, boundary-edge order, semantic edge roles,
identifiers, stitch relationships, dart topology, grain direction, panel orientation and
mirror symmetry. Coordinates and control points change. Each is a validation check.

### Transformation envelope

`transformation.yaml → envelope` declares the tested range per measurement. Beyond it the
result is an extrapolation and needs an explicit operator confirmation, recorded in the
manifest. More than `stop_factor` past the range is refused however it is confirmed. No
reliable extrapolation is claimed beyond the tested range.

## Workflow

Upload YAML → **Field definitions** → Review measurements → **Base block & differences** →
Ease & limits → Transform → **Compare base & target** → Review validation → Export.

The **Field definitions** screen is the gate; there is no override that hides an
ambiguity. The **Compare** screen renders both blocks from one server-side SVG in one
coordinate frame, with overlay / base-only / transformed-only / slider modes, displacement
vectors, a deformation heat map, achieved-against-requested dimensions, and seam mismatch
before and after balancing.

## Configuration

| File | Holds |
|---|---|
| `backend/app/config/measurement-semantics.yaml` | what every field means, candidates, origins, ranges, cross-field blocking rules |
| `backend/app/config/measurements.yaml` | the camera system's own field dictionary |
| `backend/app/config/transformation.yaml` | objective weights, solver limits, curve refit limits, seam balance, envelope, influence map |
| `backend/app/config/drafting.yaml` | geometry tolerances, ease presets, seam allowances, paper sizes, fixture defaults |
| `assets/base-blocks/catalog.yaml` | the blocks on offer, their pinned hashes, the declared reference body |

## Fixture-compatible defaults

`sample/000c79de6519526691ff7b988eb6b92a.json` is the schema fixture. Keys whose meaning
cannot be derived from the fixture alone are preserved verbatim and listed in
`drafting.yaml → fixture_defaults` (echoed into `generation-manifest.json`):

| Key | Value | Why |
|---|---|---|
| `panel.particleDistance` | `6` | constant in every fixture panel |
| `panel.collisionLayer` | `0` | constant in every fixture panel |
| `seqEdge.type` | `3` | the closed outer boundary of a panel |
| `seqEdge.circleType` | `0` | constant for a closed boundary |
| `edge.bezierPoints` | `[[0,0,0],[0,0,0]]` | constant in every fixture edge |

Representations confirmed by inspecting the library rather than guessed:

* `edge.controlPoints` is the **control polyline** of the edge — two endpoints for a
  straight edge, 3 to 98 points for a curve.
* `panel.center` is the **workspace placement** of a panel whose local geometry is centred
  on its own bounding box; world position = local + `center`.
* `seqEdge.type` `3` is a closed boundary contour and `4` an internal contour — a bust,
  waist or chest reference line, usually a single two-point line. Requiring three vertices
  everywhere, as the first schema draft did, rejects 28 of every 60 files in the library.
* A stitch record runs from a point on an edge to a point on an edge, and **those may be
  different edges**, so a seam can cross a boundary corner. Collapsing them into one edge
  loses that.

## Generation manifest

`generation-manifest.json` records the exact base block (id, version, source file, sha256,
review status, declared reference body, measurement definitions), the transformation
configuration and influence-map version, every measurement difference, the envelope and
its confirmation, the measurement schema and definition versions, per-field
interpretations, full field records, assumptions, warnings, settings, tolerances and the
validation summary.

`transformation-diagnostics.json` records the solver per panel (method, landmark count,
convergence, residuals, condition number, merged landmarks), landmark targets with the
operation that moved each one, seam mismatch before and after balancing with the rounds it
took, curve reconstruction per edge, and achieved against requested dimensions.

## API

```
POST /api/measurements/validate            { filename, content }        -> rows (syntax only)
GET  /api/measurements/semantics/schema                                 -> registry + statuses
POST /api/measurements/semantics           { yamlText, confirmations, corrections }
POST /api/measurements/report                                           -> measurement-report.json
GET  /api/base-blocks                                                   -> the catalog on offer
GET  /api/base-blocks/{id}                 id or template:<file>.json   -> block, derived maps, preview
POST /api/base-blocks/{id}/validate                                     -> geometry validation report
POST /api/base-block/deltas                                             -> differences + envelope
POST /api/patterns/transform                                            -> pattern + trace + diagnostics + validation + preview
POST /api/patterns/validate                { pattern }                  -> closure issues
POST /api/patterns/render                                               -> comparison SVG scene
POST /api/patterns/export/pdf | dxf | svg                               -> full-scale production files
POST /api/patterns/export/bundle                                        -> every file as a zip
GET  /api/config                                                        -> presets, allowances, paper, tolerances, solver, envelope
```

Errors are structured: `{"error": {"code", "message", "details": [...]}}`.

## Validation

Geometry checks block export: non-finite coordinates, missing edges, fewer than three
boundary vertices, open boundaries, endpoint discontinuity, zero-length edges,
self-intersecting boundaries, duplicate ids, invalid stitch references, stitch parameters
outside `[0,1]`, invalid cutting contours, solver non-convergence, panel inversion,
reordered or renamed semantic boundaries, changed identifiers or stitches, a rotated grain,
a broken centre line, lost mirror symmetry, unpaired dart edges, and a paired seam the
transformation pushed further apart than it found it. Tolerances live in
`drafting.yaml → tolerances`.

## Exports

`pattern.json`, `generation-manifest.json`, `transformation-diagnostics.json`,
`validation-report.json`, `base-vs-target-preview.png`, `base-vs-target-preview.svg`,
`base-block.json` with its landmark, edge, seam, dart and grainline maps, `pattern-a0.pdf`
and `pattern.dxf`.

## Known limits

* **Dartless blocks.** A and B have no darts. C's front darts are stretches of boundary
  sewn shut, which is how this schema expresses them; there is no apex landmark, so
  bust-apex measurements cannot move geometry and are reported as unmapped.
* **No hip edge.** All three blocks end at their hem, so hip measurements cannot move
  geometry and are reported as unmapped.
* **No sleeves**, until the torso transformation passes acceptance.
* **Interior closed contours** are deformed with the panel but are not modelled as darts.
  None of the three blocks has one.
* `offset_polylines` builds concave joins with the same construction as convex ones and
  does not repair self-intersecting offsets; validation reports them instead.
* Ten of the 26 fields ship as `ambiguous` with no authoritative definition, because the
  samples do not determine their convention (`shoulder_w` is 17.1 in one profile and 35.1
  in the other). Each upload therefore starts blocked until an operator confirms. This is
  intended, not a defect — pin a field to `confirmed` in the schema once the camera
  subsystem's convention is known.
* `fixtures/body_measurements_pascual.yaml` loads and displays, but cannot be transformed:
  `bust_line` 64.0 cm, `waist_line` 76.9 cm and `waist_over_bust_line` 64.0 cm fall outside
  the declared schema ranges for the only permitted vertical origin. It is a validation
  fixture, not a body.
* The **declared reference body** of each block is an operator-revisable assumption, not a
  measurement of the template. Change it in `catalog.yaml` when a block is known to fit
  another size; it is recorded in every manifest.
