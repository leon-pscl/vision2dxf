# Coding-Agent Instruction Sheet: Formula-Based Drafting Web App

## Mission

Build a production-ready web application that accepts `body_measurements.yaml`, validates and displays the camera-derived anthropometric measurements, constructs a basic bodice through explicit drafting equations, verifies the resulting geometry, and exports a GarmageSet-like pattern JSON plus preview and production files.

The defining rule is that all principal pattern landmarks must be calculated from documented drafting rules. Do not use a master pattern, learned model, mesh flattening, pattern retrieval, or undocumented visual adjustments.

## Initial Scope

Implement a basic torso garment with:

- Front bodice panel
- Back bodice panel
- Neckline, shoulder, armhole, side seam, waist, and center edges
- Bust and/or waist dart when required by the selected drafting system
- Sewing lines and configurable cutting lines
- Front-to-back seam relationships

Add sleeves only after the front and back bodice pass all acceptance tests.

## Inputs

The primary input is a YAML file with a top-level `body` mapping. Support these fields:

```yaml
body:
  arm_length: 62.6
  arm_pose_angle: 47.0
  armscye_depth: 20.1
  back_width: 39.0
  bum_points: 17.7
  bust: 85.8
  bust_line: 29.4
  bust_points: 14.6
  crotch_hip_diff: 22.7
  head_l: 21.1
  height: 167.0
  hip_back_width: 50.5
  hip_inclination: 7.5
  hips: 100.5
  hips_line: 22.5
  leg_circ: 55.4
  neck_w: 16.5
  shoulder_incl: 45.0
  shoulder_w: 17.1
  underbust: 85.8
  vert_bust_line: 35.9
  waist: 85.6
  waist_back_width: 38.7
  waist_line: 41.1
  waist_over_bust_line: 29.9
  wrist: 16.2
```

Also collect the following generation settings through the interface:

- Garment type
- Fit or ease preset
- Bust, waist, and hip ease values
- Seam allowance by edge role
- Output paper size and orientation
- Unit confirmation

## Input Validation

Implement a typed `BodyMeasurements` model. The parser must:

1. Accept `.yaml` and `.yml` only.
2. Require the `body` mapping.
3. Reject missing required fields.
4. Reject strings, nulls, booleans, NaN, infinity, and invalid negative values.
5. Display all imported values for user confirmation.
6. Warn about suspicious values without silently changing them.
7. Record the original values separately from derived dimensions.
8. Convert confirmed centimeters to millimeters for pattern geometry.
9. Return field-specific error messages.

Do not assume that similarly named vertical measurements share the same origin. Define a measurement dictionary containing the unit, landmarks, direction, and interpretation of every field.

## Required Architecture

Use the following separation of concerns:

```text
frontend/
  components/
  measurement-review/
  pattern-viewer/
  validation-view/
  export-view/
backend/
  api/
  schemas/
  drafting/
    measurements.py
    ease.py
    derived_dimensions.py
    rules.py
    landmarks.py
    curves.py
    panels.py
    stitches.py
  validation/
  serialization/
  rendering/
  export/
fixtures/
tests/
```

Recommended technologies:

- React, TypeScript, and Vite for the frontend
- Python and FastAPI for the backend
- Pydantic and PyYAML for input/schema validation
- NumPy and Shapely for geometry
- SVG for the interactive two-dimensional preview
- ReportLab for PDF
- ezdxf for DXF
- Pytest and Vitest for testing

Keep drafting and validation logic outside UI components.

## Drafting Pipeline

Implement this deterministic sequence:

```text
YAML upload
→ parsing and validation
→ measurement confirmation
→ ease assignment
→ derived drafting dimensions
→ construction lines
→ named landmarks
→ boundary lines and curves
→ panel assembly
→ seam/stitch relationships
→ geometric validation
→ GarmageSet-like JSON
→ preview and production export
```

## Drafting Rule Registry

Place all equations in a versioned rule registry. Every rule must contain:

```python
@dataclass
class DraftingRule:
    id: str
    name: str
    output_name: str
    required_inputs: list[str]
    equation: Callable
    explanation: str
    source_note: str
    valid_range: dict[str, tuple[float, float]]
```

Each generated landmark must remain traceable:

```python
@dataclass
class DraftedPoint:
    name: str
    x_mm: float
    y_mm: float
    rule_id: str
    dependencies: list[str]
    derived_values: dict[str, float]
```

Never hardcode unexplained constants in geometry functions. Store drafting constants and tolerances in configuration files.

## Measurement Mapping

At minimum, connect the following inputs to pattern regions:

| Pattern region | Relevant inputs |
|---|---|
| Bust/chest line | `bust`, `bust_line`, `vert_bust_line` |
| Waist | `waist`, `waist_line` |
| Hip/lower torso | `hips`, `hips_line` |
| Back | `back_width`, `waist_back_width`, `hip_back_width` |
| Shoulder | `shoulder_w`, `shoulder_incl` |
| Neck | `neck_w` |
| Armhole | `armscye_depth`, `back_width`, `shoulder_w` |
| Bust shaping | `bust_points`, `underbust`, `vert_bust_line` |

If a selected drafting system requires a measurement that is absent, do one of the following: request it from the user, derive it through an approved documented formula, or expose the assumption as a configurable value. Do not silently invent it.

## Internal Geometry Model

Use semantic geometry classes independent of the output schema:

```python
Point2D
LineSegment
CubicBezier
Arc
Dart
PatternEdge
PatternPanel
StitchPair
PatternDocument
```

A `PatternEdge` must contain:

- Stable internal ID
- Semantic role
- Start and end points
- Geometry type
- Curve data
- Sewing or construction classification
- Seam allowance
- Optional paired edge reference

Assemble each panel from an ordered boundary, for example:

```text
Front: center_front → neckline → shoulder → armhole
       → side_seam → waist → center_front
```

Use cubic Bézier curves for necklines and armholes. Ensure adjacent edges share exactly the same endpoint object or coordinates within tolerance.

## GarmageSet-Like Output

Treat the attached reference JSON as the schema fixture. Before implementing generation:

1. Inspect every key in the fixture.
2. Create typed models for the observed structure.
3. Validate the original fixture with those models.
4. Render the fixture successfully.
5. Write a serializer from the internal geometry model to that schema.

The primary JSON must retain the fixture's core organization:

```json
{
  "panels": [],
  "stitches": []
}
```

Each generated panel must provide the applicable fixture fields, including:

- `center`
- `collisionLayer`
- `id`
- `label`
- `particleDistance`
- `seqEdges`

Each sequential boundary must provide the applicable:

- `circleType`
- `type`
- `vertices`
- `edges`

Each edge must provide:

- `id`
- `label`
- `controlPoints`
- `bezierPoints`

Use coordinate triples `[x, y, 0]`. Keep panel geometry local and use `center` for workspace placement. Generate UUIDs for panels and edges. Do not add project-specific metadata to the primary file unless compatibility tests show that it is permitted.

For straight edges, serialize two endpoints as `controlPoints`. For cubic curves, serialize the endpoint and control sequence expected by the fixture adapter. Do not guess the meaning of `circleType`, `type`, `particleDistance`, or `bezierPoints`; preserve verified fixture-compatible defaults and document them.

Create stitch records that reference existing `clothPieceId` and `edgeId` values and use valid normalized edge parameters.

## Geometry Validation

Block export when any critical check fails:

- Non-finite coordinate
- Missing boundary edge
- Fewer than three boundary vertices
- Open boundary
- Endpoint discontinuity
- Zero-length edge
- Self-intersecting outer boundary
- Duplicate panel or edge identifier
- Invalid stitch reference
- Stitch parameter outside `[0, 1]`
- Dart outside its owning panel
- Excessive paired-seam mismatch
- Invalid cutting-line offset

Initial configurable tolerances:

```text
Endpoint continuity:       0.1 mm
Panel closure:             0.1 mm
Duplicate point:           0.01 mm
Self-intersections:        0
Seam mismatch:             max(2 mm, 0.5% of seam length)
```

Produce `validation-report.json` with pass/fail status, metrics, warnings, affected IDs, and suggested actions.

## Cutting Lines

Treat generated panel boundaries as sewing lines. Generate cutting lines afterward by applying configurable outward offsets. Handle:

- Miter, bevel, and rounded joins
- Fold edges with no allowance when configured
- Dart and notch preservation
- Offset self-intersections
- Closed output contours

Do not replace sewing lines with cutting lines; retain both representations in the production pipeline.

## Interface Requirements

Create a responsive workflow with these steps:

1. Upload YAML
2. Review measurements
3. Configure garment and ease
4. Generate pattern
5. Inspect pattern
6. Review validation
7. Export files

The pattern viewer must support zoom, pan, fit-to-view, measurement rulers, and toggles for:

- Construction lines
- Named landmarks
- Bézier control points
- Sewing lines
- Cutting lines
- Darts
- Notches
- Edge labels

Use a functional technical-tool aesthetic, keyboard-accessible controls, visible focus states, 44 px minimum touch targets, and both light and dark themes.

## API Contract

Implement at least:

```text
POST /api/measurements/validate
POST /api/patterns/generate
POST /api/patterns/validate
POST /api/patterns/render
POST /api/patterns/export/pdf
POST /api/patterns/export/dxf
```

`/generate` must return the pattern JSON, generation trace, and validation summary. Use structured errors rather than raw stack traces.

## Exports

Provide downloads for:

- `pattern.json` — fixture-compatible primary pattern
- `generation-manifest.json` — method, settings, assumptions, and rule traces
- `validation-report.json`
- `pattern-preview.png`
- `pattern-a0.pdf`
- `pattern.dxf`

Ensure PDF and DXF preserve full scale.

## Required Tests

Write tests for:

- Valid and invalid YAML parsing
- Unit conversion
- Every drafting equation
- Rule dependency tracing
- Landmark coordinates
- Curve endpoint continuity
- Panel closure
- Dart containment
- Ease application
- Front/back seam compatibility
- Cutting-line offset
- Reference JSON schema validation
- Output JSON round trip
- Deterministic generation
- PDF and DXF scale

Maintain golden files for at least two supplied measurement profiles. Golden comparisons must use numeric tolerances rather than screenshots alone.

## Acceptance Criteria

The implementation is accepted only when:

- Both supplied YAML fixtures can be loaded and reviewed.
- A valid profile produces front and back bodice panels.
- Every principal landmark identifies its drafting rule and dependencies.
- No master block is loaded or transformed.
- Critical geometry checks pass.
- JSON follows the verified GarmageSet-like schema.
- The preview corresponds to the exported geometry.
- Repeated runs with identical input and settings are geometrically identical.
- Full-scale PDF and DXF exports pass a dimensional test.
