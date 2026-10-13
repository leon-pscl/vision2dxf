# Coding-Agent Instruction Sheet: Anthropometric Surface Reconstruction and Mesh Flattening Web App

## Mission

Build a production-ready web application that accepts `body_measurements.yaml`, constructs a measurement-constrained proxy torso, creates an ease-adjusted garment surface, segments that surface using a predetermined seam network, flattens the resulting patches into two-dimensional pattern panels, verifies geometric quality, and exports a GarmageSet-like pattern JSON plus preview and production files.

The defining rule is that the two-dimensional pattern must result from computational surface reconstruction, segmentation, triangulation, and flattening. Do not use a two-dimensional master block, sequential drafting formulas, pattern retrieval, a transformer, or another learned pattern generator.

## Scientific Scope

The input contains scalar anthropometric measurements rather than a complete three-dimensional scan. Therefore:

- Call the reconstructed body a `proxy anthropometric surface`.
- Do not claim exact body-shape reconstruction.
- Record all inferred depths, shapes, and landmark assumptions.
- Keep the panel topology and seam network predetermined in the first version.
- Restrict the first implementation to a basic torso garment.
- Treat uncertain or inconsistent anatomical levels as user-review items.

## Inputs

Accept `body_measurements.yaml` with a top-level `body` object and support:

```text
arm_length, arm_pose_angle, armscye_depth, back_width,
bum_points, bust, bust_line, bust_points, crotch_hip_diff,
head_l, height, hip_back_width, hip_inclination, hips,
hips_line, leg_circ, neck_w, shoulder_incl, shoulder_w,
underbust, vert_bust_line, waist, waist_back_width,
waist_line, waist_over_bust_line, wrist
```

Collect configuration for:

- Garment type
- Cross-section model: ellipse or superellipse
- Cross-section resolution
- Longitudinal mesh resolution
- Bust, waist, hip, and armhole ease
- Seam-network preset
- Flattening algorithm
- Distortion and seam weights
- Seam allowances
- Paper output settings
- Unit confirmation

## Input Validation

Implement a typed measurement model. The parser must:

1. Accept `.yaml` and `.yml`.
2. Require the `body` mapping.
3. Reject absent, non-numeric, non-finite, or invalid negative values.
4. Display all measurements for confirmation.
5. Show the assumed unit and anatomical reference definitions.
6. Preserve original values separately from all inferred dimensions.
7. Warn when vertical levels have ambiguous origins or implausible ordering.
8. Convert confirmed centimeters to millimeters internally.
9. Never silently correct camera-derived measurements.

## Required Architecture

```text
frontend/
  components/
  measurement-review/
  cross-section-view/
  mesh-viewer/
  flattening-view/
  pattern-viewer/
  validation-view/
  export-view/
backend/
  api/
  schemas/
  reconstruction/
    anatomical_levels.py
    cross_sections.py
    perimeter_solver.py
    landmarks.py
    lofting.py
    smoothing.py
  garment_surface/
    ease.py
    offset.py
    seams.py
    segmentation.py
  flattening/
    triangulation.py
    parameterization.py
    distortion.py
    seam_balance.py
    boundary_refit.py
  validation/
  serialization/
  rendering/
  export/
fixtures/
tests/
```

Recommended technologies:

- React, TypeScript, Vite
- Three.js for the 3D viewer
- SVG for flattened pattern inspection
- Python, FastAPI
- Pydantic, PyYAML
- NumPy, SciPy, Shapely
- A maintained mesh library such as trimesh or libigl bindings where deployment permits
- ReportLab and ezdxf
- Pytest and Vitest

Keep mesh computation in backend services or dedicated workers rather than UI components.

## Generation Pipeline

```text
YAML upload
→ validation and measurement confirmation
→ anatomical-level definition
→ cross-section reconstruction
→ proxy torso lofting
→ constraint-preserving smoothing
→ ease-adjusted garment surface
→ predetermined seam placement
→ panel segmentation
→ constrained triangulation
→ distortion-minimizing flattening
→ boundary extraction and curve fitting
→ seam balancing
→ geometric validation
→ GarmageSet-like JSON
→ preview and production export
```

## Assumption Registry

Every inferred value must be recorded:

```python
@dataclass
class SurfaceAssumption:
    id: str
    name: str
    value: float | str
    unit: str | None
    reason: str
    source_fields: list[str]
    configurable: bool
```

Examples include inferred torso depth, superellipse exponent, front/back circumference distribution, smoothing strength, shoulder-section construction, and surface-offset interpretation.

Write assumptions to `generation-manifest.json`, not to the compatibility pattern JSON.

## Anatomical Cross-Sections

Construct cross-sections at the available levels, including:

- Neck
- Upper chest or shoulder
- Bust
- Underbust
- Waist
- Hip

Represent a section as an ellipse or superellipse:

```text
abs(x/a)^n + abs(z/b)^n = 1
```

Requirements:

- The numerical perimeter must match the target circumference within tolerance.
- Use measured width information when its anatomical definition matches the section.
- Solve for missing depth numerically when circumference and width are available.
- Reject impossible width/circumference combinations.
- Allow controlled front/back asymmetry only when supported by an input or explicit assumption.
- Record residual error for every solved section.

Do not use an approximate circumference formula without reporting its residual against a numerical perimeter calculation.

## Anatomical Levels

Create a normalized vertical coordinate system after user confirmation. Store:

```python
@dataclass
class AnatomicalLevel:
    name: str
    y_mm: float
    source_field: str | None
    origin_definition: str
    confidence: float | None
```

Validate ordering and minimum separation. If `bust_line`, `waist_line`, or `hips_line` appear inconsistent, stop and request confirmation rather than rearranging them automatically.

## Proxy Surface Lofting

For each section, create a consistently sampled ring:

```python
ring_i = np.ndarray[(ring_vertex_count, 3)]
```

Requirements:

- Adjacent rings use the same angular correspondence.
- Connect rings with consistently oriented triangles.
- Avoid duplicate poles and zero-area faces.
- Produce a watertight or explicitly open manifold appropriate to the garment scope.
- Preserve target circumferences during smoothing.
- Validate face normals and manifold edges.
- Retain anatomical section and landmark tags.

Store the intermediate proxy mesh for debugging.

## Garment Surface and Ease

Construct the garment surface from the proxy body using level-dependent ease. Support separate bust, waist, hip, and armhole values.

Requirements:

- Interpolate ease smoothly between levels.
- Avoid applying one uniform offset to every vertex.
- Preserve garment symmetry when configured.
- Avoid self-intersection and local surface inversion.
- Distinguish circumference ease from normal-distance offset.
- Calculate and display achieved garment circumferences.

Document the mathematical conversion between circumference ease and surface displacement.

## Predetermined Seam Network

For the first version, define these seam/boundary paths:

- Center front
- Center back
- Left and right side seam
- Neckline
- Left and right armhole
- Shoulder seams
- Waist or hem boundary

The seam network must follow mesh vertices or constrained paths so each seam becomes a panel boundary. Do not implement unconstrained automatic seam discovery in the first release.

Create stable semantic labels for all seam paths so they can become output edge labels and stitch references.

## Panel Segmentation

Represent each surface patch as:

```python
@dataclass
class SurfacePanel:
    id: str
    label: str
    vertices_3d: np.ndarray
    faces: np.ndarray
    boundary_loops: list[list[int]]
    semantic_boundaries: dict[str, list[int]]
```

Each panel intended for one-piece flattening must be topologically disk-like unless the chosen parameterization explicitly supports holes. Reject disconnected and non-manifold patches.

## Flattening Algorithm

Implement one documented parameterization method, preferably:

- Least-Squares Conformal Mapping for initialization followed by ARAP refinement, or
- Another validated constrained flattening method

Remove rigid-motion ambiguity by fixing at least two non-collinear vertices or applying equivalent constraints.

Optimize an objective equivalent to:

```text
E = w_distortion * E_distortion
  + w_boundary * E_boundary
  + w_seam * E_seam
  + w_grain * E_grain
  + w_flip * E_flip
```

Requirements:

- Minimize angular and length distortion.
- Keep semantic seam endpoints stable.
- Align the selected grain direction.
- Penalize flipped triangles strongly.
- Report convergence status and residuals.
- Reject any result containing flipped triangles.
- Keep all weights in a versioned configuration file.

## Flattening Diagnostics

Calculate and export:

- Mean and maximum angular distortion
- Mean and maximum area distortion
- Mean and maximum edge-length distortion
- Number of flipped triangles
- Panel-boundary self-intersections
- Paired-seam length differences
- Cross-section reconstruction residuals
- Garment-ease residuals
- Solver iterations and convergence status

Present a distortion heat map in the web interface. Store numerical results in `flattening-diagnostics.json` and `validation-report.json`.

## Boundary Reconstruction

Do not serialize every dense mesh-boundary vertex as a garment corner. Convert flattened boundaries into practical pattern edges:

1. Extract ordered boundary loops.
2. Split loops using semantic seam intervals.
3. Simplify within a configurable geometric tolerance.
4. Detect nearly straight intervals.
5. Fit lines to straight intervals.
6. Fit cubic Bézier curves to curved intervals.
7. Preserve semantic endpoints exactly.
8. Measure curve-fit and seam-length error.
9. Reject a fit that exceeds tolerance.
10. Generate stitches between compatible semantic edges.

Retain the dense flattened mesh separately for diagnostics.

## GarmageSet-Like Output

Treat the attached reference JSON as the authoritative schema fixture. Before implementing the serializer:

1. Enumerate every observed key and nesting level.
2. Create typed schema models.
3. Validate the original fixture.
4. Build a renderer for the fixture.
5. Map reconstructed panel boundaries into the verified schema.

The primary output must retain the core structure:

```json
{
  "panels": [],
  "stitches": []
}
```

Support observed panel fields including `center`, `collisionLayer`, `id`, `label`, `particleDistance`, and `seqEdges`; sequential boundary fields including `circleType`, `type`, `vertices`, and `edges`; and edge fields including `id`, `label`, `controlPoints`, and `bezierPoints`.

Output two-dimensional panel coordinates as `[x, y, 0]` in millimeters. Use local coordinates with `center` for global workspace placement. Generate UUIDs and ensure each stitch refers to existing `clothPieceId` and `edgeId` values with valid normalized parameters.

Do not put mesh diagnostics or reconstruction assumptions into the primary compatibility JSON unless the fixture explicitly supports them.

## Geometry and Mesh Validation

Stop generation for any of the following:

- Unsolvable cross-section
- Cross-section residual above tolerance
- Invalid anatomical-level ordering
- Non-finite mesh vertex
- Zero-area or inverted 3D face
- Non-manifold panel mesh
- Disconnected panel
- Flattening non-convergence
- Flipped 2D triangle
- Self-intersecting flattened boundary
- Open reconstructed pattern boundary
- Excessive curve-fit error
- Excessive paired-seam mismatch
- Invalid stitch reference
- Invalid cutting-line offset

Suggested initial configurable limits:

```text
Panel endpoint continuity:       0.1 mm
Pattern closure:                 0.1 mm
Duplicate point:                 0.01 mm
Flattened triangle flips:        0
Boundary self-intersections:     0
Seam mismatch:                   max(2 mm, 0.5% of seam length)
Cross-section residual:          documented per prototype validation
Curve-fit maximum deviation:     documented per prototype validation
```

Do not assign unsupported residual thresholds without experimental calibration; expose them as configuration and report the selected values.

## Interface Requirements

Create these stages or tabs:

1. Measurements
2. Cross-sections
3. Proxy surface
4. Garment surface
5. Seam network
6. Flattening
7. Two-dimensional pattern
8. Validation
9. Export

Required visualizations:

- Measurement table
- Cross-section overlays with target perimeter
- Rotatable 3D proxy torso
- Ease-adjusted garment surface
- Seam paths and segmented panels
- Triangular mesh overlay
- Flattened panels
- Distortion heat map
- Sewing and cutting lines

Provide accessible keyboard controls, visible focus states, responsive layouts, 44 px minimum touch targets, and light/dark themes. Include a reduced-motion mode.

## API Contract

Implement at least:

```text
POST /api/measurements/validate
POST /api/reconstruction/cross-sections
POST /api/reconstruction/proxy-surface
POST /api/garment/surface
POST /api/garment/segment
POST /api/patterns/flatten
POST /api/patterns/validate
POST /api/patterns/render
POST /api/patterns/export/pdf
POST /api/patterns/export/dxf
```

Long-running geometry jobs should return structured progress and cancellation support. Never expose raw stack traces.

## Debugging Outputs

Allow optional downloads of:

- `cross-sections.json`
- `proxy-mesh.obj`
- `garment-mesh.obj`
- `segmented-panels.obj`
- `flattened-mesh.json`
- `flattening-diagnostics.json`

These do not replace the required compatibility outputs.

## Production Exports

Provide:

- `pattern.json`
- `generation-manifest.json`
- `validation-report.json`
- `flattening-diagnostics.json`
- `pattern-preview.png`
- `pattern-a0.pdf`
- `pattern.dxf`

Generate cutting lines after sewing-line validation. Pattern placement must use cutting-line polygons rather than sewing-line boundaries.

## Required Tests

Write tests for:

- YAML parsing and units
- Anatomical-level validation
- Ellipse and superellipse perimeter solving
- Impossible section rejection
- Ring correspondence
- Mesh manifoldness
- Face orientation
- Constraint-preserving smoothing
- Ease interpolation
- Seam-network continuity
- Panel segmentation
- Flattening convergence
- Triangle-flip detection
- Distortion metrics
- Boundary extraction
- Curve simplification and Bézier fitting
- Seam balancing
- Reference JSON validation
- Deterministic output
- PDF and DXF scale

Include controlled geometry fixtures:

- Plane: near-zero distortion expected
- Cylinder patch: rectangular development expected
- Truncated cone patch: sector-like development expected
- Symmetric torso proxy: left/right consistency expected

Maintain golden outputs for the supplied body-measurement profiles, but compare numeric geometry and diagnostic tolerances rather than screenshots alone.

## Acceptance Criteria

The implementation is accepted only when:

- Both supplied YAML fixtures load and display correctly.
- The app builds and displays a proxy torso from a valid profile.
- Reconstructed cross-sections report their target and achieved perimeters.
- The ease-adjusted surface reports achieved circumferences.
- The fixed seam network produces valid front and back patches.
- Every panel is flattened without triangle inversion.
- Flattened boundaries are closed and non-self-intersecting.
- Boundary curves remain within the configured fitting tolerance.
- Paired seams pass the common compatibility test.
- The common GarmageSet-like schema validator accepts the output.
- The app exposes assumptions and distortion diagnostics.
- No two-dimensional master block or formula-drafting sequence is used.
