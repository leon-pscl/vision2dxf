# Coding Agent Brief: Image-Based Body Measurement and Pattern-Ready Export

## How to use this brief

Place this file at the repository root and give it to the coding agent as the governing specification.
Build **one milestone at a time**. After every milestone:

1. Run all automated checks.
2. Save the required visual/debug artifacts.
3. Report completed acceptance criteria, failed checks, assumptions, and measured runtime.
4. Stop and wait for approval before starting the next milestone.

Do not silently change a landmark definition, formula, model, threshold, output key, or dependency. If a requirement cannot be implemented as written, state the conflict and propose a testable alternative before coding it.

---

## 1. Product objective

Build a local web application that acquires body measurements from calibrated images for a made-to-measure pattern-generation workflow.

The app must:

1. Capture or upload raw front, left-side, right-side, back, and optional multi-angle images.
2. Detect the person with YOLO26 segmentation and obtain pose landmarks with YOLO26 Pose.
3. preserve and display the image at every meaningful preprocessing and measurement stage.
4. Compute Montazerian-style direct measurements from paired front/side silhouettes.
5. Compute additional pattern-driving measurements using documented landmark, contour, planar-section, polyline, and optional multi-view mesh methods.
6. Allow an operator to inspect and correct landmarks before accepting results.
7. Display final measurements, method, source views, confidence, warnings, and edit history.
8. Export a pattern-ready body-measurement file and a detailed provenance report.

This is a research prototype. It may demonstrate acquisition, traceability, repeatability, and pattern-engine compatibility. Until comparison measurements are collected under a standardized protocol, it must **not** claim human-measurement accuracy or equivalence to manual anthropometry.

---

## 2. Evidence-to-implementation map

Implement only methods traceable to the following recent studies and primary project documentation:

- **Montazerian (2024)**: portable RGB capture; image correction/enhancement; body isolation; calibration; feature extraction; semi-automatic marker placement; paired frontal breadth and side depth; ellipse-based torso circumference; shoulder and sleeve-related upper-body measurements.
- **Li et al. (2023), “Non-contact clothing anthropometry based on two-dimensional image contour detection and feature point recognition”**: improved contour extraction, morphology, learned/anatomical landmark support, front/side length measurements, and girth estimation.
- **Zong et al. (2023), “Predicting Human Body Measurements Using MediaPipe Pose Auto-Capture”**: guided multi-pose capture; pose-based capture gates; user-height scaling; shoulder, chest, waist, and hip widths, depths, and levels from outlines.
- **Jin et al. (2023), “Pants design and pattern generation based on body images”**: front/side silhouette processing; side-profile landmarks at waist, abdomen, hip, crotch, thigh, knee, and ankle; connection between extracted body dimensions and pants-pattern generation.
- **Korosteleva et al. (2024), GarmentCodeData**: pattern-driving measurements from body geometry using landmarks, optimized polylines/geodesics, and planar mesh intersections; body measurements are separate from garment design parameters.
- **GarmentCode official repository and body-measurement specification**: exact output names and compatibility contract for the pinned revision.

Implementation consequences:

- Use Montazerian's two-view method for the core upper-body outputs.
- Use pose keypoints only as initialization and capture-quality evidence, not as apparel landmarks when definitions differ.
- Use silhouettes and operator-adjustable anatomical rows for 2D measurements.
- Use an optional multi-view reconstructed body mesh for surface-path or section measurements that a two-view silhouette cannot defensibly provide.
- Never manufacture a missing value from a population ratio without labeling it `estimated` and requiring explicit operator acceptance.
- Never describe a mesh-derived value as directly measured from a photograph.

---

## 3. Scope and output hierarchy

### 3.1 Core direct outputs

The first result panel must show the measurements acquired in Montazerian's workflow:

- Shoulder length, left and right, plus accepted mean.
- Sleeve/arm length, left and right, plus accepted mean.
- Chest circumference.
- Bust circumference.
- Waist circumference.
- Hip circumference.

The four circumferences must retain their intermediate values:

- Front breadth in millimetres.
- Side depth in millimetres.
- Front and side source rows.
- Ellipse formula used.
- Scale for every contributing view.

### 3.2 Extended pattern-ready outputs

The second result panel must show the body inputs required by the pinned GarmentCode revision. At minimum, support the following 26 primary fields, subject to compatibility verification against that revision:

- `height`
- `head_l`
- `neck_w`
- `shoulder_w`
- `shoulder_incl`
- `armscye_depth`
- `arm_length`
- `arm_pose_angle`
- `bust`
- `underbust`
- `waist`
- `hips`
- `leg_circ`
- `wrist`
- `back_width`
- `waist_back_width`
- `hip_back_width`
- `waist_line`
- `waist_over_bust_line`
- `bust_line`
- `vert_bust_line`
- `hips_line`
- `crotch_hip_diff`
- `hip_inclination`
- `bust_points`
- `bum_points`

Before implementation, pin the GarmentCode commit in `config/app.yaml`. Add a test that loads the exported file with that pinned code. If the schema differs, stop and report the exact difference instead of renaming keys silently.

### 3.3 Additional research measurements

Store the following when the active garment profile needs them, even if they are not primary GarmentCode keys:

- Neck-base circumference.
- Across-front width.
- Across-back width.
- Front waist length.
- Back waist length.
- Side waist length.
- Waist-to-hip length.
- Bust-point height and separation.
- Upper-arm circumference.
- Elbow circumference.
- Shoulder-to-elbow length.
- Elbow-to-wrist length.
- Crotch depth.
- Total, front, and back crotch lengths.
- Inseam and outseam.
- Thigh, knee, calf, and ankle circumferences.
- Knee and ankle heights.
- Front/back waist and hip arcs.

These fields belong in `extended_measurements.yaml`. Do not insert unsupported keys into `body_measurements.yaml`.

### 3.4 Body versus design parameters

Body measurements describe the person. Ease, seam allowance, hem allowance, waistband height, garment length, pocket dimensions, sleeve style, collar style, and fit preference are garment-design inputs and must be stored separately in `design_parameters.yaml`.

---

## 4. Required acquisition modes

### 4.1 Two-view core mode

Required for the Montazerian outputs:

- One accepted front image.
- At least one accepted side image.
- A second side image is recommended and averaged only after side-orientation checks pass.
- Measured stature or a known-dimension calibration target for metric scale.

### 4.2 Four-view extended mode

Required for front/back distribution and better landmark review:

- Front.
- Left side.
- Back.
- Right side.

### 4.3 Multi-view mesh mode

Required for measurements whose defensible implementation uses a body surface or planar mesh section:

- 12 guided views at nominal 30-degree intervals, or a guided slow-turn video sampled into equivalent views.
- Constant camera position and zoom.
- Constant subject posture.
- A-pose with visible arms and legs.
- Known stature or a visible calibration target.
- Capture-angle metadata for every frame.

The system must show which mode produced each value. Two-view mode must never pretend to provide a validated 3D geodesic.

### 4.4 Scale is not validation ground truth

A measured stature or known calibration target may be used solely to convert pixels to physical units. This is a calibration input, not evidence that the remaining body measurements are accurate.

Supported scale modes:

- `stature`: operator enters stature measured without shoes.
- `reference_target`: a known-length target is visible in the same approximate body plane.

Do not infer metric scale from camera metadata, assumed camera distance, demographic averages, or image resolution.

---

## 5. Hard constraints

- Python 3.11.
- CPU-only target: Core i5-class PC with 16 GB RAM.
- Fully local operation; no cloud APIs, remote inference, analytics, or telemetry.
- Backend: FastAPI and Uvicorn.
- Frontend: semantic HTML, CSS, and vanilla JavaScript served by FastAPI; no frontend build step.
- Canvas is permitted for image overlays and draggable landmarks.
- Core libraries: `ultralytics`, `opencv-python`, `numpy`, `pyyaml`, `pydantic`, `pytest`, and `ruff`.
- Optional mesh milestone may add `trimesh`, `scipy`, and `scikit-image`; document why each is needed before installation.
- GarmentCode/`pygarment` is used only at the pinned revision for compatibility and pattern-load testing.
- Model paths and SHA-256 hashes are configured; production startup must not download weights.
- Internal linear unit: millimetres as `float`.
- Display/export linear unit: centimetres rounded to one decimal; retain unrounded internal values in provenance.
- Angles are in degrees.
- Type hints on all functions and public-method docstrings.
- `ruff check` and `pytest` must pass at each milestone.
- Never use demographic averages as unlabeled measurements.
- Never replace a failed calculation with zero.
- An unavailable result is `null` with a reason.

---

## 6. Repository layout

```text
bodymeasure/
├── AGENT_BRIEF.md
├── README.md
├── pyproject.toml
├── config/
│   ├── app.yaml
│   ├── calibration.yaml
│   ├── garmentcode.yaml
│   └── measurement_schema.yaml
├── models/                         # provisioned separately; git-ignored
├── src/bodymeasure/
│   ├── api.py
│   ├── config.py
│   ├── store.py
│   ├── session.py
│   ├── preprocess.py
│   ├── vision.py
│   ├── pose_gate.py
│   ├── mask_clean.py
│   ├── landmarks.py
│   ├── geometry.py
│   ├── measure_2d.py
│   ├── reconstruct.py              # optional multi-view milestone
│   ├── measure_mesh.py             # optional multi-view milestone
│   ├── provenance.py
│   ├── artifacts.py
│   └── export.py
├── web/
│   ├── index.html
│   ├── review.html
│   ├── sessions.html
│   ├── app.js
│   └── style.css
├── tools/
│   ├── calibrate_lens.py
│   ├── provision_models.py
│   └── verify_garmentcode.py
├── tests/
│   ├── synthetic/
│   ├── fixtures/
│   └── test_*.py
├── scripts/
│   └── check.sh
└── data/                            # git-ignored
    ├── app.db
    └── sessions/
```

---

## 7. Session data contract

Each session uses a UUID4 hex folder; never use a person's name in a path.

```text
data/sessions/<session_id>/
├── subject.yaml
├── views/
│   ├── front/
│   │   ├── 00_raw.jpg
│   │   ├── 01_oriented.png
│   │   ├── 02_undistorted.png
│   │   ├── 03_enhanced.png
│   │   ├── 04_detection.png
│   │   ├── 05_roi.png
│   │   ├── 06_mask_raw.png
│   │   ├── 07_mask_clean.png
│   │   ├── 08_body_isolated.png
│   │   ├── 09_edges.png
│   │   ├── 10_keypoints.png
│   │   ├── 11_landmarks.png
│   │   └── 12_measurement_overlay.png
│   └── <other views use the same stage names>/
├── predictions/
│   ├── <view>_detections.json
│   └── <view>_keypoints.json
├── landmarks.json
├── mesh/                            # optional
│   ├── visual_hull.obj
│   └── mesh_measurements.json
├── body_measurements.yaml
├── extended_measurements.yaml
├── measurement_report.yaml
└── manifest.json
```

Every generated image must be listed in `manifest.json` with:

```yaml
stage_id: 07_mask_clean
view: front
path: views/front/07_mask_clean.png
created_at: ISO-8601
algorithm: clean_person_mask
parameters:
  kernel_px: 5
  max_hole_area_px: 500
source_files:
  - views/front/06_mask_raw.png
```

Write files atomically: write a temporary file in the same directory, flush it, then use `os.replace`.

---

## 8. Mandatory visual pipeline

The web app must let the operator inspect the following stages for each view:

1. **Raw capture** — exact camera/upload input; never overwritten.
2. **Orientation correction** — rotation and side-view mirroring status visible.
3. **Lens correction** — undistorted image, or a clear `not applied` badge.
4. **Image enhancement** — the exact image sent to downstream vision; show parameters.
5. **Human detection** — bounding box, class, confidence, and selected instance.
6. **Extracted ROI** — cropped person region with coordinates relative to the raw image.
7. **Raw segmentation mask** — model output before cleanup.
8. **Cleaned mask** — opening/closing, connected-component filtering, and hole treatment.
9. **Separated body** — foreground body on a neutral/transparent background.
10. **Edge/contour image** — final contour used by geometric functions.
11. **Pose keypoints** — names and confidence values.
12. **Landmark overlay** — anatomical points and rows used for measurement.
13. **Measurement overlay** — breadths, depths, paths, arcs, and numeric labels.
14. **Final measurements** — values, units, provenance, confidence, warning status, and edit status.

### 8.1 UI behavior

- Use a stage timeline or filmstrip plus one large comparison viewer.
- Provide `Previous`, `Next`, view selector, stage selector, zoom, pan, fit-to-screen, and raw-versus-current comparison.
- Provide side-by-side and opacity-slider comparison modes.
- Never stretch an image; preserve the aspect ratio and map overlays through a single coordinate-transform utility.
- Show loading skeletons while processing and a specific error at the failed stage.
- A failed stage keeps earlier successful artifacts available.
- Each stage displays runtime, image dimensions, and relevant parameters in an expandable panel.
- Do not use stage colors as the only means of communicating status.

---

## 9. Image-processing specification

### 9.1 Preprocessing

For each raw view:

1. Validate MIME type, decode safely, strip nonessential metadata, and enforce configurable size limits.
2. Apply configured rotation.
3. Mirror side views into the canonical direction only in a derived artifact; retain raw orientation metadata.
4. Apply `cv2.undistort` when a valid camera matrix and distortion coefficients exist.
5. Apply restrained enhancement: configurable white-balance/contrast normalization and edge-preserving denoising. Do not alter body geometry.
6. Save every stage before proceeding.

Lens calibration must use a printed checkerboard with a known square size and multiple captures spanning the image area and orientations. Reject calibration with insufficient valid frames or unacceptable reprojection error. Calibration estimates lens parameters only; it does not create body scale.

### 9.2 Person segmentation

- Load YOLO26 segmentation once at application startup.
- Use the configured model, image size, device, and confidence threshold.
- Keep class `person` only.
- Default selection: largest accepted person instance.
- Reject ambiguous frames with multiple similarly sized people rather than guessing.
- Save bounding boxes, confidence, mask polygon, raw raster mask, and model metadata.
- Use full-resolution masks when supported.

### 9.3 Mask cleanup

All parameters must be resolution-aware. Store both configured normalized values and resolved pixel values.

1. Morphological opening.
2. Morphological closing.
3. Largest connected component.
4. Fill only holes below the configured normalized area threshold.
5. Preserve gaps between arms/torso and legs.
6. Smooth the contour without changing width beyond a configurable pixel tolerance verified by tests.

### 9.4 Pose gate

Use YOLO26 Pose keypoints for capture validation and initialization.

Front-view checks:

- Required head/ear, shoulder, elbow, wrist, hip, knee, and ankle confidence.
- A-pose arm separation.
- Wrists do not overlap the torso mask.
- Both feet and the top of the head are visible.
- Body is sufficiently centered and occupies the configured height range.

Side-view checks:

- Visible-side shoulder, hip, knee, and ankle confidence.
- Shoulder-pair projected width below the configured fraction of the accepted front view.
- Head, torso, legs, and feet not clipped.
- Side-facing direction can be determined.

Back-view checks:

- Symmetric shoulders and hips within configurable tolerances.
- Arms separated from torso.
- Full body visible.

Return structured failures such as:

```json
{
  "ok": false,
  "reasons": [
    {"code": "WRIST_OVERLAP", "message": "Move the left arm farther from the torso."}
  ]
}
```

---

## 10. Calibration and alignment

Each view has its own `mm_per_px` and physical-height mapping.

For stature scaling:

```text
mm_per_px = stature_mm / (floor_row - head_row)
height_above_floor_mm(y) = (floor_row - y) * mm_per_px
```

Requirements:

- `head_row` and `floor_row` are reviewable and editable.
- A scale edit invalidates and recomputes every dependent measurement.
- Front and side rows are matched by physical height above the floor, never by raw pixel row.
- If scale differs excessively between repeated same-orientation captures, warn and require review.
- Hair, headwear, shoes, and floor occlusion must generate warnings.

For reference-target scaling, calculate scale from the detected/confirmed target length in the same approximate body plane. Store target endpoints, physical length, and reprojection/placement warning.

---

## 11. Landmark specification

All automatic landmarks are proposals. Store pixel coordinates, physical height, source method, confidence, and operator edits.

### 11.1 Front landmarks

- Head top and floor contact row.
- Neck-base left/right.
- Shoulder point left/right.
- Armpit left/right and armscye row.
- Chest/bust row.
- Underbust row.
- Natural waist row.
- Maximum hip row.
- Crotch row.
- Bust points when visually/algorithmically supported; otherwise operator-required.
- Elbow and wrist center points.
- Left/right leg centerlines.
- Thigh, knee, calf, and ankle rows.

### 11.2 Side landmarks

- Nape.
- High point shoulder.
- Balance/coronal line.
- Front chest/bust prominence.
- Underbust.
- Waist indentation.
- Abdomen prominence.
- Buttock/maximum-hip prominence.
- Crotch level.
- Thigh, knee, calf, and ankle levels.
- Heel/floor point.

### 11.3 Back landmarks

- Back-neck point.
- Shoulder points.
- Across-back level and endpoints.
- Back waist endpoints.
- Back hip endpoints.
- Buttock prominence points when supported; otherwise operator-required.

### 11.4 Editing rules

- Points and horizontal rows are draggable.
- Path-based measurements expose editable control points.
- Every edit records old value, new value, timestamp, optional note, and affected measurements.
- Recompute only the dependency graph downstream of the edited item.
- Provide `Undo`, `Redo`, and `Reset to automatic` during the current review session.

---

## 12. Geometry specification

`geometry.py` contains pure functions and no file I/O.

Required functions:

```python
width_at(mask, y, x_ref, mm_per_px) -> float
split_depth_at(mask, y, balance_x, mm_per_px) -> tuple[float, float]
euclidean_length(points, mm_per_px) -> float
polyline_length(points_mm) -> float
convex_hull_path_length(points_mm) -> float
ellipse_perimeter_montazerian(a, b) -> tuple[float, str]
ramanujan_ii(a, b) -> float
angle_deg(p1, vertex, p2) -> float
planar_section(mesh, plane_origin, plane_normal) -> Section
surface_geodesic(mesh, start, end) -> float
```

### 12.1 Montazerian circumference

For chest, bust, waist, and hip:

1. Obtain front breadth `W` at the accepted anatomical row.
2. Obtain side depth `D` at the same physical height.
3. Set ellipse semi-axes `a = max(W, D) / 2` and `b = min(W, D) / 2`.
4. Apply the same two-equation selection rule documented by Montazerian: use the first selected approximation when `a <= 3b`; use the elongated-ellipse approximation otherwise.
5. Save the chosen formula identifier and inputs.

Do not replace this with a single formula without an explicit approved change request and regression comparison.

### 12.2 Asymmetric front/back arcs

Where a pattern field requires front/back distribution, use the accepted side balance line to split depth into `d_front` and `d_back`. Compute front and back semi-ellipse arcs separately and save both. Mark these values as method-specific approximations until validated.

### 12.3 Lengths and paths

- Shoulder length: neck-base point to shoulder point, separately left/right.
- Arm length: shoulder to elbow to wrist polyline, separately left/right.
- Straight vertical levels: use physical-height differences.
- Surface-following torso paths: use reviewed silhouette paths in 2D mode or mesh geodesics in mesh mode.
- Crotch lengths: require a reviewed surface path or mesh geodesic; never infer total crotch length from crotch depth alone.

### 12.4 Limb circumferences

Use paired front breadth and side depth at the same physical height when both are observable. For each thigh, knee, calf, ankle, upper arm, elbow, and wrist result, save whether the cross-section was modeled as an ellipse, circle, planar mesh intersection, or operator input.

---

## 13. Optional multi-view reconstruction

Implement only after the two-view pipeline passes its acceptance tests.

### 13.1 Purpose

The mesh exists to support measurements that depend on surface paths, cross-section planes, front/back distribution, or three-dimensional landmarks. It is not required for the six Montazerian outputs.

### 13.2 Minimum method

- Use calibrated multi-angle silhouettes and known capture angles.
- Construct a visual hull or equivalent documented mesh approximation.
- Place the mesh in a common floor-centered metric coordinate system.
- Remove disconnected artifacts and preserve an untouched reconstruction artifact.
- Extract planar cross-sections for girths.
- Compute shortest surface paths only between operator-reviewable anatomical endpoints.

### 13.3 Mesh measurement rules

Follow the GarmentCodeData measurement families:

- Landmark-to-landmark distances for straight measurements.
- Surface geodesic or optimized polyline for tape-like path measurements.
- Horizontal planar intersection and closed contour/convex hull for torso and limb girths.

The result record must identify `mesh_visual_hull` or the exact reconstruction method. Do not label visual-hull measurements as scan ground truth.

### 13.4 Failure behavior

If reconstruction is incomplete, non-watertight where a closed section is required, or inconsistent across views:

- Keep all 2D outputs.
- Set affected mesh outputs to `null`.
- Show a recapture or manual-entry action.
- Never interpolate a plausible-looking final value silently.

---

## 14. Measurement provenance contract

Every result uses this structure internally:

```yaml
name: waist
value_mm: 843.27
value_cm_display: 84.3
status: acquired
source: geometric_2d
method: montazerian_ellipse
views: [front, side_left, side_right]
inputs:
  front_breadth_mm: 286.4
  side_depth_mm: 221.8
  row_height_mm: 1012.2
formula: ellipse_equation_1
confidence: medium
validated_against_manual: false
auto: true
edited: false
warnings: []
```

Allowed `status` values:

- `acquired`
- `operator_confirmed`
- `manual_input`
- `estimated_pending_acceptance`
- `unavailable`
- `invalidated`

Allowed `source` values:

- `geometric_2d`
- `mesh_visual_hull`
- `operator_input`
- `derived_pattern_parameter`

The app must visually distinguish confidence from validation. A high-confidence detector output is not a validated anthropometric measurement.

---

## 15. Export contracts

### 15.1 `body_measurements.yaml`

Export only after required fields for the selected pattern profile are present and operator-confirmed.

```yaml
metadata:
  session_id: "<uuid>"
  garmentcode_commit: "<pinned commit>"
  unit: cm
  validation_status: unvalidated_against_manual_anthropometry
body:
  height: 171.0
  head_l: null
  neck_w: 18.9
  shoulder_w: 36.5
  shoulder_incl: 21.7
  armscye_depth: 12.9
  arm_length: 53.0
  arm_pose_angle: 45.5
  bust: 99.8
  underbust: 86.2
  waist: 84.3
  hips: 103.0
  leg_circ: 60.2
  wrist: 16.6
  back_width: 47.7
  waist_back_width: 39.1
  hip_back_width: 54.8
  waist_line: 36.9
  waist_over_bust_line: 40.6
  bust_line: 25.7
  vert_bust_line: 21.1
  hips_line: 23.5
  crotch_hip_diff: 8.8
  hip_inclination: 9.9
  bust_points: null
  bum_points: null
```

The example values are schema illustrations only and must never be emitted as defaults.

### 15.2 `measurement_report.yaml`

Include:

- Model names, versions, hashes, thresholds, and runtime.
- Lens-calibration identity and reprojection error.
- Scale method and inputs.
- Every measurement provenance record.
- Automatic and edited landmarks.
- Warnings and failed quality checks.
- `validated_against_manual: false` until a validation dataset exists.
- Links/relative paths to all visual artifacts.

### 15.3 Export gate

Pattern-ready export is blocked when:

- A required field is `null`, `unavailable`, or `invalidated`.
- An `estimated_pending_acceptance` value has not been accepted.
- A required source view failed its quality gate.
- The GarmentCode compatibility test fails.

Allow report export at any time, including partial/failed sessions.

---

## 16. Web application specification

### 16.1 Capture page

Show:

- Consent checkbox and consent-version text.
- Subject code, not full name.
- Scale-mode selector.
- Stature or reference-target input as applicable.
- Garment profile selector: core upper body, top/shirt, skirt, trousers/cargo, or full pattern set.
- Live preview with pose guide and framing boundaries.
- Capture buttons for required views with a three-second countdown.
- Upload alternative for every view.
- Gate result and specific retake instructions.
- Capture-progress indicator.

Instructions must include fitted clothing, shoes removed, hair controlled, uncluttered background, consistent posture, and stable camera.

### 16.2 Processing page

Show one processing step at a time as it completes:

- Stage timeline.
- Raw and current images.
- Per-stage status, runtime, dimensions, and parameters.
- Retry from failed stage.
- Cancel processing.

Do not hide intermediate images after completion.

### 16.3 Review page

Desktop layout:

- Left: view selector and preprocessing stage filmstrip.
- Center: large zoomable image/canvas with overlays.
- Right: measurement table and provenance/details drawer.

Mobile layout:

- Stacked viewer, stage selector, and measurement cards.
- Sticky primary action that does not obscure content.
- Minimum 44-by-44-pixel touch targets.

Review functions:

- Toggle mask, contour, keypoints, landmark names, slice rows, paths, and values.
- Drag landmarks and rows.
- Compare automatic versus edited result.
- Filter measurements by core, torso, arm, lower body, mesh, warning, or missing.
- Display raw value, final value, unit, method, views, confidence, and validation status.
- Require a note for direct numeric overrides.
- Confirm measurements before pattern export.

### 16.4 Final measurement display

The results area must have:

1. **Montazerian core measurements** prominently displayed.
2. **Pattern-ready body parameters** grouped by torso, arms, and lower body.
3. **Additional research measurements** in a separate expandable section.
4. Missing/blocked measurements with an actionable reason.
5. Export buttons for YAML, CSV, and provenance report.

Use tabular numerals. Do not imply medical or diagnostic use.

### 16.5 Accessibility and visual design

- Semantic HTML and one logical `h1` per page.
- Keyboard operation for capture, stage navigation, overlay toggles, and landmark editing alternatives.
- Visible focus states and WCAG AA text contrast.
- Light and dark themes.
- Body text at least 16 px; metadata at least 12 px.
- Responsive behavior from 375 px through desktop widths.
- Neutral surfaces with one restrained accent color.
- No decorative gradients, irrelevant imagery, or icon-only actions without labels/tooltips.
- Respect `prefers-reduced-motion`.

---

## 17. API

```text
POST   /api/sessions
GET    /api/sessions
GET    /api/sessions/{id}
DELETE /api/sessions/{id}

POST   /api/sessions/{id}/views/{view}
GET    /api/sessions/{id}/views/{view}/gate
POST   /api/sessions/{id}/process
GET    /api/sessions/{id}/process/status

GET    /api/sessions/{id}/artifacts
GET    /api/sessions/{id}/artifacts/{view}/{stage_id}

GET    /api/sessions/{id}/landmarks
PUT    /api/sessions/{id}/landmarks
POST   /api/sessions/{id}/landmarks/reset

GET    /api/sessions/{id}/measurements
PUT    /api/sessions/{id}/measurements/{name}
POST   /api/sessions/{id}/measurements/recompute
POST   /api/sessions/{id}/confirm

GET    /api/sessions/{id}/export/body
GET    /api/sessions/{id}/export/extended
GET    /api/sessions/{id}/export/report
GET    /api/sessions/{id}/export/csv
```

Rules:

- JSON for metadata; multipart upload for images.
- Validate session ownership/existence and legal state transitions.
- Prevent path traversal by resolving every path beneath the configured data root.
- Enforce upload size and pixel-count limits.
- Return structured error codes and human-readable messages.
- Never include stack traces in production responses.

---

## 18. Storage and privacy

Use SQLite from the Python standard library for session index, consent, status, and edit audit. Store images and YAML/JSON files on disk; never store image blobs in SQLite.

Minimum tables:

- `schema_version`
- `sessions`
- `edits`
- `deletions`

Requirements:

- SQLite WAL mode and foreign keys enabled.
- All database access through `store.py`.
- Session deletion removes files, session row, and edit rows, then retains only a non-personal deletion event.
- Configurable image-retention interval.
- Unconfirmed abandoned sessions have a shorter configurable retention interval.
- Do not log subject codes, body values, image paths, or images.
- No network synchronization.
- Document encrypted-disk and backup requirements.
- Deleting a session must be idempotent and recover cleanly from partial filesystem failure.

---

## 19. Validation plan without manual ground truth

Until standardized manual measurements exist, separate verification into three layers.

### 19.1 Mathematical verification

- Circle case for each ellipse formula.
- Comparison with numerical ellipse perimeter integration.
- Known widths, depths, path lengths, and angles on synthetic masks.
- Invariance to image offset and proportional resizing when scale is updated consistently.

### 19.2 Known-object verification

Use calibrated rectangular, circular, cylindrical, and elliptical objects plus a mannequin/dress form with independently known dimensions. Report absolute error and repeated-capture variation. Do not generalize object results to human accuracy.

### 19.3 Repeatability

For each consenting participant or mannequin:

- At least three complete recaptures with repositioning.
- Report mean, standard deviation, range, and coefficient of variation per measurement.
- Treat repeatability as precision only, not accuracy.

### 19.4 Later human validation

Prepare the schema to add standardized manual references later. The future protocol should use trained measurers, repeated readings, agreed anatomical definitions, and agreement/error analysis. Do not add a manual-reference value to the current dataset unless its collection protocol is recorded.

---

## 20. Testing requirements

### 20.1 Unit tests

- Geometry formulas and degenerate inputs.
- Pixel/metric transforms.
- Physical-height row matching.
- Connected-component and hole-filling behavior.
- Measurement dependency graph.
- Null propagation and export gates.
- Atomic writes and path safety.
- Database state transitions and deletion.

### 20.2 Synthetic visual tests

Generate silhouettes with known:

- Torso widths and depths at chest, bust, waist, and hip.
- Asymmetric front/back depths.
- Shoulder slopes.
- Arm segments.
- Crotch and leg levels.
- Limb cross-sections.

Save expected-versus-actual overlay images as test artifacts.

### 20.3 Integration tests

- Uploaded front plus side produces all eligible core outputs.
- Front plus both sides records bilateral differences and averages only valid values.
- Four-view mode enables front/back-distribution fields.
- Multi-view failure does not destroy valid two-view outputs.
- Landmark edits recompute only dependent measurements.
- Every displayed stage corresponds to a manifest entry and readable file.
- Export is blocked for unresolved required fields.
- Export loads in the pinned GarmentCode revision.

### 20.4 UI tests

At minimum, manually verify at 375 px and 1280 px:

- Capture/upload flow.
- Stage gallery, zoom, comparison, and error recovery.
- Landmark review with keyboard alternative.
- Measurement filtering and provenance drawer.
- Light/dark modes.
- No clipped controls or horizontal page overflow.

### 20.5 Performance tests

Record on the target CPU:

- Model initialization time.
- Segmentation time per view.
- Pose time per view.
- Artifact-generation time.
- Measurement recomputation time after one landmark edit.
- Multi-view reconstruction time when enabled.

Do not invent a passing time threshold. Establish the baseline in Milestone 2, then obtain approval for thresholds.

---

## 21. Milestones

### M0 — Skeleton and storage

- Repository structure.
- Configuration validation.
- FastAPI health endpoint and static page.
- SQLite schema and session lifecycle.
- Atomic file helper.
- `scripts/check.sh` running `ruff` and `pytest`.

Stop and report.

### M1 — Design-system proof and UI shell

- Create a temporary design-system page with typography, surfaces, controls, image viewer, measurement table, light/dark themes, and contrast checks.
- Validate at mobile and desktop widths.
- Carry approved tokens into the app and remove the temporary page.
- Build nonfunctional capture, processing, and review shells.

Stop and report screenshots and accessibility checks.

### M2 — Vision and visual artifacts

- Model provisioning and hash verification.
- Image ingestion, orientation, optional undistortion, enhancement.
- YOLO26 segmentation and pose wrappers loaded once.
- Pose gate and mask cleanup.
- Save and display stages 00 through 10.
- Record CPU baseline.

Stop and provide one complete artifact sequence per view.

### M3 — Core Montazerian geometry

- Scale and physical-height alignment.
- Editable core landmarks.
- Shoulder and arm lengths.
- Chest, bust, waist, and hip breadth/depth pairing.
- Montazerian two-equation ellipse logic.
- Core result panel and provenance.
- Synthetic tests.

Stop and report formulas, artifacts, and tests.

### M4 — Extended 2D measurements

- Neck, shoulder, armscye, torso-path, wrist, thigh/leg, and front/back-arc methods supported by the accepted views.
- Jin-style lower-body side landmarks.
- Operator review and dependency-based recomputation.
- `extended_measurements.yaml`.

Stop and list acquired, manual-required, and unavailable fields.

### M5 — Multi-view mesh

- Guided multi-angle capture.
- Reconstruction, mesh QA, and mesh viewer/download.
- Planar sections and surface paths.
- GarmentCodeData-style measurement methods.
- Graceful failure with 2D fallback.

Stop and provide mesh-quality diagnostics and comparison against synthetic/known geometry.

### M6 — Final review and export

- Full preprocessing timeline.
- Draggable landmarks and path controls.
- Numeric override with mandatory note.
- Confirmation flow.
- YAML/CSV/report export.
- Pinned GarmentCode load test.

Stop and report end-to-end results.

### M7 — Privacy, retention, and hardening

- Retention task.
- Idempotent deletion.
- Upload hardening and path tests.
- Production error handling.
- README deployment, privacy, calibration, capture, and limitations sections.

Stop and run the full definition-of-done checklist.

---

## 22. Definition of done

The prototype is complete only when:

- Raw images are preserved and visible.
- Every required preprocessing stage is generated, manifested, and viewable.
- The final six Montazerian core measurements are displayed with their intermediate dimensions and formula provenance.
- Extended measurements clearly state whether they came from 2D geometry, mesh geometry, operator input, or remain unavailable.
- Every measurement has a visual overlay or an explicit explanation of why no 2D overlay applies.
- Landmark edits recompute the correct dependent measurements.
- Missing values are never silently replaced.
- Pattern-ready export is schema-valid and loads with the pinned GarmentCode revision.
- The interface works at 375 px and 1280 px, in light and dark themes, with keyboard access.
- Session deletion and retention tests pass.
- `ruff check` and `pytest` pass.
- Runtime is reported on the target CPU.
- README states that current outputs are not validated against standardized manual anthropometry.
- README documents capture protocol, scale method, lens calibration, formulas, assumptions, limitations, privacy, and later validation plan.

---

## 23. Prohibited shortcuts

Do not:

- Hide or overwrite preprocessing artifacts.
- Treat a detection confidence score as measurement accuracy.
- Claim Montazerian's reported error for this implementation.
- Use image width, EXIF focal length, or assumed distance as metric scale.
- Use pose joints as exact tailoring landmarks without review.
- Infer crotch paths from one vertical distance.
- Export demographic means as acquired measurements.
- Fill missing values with zero.
- Couple body measurement code to garment style/ease parameters.
- Download models or datasets during normal app startup.
- Continue to the next milestone without approval.

When uncertain, stop, identify the exact unresolved requirement, state the assumption the code would otherwise make, and wait for confirmation.
