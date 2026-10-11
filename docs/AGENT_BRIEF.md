# Coding Agent Brief: Body Measurement Demo (Objective 1.1)

How to use this file: put it at the repo root (or paste it as your first message).
Ask the agent to build **one milestone at a time**. After each milestone it must stop,
run the tests, and report what passed. Do not let it start the next milestone until
you have checked the debug images yourself.

---

## 1. What we are building

A local web app that runs on the measuring-station PC (Core i5, 16 GB RAM, no GPU).

1. The operator captures a **front** photo and **at least one side** photo (left, right,
   or both) from a USB webcam.
2. The app finds the body silhouette (YOLO26-seg) and keypoints (YOLO26-pose).
3. A **geometric engine** converts silhouette widths and depths into body
   measurements in cm. It uses no machine-learned measurement model.
4. The tailor reviews landmarks drawn over the photos. Dragging a landmark recomputes
   the affected measurements.
5. The app exports `body_measurements.yaml` in the **GarmentCode body-parameter format**
   (Korosteleva & Sorkine-Hornung, 2023), which the pattern generator (Objective 1.2)
   loads directly, plus a sidecar `measurement_report.yaml` with provenance.

This is a demo of the method, not a product. Prefer clear, testable code over features.

## 2. Hard constraints

- Python 3.11, CPU only, runs fully offline. No cloud APIs, no telemetry.
- Backend: FastAPI + Uvicorn. Frontend: plain HTML, CSS and vanilla JavaScript served by
  FastAPI (no build step). Use `<canvas>` for overlays.
- Libraries: `ultralytics`, `opencv-python`, `numpy`, `pyyaml`, `pydantic`, `pytest`,
  `ruff`, `trimesh` (bench test only), `pygarment` (load test only). Ask before adding anything else.
- Models: `yolo26s-seg.pt` and `yolo26s-pose.pt` (pretrained). Model paths come
  from `config/app.yaml`, so fine-tuned weights can be swapped in later without code
  changes.
- Do **not** train, fine-tune, or download datasets. Do **not** invent
  measurements, correction factors, or scales. Every scale comes from
  `config/calibration.yaml`.
- Internal unit: millimetres (float). Output unit: centimetres, 1 decimal.
- PEP 8, type hints everywhere, `ruff check` clean, docstrings on public functions.
- Privacy: images stay in `data/sessions/<id>/`. Each session has a consent checkbox
  that must be ticked before capture, and a "Delete session" button that removes the
  folder.

## 3. Repository layout

```
bodymeasure/
├─ AGENT_BRIEF.md
├─ config/
│  ├─ app.yaml            # model paths, thresholds, camera index, rotation
│  ├─ calibration.yaml    # written by the calibration page
│  └─ schema.yaml         # measurement names, units, which views they need
├─ src/bodymeasure/
│  ├─ api.py              # FastAPI routes
│  ├─ session.py          # session folders, state, export
│  ├─ calibrate.py        # checkerboard intrinsics + body-plane scale
│  ├─ preprocess.py       # rotate, undistort
│  ├─ vision.py           # YOLO26 seg + pose wrappers (one load, reused)
│  ├─ posegate.py         # accept/reject a capture, with reasons
│  ├─ mask_clean.py       # mask clean-up
│  ├─ landmarks.py        # rows and points from mask + keypoints
│  ├─ geometry.py         # widths, depths, girths, lengths, angles
│  ├─ measure.py          # landmarks + geometry -> measurements
│  └─ overlay.py          # debug and review images
├─ web/                   # index.html, calibrate.html, review.html, app.js, style.css
├─ tests/
│  ├─ synthetic/          # generated silhouettes with known answers
│  └─ test_*.py
└─ data/                  # git-ignored: sessions/, calib/
```

## 4. Data contracts

### config/calibration.yaml
```yaml
camera_matrix: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
dist_coeffs: [k1, k2, p1, p2, k3]
image_size: [width, height]        # after rotation
rotation: 90                       # degrees clockwise applied to raw frames (webcam mounted portrait)
mm_per_px_x: 1.04                  # horizontal scale at the body plane (floor mark)
mm_per_px_y: 1.04                  # vertical scale at the body plane
floor_row: 1880                    # image row of the floor line at the floor mark
calibrated_at: 2026-10-11T10:00:00+08:00
```

### Session folder `data/sessions/<id>/`
```
subject.yaml      # id, sex (M/F/other), consent: true, notes
front.jpg  side_left.jpg  side_right.jpg      # raw captures (side files optional, >= 1 required)
<view>_mask.png   <view>_kp.json              # model outputs per view
landmarks.json    # all rows/points used, in image px, editable
body_measurements.yaml    # output for Objective 1.2 (GarmentCode format)
measurement_report.yaml   # provenance sidecar
debug/            # overlay PNGs for every step
```

### body_measurements.yaml (consumed by pygarment / GarmentCode)
Top-level key `body:`, flat, centimetres and degrees, **exactly the 26 primary keys** below.
Do **not** write the underscore keys (`_waist_level`, `_leg_length`, `_bust_line`,
`_base_sleeve_balance`, `_hip_inclination`, `_shoulder_incl`, `_armscye_depth`):
pygarment's `BodyParameters` recomputes them every time it loads the file.
```yaml
body:
  arm_length: 53.0
  arm_pose_angle: 45.5
  armscye_depth: 12.9
  back_width: 47.7
  bum_points: 18.2
  bust: 99.8
  bust_line: 25.7
  bust_points: 16.9
  crotch_hip_diff: 8.8
  head_l: 26.3
  height: 171.0
  hip_back_width: 54.8
  hip_inclination: 9.9
  hips: 103.0
  hips_line: 23.5
  leg_circ: 60.2
  neck_w: 18.9
  shoulder_incl: 21.7
  shoulder_w: 36.5
  underbust: 86.2
  vert_bust_line: 21.1
  waist: 84.3
  waist_back_width: 39.1
  waist_line: 36.9
  waist_over_bust_line: 40.6
  wrist: 16.6
```

### measurement_report.yaml (for the tailor and for Objective 2)
```yaml
subject_id: S014
side_views_used: [left, right]
values:
  waist:      {source: geometric, confidence: medium, edited: false, views: [front, side_left, side_right]}
  bust_points: {source: default_ratio, confidence: estimated, edited: false}
warnings:
  - "waist depth differs 1.4 cm between left and right side views"
```

## 5. Algorithm specification (implement exactly; ask before deviating)

### 5.1 Preprocess
1. Rotate the raw frame by `rotation`, then `cv2.undistort` with the calibration.
2. All later steps use the undistorted image.

### 5.2 Pose gate (`posegate.py`) - return `(ok: bool, reasons: list[str])`
Use COCO-17 keypoints (0 nose, 1/2 eyes, 3/4 ears, 5/6 shoulders, 7/8 elbows,
9/10 wrists, 11/12 hips, 13/14 knees, 15/16 ankles).
- Front: all 17 keypoints confidence >= 0.5 (from app.yaml).
- Front: each upper arm 20-45 degrees from the torso line (A-pose). Wrists must not overlap the torso mask.
- Side: the visible side's shoulder, hip, knee and ankle confidence >= 0.5. The
  shoulder-to-shoulder horizontal distance must be < 25% of the front view's (truly side-on).
- Every view: the whole mask lies inside the frame with >= 2% margin top and bottom.
  The lowest mask row must be within ±15 px of `floor_row` (standing on the mark).
- The reasons are shown to the operator verbatim, e.g. "Left arm too close to body".

### 5.3 Mask clean-up (`mask_clean.py`)
1. Keep the person instance with the largest area (class 0, conf >= 0.5).
   Use `retina_masks=True` and `imgsz` from app.yaml (default 1280).
2. Morphological opening, then closing, with a 5 px elliptical kernel.
3. Keep the largest connected component.
4. Fill only holes smaller than 500 px² (gaps between arm and torso or between the legs
   are larger and must remain background).

### 5.4 Heights and view alignment
- Height of image row `y` above the floor: `h_mm = (floor_row - y) * mm_per_px_y`.
- Front and side rows are matched by `h_mm`, never by pixel row.
- Side facing direction: the nose keypoint x relative to the visible ear's x. Mirror
  a side image if needed so the body always faces right before landmarking.

### 5.5 Landmarks (`landmarks.py`)
Definitions follow the GarmentCode measurement sheet (`docs/Body Measurements GarmentCode.pdf`
in the GarmentCode repository). "Torso run" at row y means the contiguous foreground run
that contains `torso_x`, the midpoint of the two hip keypoints' x.

Front view:
- **Head top**: smallest y of the mask.
- **Neck-base row**: the row where the neck widens into the shoulders. Find the row of minimum
  mask width between the ear row and the shoulder-keypoint row, then move down to the first row
  where the width exceeds that minimum by 25%. **Neck-base side points (L/R)** are the run's
  ends on that row.
- **Shoulder points (L/R)**: on the outer contour between the neck-base row and 10% of
  shoulder-to-hip distance below the shoulder keypoint, the point of maximum turning angle
  (curvature over ±15 contour points). These are proxies for the collarbone ends.
- **Armpit row**: the first row below the shoulder keypoints where background appears between
  arm and torso on both sides.
- **Waist row**: the row of minimum torso-run width between 30% and 90% of the span from the
  armpit row to the hip-keypoint row.
- **Crotch row**: the first row below the hip keypoints where background appears between the legs.
- **Wrist points (L/R)**: the narrowest arm cross-section within ±4 cm of each wrist keypoint,
  measured perpendicular to the elbow-wrist line.

Side view (body faces right after mirroring, see 5.4):
- **Balance line**: the line through the side-view shoulder keypoint and hip keypoint,
  evaluated at each row. It splits every row into a front depth `d_f` and a back depth `d_b`.
  This is a proxy for GarmentCode's coronal "balance line". Make the line selectable in the
  review UI.
- **Nape point**: on the back contour between the ear row and the shoulder-keypoint row,
  the point of maximum turning angle (neck-to-back junction).
- **High point shoulder (HPS)**: the point on the top contour of the shoulder where it crosses
  the balance line.
- **Bust row**: the row of maximum `d_f` between the armpit row and the waist row.
- **Underbust row**: the row of minimum `d_f` between the bust row and the waist row
  (for flat chests it falls close to the bust row; that is expected).
- **Hip row**: the row of maximum `d_b` (buttock prominence) between the waist row and the crotch row.
- **Thigh rows**: every row from the crotch row to crotch + 10 cm.

Store every landmark in `landmarks.json` as `{name, view, x, y, auto: true}`.

### 5.6 Geometry (`geometry.py`) - pure functions, no I/O
- `width_at(mask, y, x_ref) -> mm`: length of the run containing `x_ref` × `mm_per_px_x`.
- `split_depth_at(side_mask, h_mm, balance_x) -> (d_f, d_b)` in mm.
- `ellipse_perimeter(a, b)`: Ramanujan II,
  `P = π(a+b)(1 + 3h / (10 + sqrt(4 - 3h)))`, `h = (a-b)² / (a+b)²`.
- **Two-semi-ellipse girth** (the body is not front-back symmetric):
  `a = width / 2`, `front_arc = P(a, d_f) / 2`, `back_arc = P(a, d_b) / 2`,
  `girth = front_arc + back_arc`. The `*_back_width` measurements are `back_arc`.
- `contour_length(points)`: polyline length in mm. `hull_length(points)` is the same but taken
  on the convex hull of the points (tape-measure behaviour over hollows).
- If two side views exist, average `d_f` and `d_b` row by row. Add a warning if the
  girths they give differ by more than 10 mm.

### 5.7 Measurements (`measure.py`): the 26 GarmentCode keys
F = front view, S = side view. Confidence goes into `measurement_report.yaml`.

| key | view | method | confidence |
|---|---|---|---|
| height | F | floor to head top | high |
| head_l | S | straight distance from the nape point to the head top | medium |
| neck_w | F | distance between the neck-base side points | medium |
| shoulder_w | F | distance between the two shoulder points | medium |
| shoulder_incl | F | angle above horizontal of neck-base side point → shoulder point, mean L/R | medium |
| armscye_depth | F | shoulder-point height minus armpit-row height, mean L/R | medium |
| arm_length | F | shoulder point → elbow kp → wrist kp polyline, mean L/R | medium |
| arm_pose_angle | F | angle of shoulder→elbow below horizontal, mean L/R (a pose value, not a size) | high |
| bust | F+S | two-semi-ellipse girth at bust row | medium |
| underbust | F+S | two-semi-ellipse girth at underbust row | low |
| waist | F+S | two-semi-ellipse girth at waist row | medium |
| hips | F+S | two-semi-ellipse girth at hip row | medium |
| leg_circ | F+S | maximum over thigh rows of a one-leg ellipse girth (front run of one leg, side leg depth) | medium |
| wrist | F | π × narrowest wrist width (circle) | low |
| back_width | F+S | back_arc at bust row | low |
| waist_back_width | F+S | back_arc at waist row | low |
| hip_back_width | F+S | back_arc at hip row | low |
| waist_line | S | back contour length from nape point down to the waist row | medium |
| waist_over_bust_line | S | hull_length of the front contour from HPS to the waist row | low |
| bust_line | S | the same path, from HPS to the bust row | low |
| vert_bust_line | S | nape height minus bust-row height | medium |
| hips_line | F+S | waist-row height minus hip-row height | medium |
| crotch_hip_diff | F+S | hip-row height minus crotch-row height | medium |
| hip_inclination | F | angle from vertical of the torso side edge from waist row to hip row, mean L/R | medium |
| bust_points | — | not visible in a silhouette: tailor enters it; default 0.170 × bust | estimated |
| bum_points | — | not visible in a silhouette: tailor enters it; default 0.176 × hips | estimated |

The two default ratios come from GarmentCode's `mean_all.yaml` (16.95 / 99.84 and
18.23 / 103.48). The review UI must highlight every `estimated` and `low` value.

`measure.py` takes `landmarks.json` as input, so the review UI can edit landmarks and call it
again. Round every output to 1 decimal.

## 6. Web app pages and API

Pages:
1. `/calibrate`: live preview. Two steps: (a) capture 15-20 checkerboard frames and
   compute intrinsics; (b) capture a reference board of known size held upright on the floor
   mark and compute `mm_per_px_x/y` and `floor_row`. Show the computed numbers before saving.
2. `/` (capture): consent checkbox, subject ID, sex. Live preview at the camera's full resolution
   (`getUserMedia` with ideal 1920×1080), with a drawn outline guide and floor-mark
   overlay. Buttons: Capture Front, Capture Left, Capture Right, with a 3-second countdown.
   After each capture the server runs the pose gate. Show ✓, or the reasons with a Retake button.
   "Measure" is enabled once front plus at least one side pass.
3. `/review/<id>`: front and side images with mask outline, slice lines and points. Lines
   and points are draggable. Each drag calls the recompute endpoint and updates the table. Each
   value is editable with a note, which sets `edited: true`. Values marked `estimated` or `low`
   are highlighted, and `bust_points` / `bum_points` show an input box. Buttons: Export YAML,
   Delete session.

API (JSON in and out; images as multipart):
```
POST   /api/calibration/checkerboard        frames -> intrinsics
POST   /api/calibration/scale               frame + board size -> scale, floor_row
POST   /api/sessions                        {subject_id, sex, consent} -> {id}
POST   /api/sessions/{id}/views/{view}      view in {front, side_left, side_right}; image -> gate result
POST   /api/sessions/{id}/measure           -> landmarks + measurements
PUT    /api/sessions/{id}/landmarks         edited landmarks -> recomputed measurements
GET    /api/sessions/{id}/overlay/{view}    PNG
GET    /api/sessions/{id}/export            body_measurements.yaml
GET    /api/sessions/{id}/report            measurement_report.yaml
DELETE /api/sessions/{id}
```
Also accept an **image upload** in place of the webcam on the capture page, so the demo
works without the station.

## 7. Milestones (stop after each one)

**M0 - Skeleton.** Repo layout, config loading, FastAPI serving an empty page, `pytest`
and `ruff` passing in CI-style script `scripts/check.sh`.

**M1 - Geometry with synthetic tests.** `geometry.py` + `tests/synthetic/` generator:
draw front and side silhouettes of stacked ellipses (known widths and depths) at a known
mm/px. Tests: `ellipse_perimeter(a, a) == 2πa` to 1e-9; it matches numerical
integration within 0.01% for b/a in [0.3, 1]; `width_at` and `depth_at` recover the drawn
sizes within 1 px; `split_depth_at` recovers asymmetric front/back depths within 1 px.

**M2 - Calibration page.** Checkerboard intrinsics + body-plane scale, written to
calibration.yaml. Test with saved sample frames in `tests/data/`.

**M3 - Vision + pose gate + clean-up.** YOLO26 wrappers (models loaded once at startup),
`posegate.py`, `mask_clean.py`, debug overlays. Report CPU time per image.

**M4 - Landmarks + measurements.** `landmarks.py`, `measure.py`, end-to-end on uploaded
images, overlay PNGs for every landmark. Synthetic-figure tests where landmark rows are known.

**M4b - GarmentCode bench test.** Clone the GarmentCode repository (MIT licence) and use
`assets/bodies/mean_all.obj`, `mean_female.obj`, `mean_male.obj` and the two
`*_smpl_average_A40.obj` bodies, each with its `.yaml` of true measurements. Write
`bench/render_silhouettes.py` (trimesh): render front and side binary silhouettes with an
orthographic projection at 1 mm/px, feet on a fixed floor row. Run the engine on them
(keypoints from YOLO26-pose on a plain grey rendering, or from mesh joints if that fails) and
write `bench/results.csv`: key, true, measured, error. **Report the errors; do not tune
thresholds to fit these five bodies.** Also check that pygarment's `BodyParameters` loads
every exported file without error.

**M5 - Capture UI.** Webcam capture flow, countdown, gate feedback, retake.

**M6 - Review UI.** Draggable landmarks, recompute, edit values, export, delete.

## 8. Definition of done (whole demo)

- Front plus one side, and front plus both sides, both produce body_measurements.yaml
  (26 keys, loadable by pygarment) and measurement_report.yaml.
- Every measurement in the table has a debug overlay showing where it was taken.
- Dragging a landmark changes only the measurements that depend on it.
- `scripts/check.sh` passes (ruff + pytest).
- README: setup, calibration procedure, capture procedure (tight clothing, A-pose,
  standing on the mark), and known limitations.

## 9. When unsure

Do not guess. Stop and ask about anything not specified here, especially
landmark definitions, thresholds, or new dependencies. List the assumption
you would make, and wait for confirmation.
