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
  measurements, correction factors, or scales. Every scale comes from the
  subject's **measured height** over that view's head-to-floor pixel span; the
  optional lens calibration in `config/calibration.yaml` carries a camera model
  and nothing else.
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
│  ├─ calibration.yaml    # OPTIONAL lens model, written by the monitor tool
│  └─ schema.yaml         # measurement names, units, which views they need
├─ src/bodymeasure/
│  ├─ api.py              # FastAPI routes
│  ├─ session.py          # session folders, state, export
│  ├─ preprocess.py       # rotate, undistort (skipped with no lens model)
│  ├─ vision.py           # YOLO26 seg + pose wrappers (one load, reused)
│  ├─ posegate.py         # accept/reject a capture, with reasons
│  ├─ mask_clean.py       # mask clean-up
│  ├─ landmarks.py        # rows and points from mask + keypoints, and the scale
│  ├─ geometry.py         # widths, depths, girths, lengths, angles
│  ├─ measure.py          # landmarks + geometry -> measurements
│  └─ overlay.py          # debug and review images
├─ tools/
│  └─ calibrate_lens_on_monitor.py   # optional, separate milestone
├─ web/                   # index.html, review.html, app.js, style.css
├─ tests/
│  ├─ synthetic/          # generated silhouettes with known answers
│  └─ test_*.py
└─ data/                  # git-ignored: sessions/
```

## 4. Data contracts

### config/calibration.yaml (optional)
```yaml
camera_matrix: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
dist_coeffs: [k1, k2, p1, p2, k3]
rotation: 90                       # degrees clockwise applied to raw frames (webcam mounted portrait)
```
There is no scale here and no floor mark: the station has neither. If
`camera_matrix` is missing, frames are measured undistorted and
`measurement_report.yaml` carries the warning `lens distortion not corrected`.
Written by `tools/calibrate_lens_on_monitor.py`, which shows a checkerboard on the
monitor and asks for that screen's visible width in millimetres.

### Session folder `data/sessions/<id>/`
```
subject.yaml      # id, sex (M/F/other), height_cm, consent: true, notes
front.jpg  side_left.jpg  side_right.jpg      # raw captures (side files optional, >= 1 required)
<view>_mask.png   <view>_kp.json              # model outputs per view
landmarks.json    # all rows/points used, in image px, editable, plus each view's scale
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
  height:     {source: tailor_input, confidence: high, edited: false, views: [front]}
  bust_points: {source: default_ratio, confidence: estimated, edited: false}
warnings:
  - "waist depth differs 1.4 cm between left and right side views"
  - "lens distortion not corrected"
```

## 5. Algorithm specification (implement exactly; ask before deviating)

### 5.1 Preprocess
1. Rotate the raw frame by `rotation`, then `cv2.undistort` with the stored lens
   model. With no `camera_matrix`, skip the undistortion: the report carries
   `lens distortion not corrected` and the scale is unaffected.

### 5.2 Pose gate (`posegate.py`) - return `(ok: bool, reasons: list[str])`
Use COCO-17 keypoints (0 nose, 1/2 eyes, 3/4 ears, 5/6 shoulders, 7/8 elbows,
9/10 wrists, 11/12 hips, 13/14 knees, 15/16 ankles).
- Front: all 17 keypoints confidence >= 0.5 (from app.yaml).
- Front: each upper arm 20-45 degrees from the torso line (A-pose). Wrists must not overlap the torso mask.
- Side: the visible side's shoulder, hip, knee and ankle confidence >= 0.5. The
  shoulder-to-shoulder horizontal distance must be < 25% of the front view's (truly side-on).
- Every view: top and bottom margins each >= 8% of the frame height, the mask
  spanning >= 70% of the frame height, and the body's horizontal centre within
  the middle 30% of the frame width. There is no floor mark to stand on.
- The reasons are shown to the operator verbatim, e.g. "Left arm too close to body".

### 5.3 Mask clean-up (`mask_clean.py`)
1. Keep the person instance with the largest area (class 0, conf >= 0.5).
   Use `retina_masks=True` and `imgsz` from app.yaml (default 1280).
2. Morphological opening, then closing, with a 5 px elliptical kernel.
3. Keep the largest connected component.
4. Fill only holes smaller than 500 px² (gaps between arm and torso or between the legs
   are larger and must remain background).

### 5.4 Scale, heights and view alignment
- Each view carries its **own** scale, from the subject's measured height:
  `floor_row` = the lowest foreground row of that view's mask, `head_row` = the
  highest, and `mm_per_px = height_cm * 10 / (floor_row - head_row)`.
- Height of image row `y` above the floor, on the view that owns the row:
  `h_mm = (floor_row - y) * mm_per_px`.
- Front and side rows are matched by `h_mm`, never by pixel row, so the two views
  need not be the same size in the frame.
- `head_row` is a draggable landmark: dragging it down over hair or a cap
  recomputes that view's scale and every measurement from that view.
- Side facing direction: the nose keypoint x relative to the visible ear's x. Mirror
  a side image if needed so the body always faces right before landmarking.

### 5.5 Landmarks (`landmarks.py`)
Definitions follow the GarmentCode measurement sheet (`docs/Body Measurements GarmentCode.pdf`
in the GarmentCode repository). "Torso run" at row y means the contiguous foreground run
that contains `torso_x`, the midpoint of the two hip keypoints' x.

Front view:
- **Head row**: smallest y of the mask (the scale row; see 5.4).
- **Floor row**: largest y of the mask.
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

Store every landmark in `landmarks.json` as `{name, view, x, y, auto: true}`,
beside a per-view record of the `floor_row`, `head_row` and `mm_per_px` they
imply. The rows are the single source of truth; the scale beside them is what
those rows give.

### 5.6 Geometry (`geometry.py`) - pure functions, no I/O
- `width_at(mask, y, x_ref, mm_per_px) -> mm`: length of the run containing `x_ref` × `mm_per_px`.
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
| height | F | the tailor's measured `height_cm`, verbatim | high |
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
1. `/` (capture): consent checkbox, subject ID, sex, **height in cm** with the hint
   "Measured by tailor, shoes off. Do not use the client's own estimate." Live preview at the
   camera's full resolution (`getUserMedia` with ideal 1920×1080), with the capture
   instructions on screen - shoes off, hair tied flat or a tight cap, fitted clothing, A-pose -
   and a guide box showing the gate's bounds. Buttons: Capture Front, Capture Left,
   Capture Right, with a 3-second countdown. After each capture the server runs the pose
   gate. Show ✓, or the reasons with a Retake button. "Measure" is enabled once front plus
   at least one side pass.
2. `/review/<id>`: front and side images with mask outline, slice lines and points. Lines
   and points are draggable, including `head_row`, which sets that view's scale. Each drag
   calls the recompute endpoint and updates the table. Each value is editable with a note,
   which sets `edited: true`. Values marked `estimated` or `low` are highlighted, and
   `bust_points` / `bum_points` show an input box. Buttons: Export YAML, Delete session.

API (JSON in and out; images as multipart):
```
POST   /api/sessions                        {subject_id, sex, consent, height_cm} -> {id}
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

**M2 - Optional lens calibration (separate milestone).** `tools/calibrate_lens_on_monitor.py`:
draw a checkerboard on the station monitor, ask for the screen's visible width in mm, fit
`camera_matrix` / `dist_coeffs` with `cv2.calibrateCamera`. There is no scale step and no
calibration page.

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
- Dragging a landmark changes only the measurements that depend on it; dragging
  `head_row` rescales that view.
- A synthetic figure redrawn at any size and offset measures the same when
  `height_cm` matches what was drawn, and a +1 cm height error moves the waist by
  about `waist / height`.
- `scripts/check.sh` passes (ruff + pytest).
- README: setup, how the scale works, capture procedure (shoes off, hair tied
  flat, fitted clothing, A-pose), and known limitations.

## 9. When unsure

Do not guess. Stop and ask about anything not specified here, especially
landmark definitions, thresholds, or new dependencies. List the assumption
you would make, and wait for confirmation.

## 10. Storage (add to Milestone 0)

### 10.1 Approach
- Files on disk hold the large and pipeline-facing data. A single SQLite database
  (`data/app.db`, Python stdlib `sqlite3`, no ORM, no server) holds the index, consent and audit trail.
- Never store images or YAML contents inside the database. Store only paths relative to `data/`.
- All database access goes through `src/bodymeasure/store.py`. No other module opens
  `app.db` or deletes session folders.

### 10.2 Files (unchanged from section 4)
data/sessions/<session_id>/
  subject.yaml, front.jpg, side_left.jpg, side_right.jpg,
  <view>_mask.png, <view>_kp.json, landmarks.json,
  body_measurements.yaml, measurement_report.yaml, debug/

- `session_id` = UUID4 hex. Never put the subject's name in a path or file name.
- Write every file atomically: write `<name>.tmp`, then `os.replace` it to the final name.

### 10.3 Database schema (create on startup if missing)
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);

CREATE TABLE IF NOT EXISTS sessions (
  id              TEXT PRIMARY KEY,           -- UUID4 hex = folder name
  subject_code    TEXT NOT NULL,              -- tailor's client code, not a full name
  sex             TEXT CHECK (sex IN ('M','F','other')),
  height_cm       REAL NOT NULL,                 -- the measured height every view is scaled by
  consent_at      TEXT NOT NULL,              -- ISO 8601, Asia/Manila
  consent_version TEXT NOT NULL,              -- version of the consent form text shown
  created_at      TEXT NOT NULL,
  status          TEXT NOT NULL CHECK (status IN
                    ('capturing','measured','confirmed','exported')),
  views           TEXT NOT NULL DEFAULT '[]', -- JSON list, e.g. ["front","side_left"]
  confirmed_at    TEXT,
  images_purge_after TEXT                     -- date after which photos and masks are deleted
);

CREATE TABLE IF NOT EXISTS edits (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  at          TEXT NOT NULL,
  kind        TEXT NOT NULL CHECK (kind IN ('landmark','value')),
  target      TEXT NOT NULL,                  -- landmark name or measurement key
  old_value   TEXT,                           -- JSON
  new_value   TEXT,                           -- JSON
  note        TEXT
);

CREATE TABLE IF NOT EXISTS deletions (        -- no personal data, by design
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id  TEXT NOT NULL,                  -- no foreign key: the session row is gone
  at          TEXT NOT NULL,
  scope       TEXT NOT NULL CHECK (scope IN ('session','images')),
  reason      TEXT NOT NULL CHECK (reason IN ('user_request','retention'))
);

### 10.4 store.py functions (type-hinted, each with a pytest test)
create_session(subject_code, sex, consent_version, height_cm) -> str
add_view(session_id, view) -> None
set_status(session_id, status) -> None
log_edit(session_id, kind, target, old, new, note) -> None
list_sessions(status: str | None = None) -> list[dict]
delete_session(session_id, reason) -> None
    # delete the folder, then the row, then insert into deletions,
    # all inside one transaction; if the folder delete fails, roll back.
purge_images(today) -> int
    # for sessions whose images_purge_after <= today: delete the photos, masks and debug/
    # but keep landmarks.json and both YAML files; log scope='images'.
export_csv(path) -> None
    # one row per confirmed session: session_id, subject_code, sex, then the 26 GarmentCode
    # keys, then edited flags. Used for Objective 2 analysis.

### 10.5 Configuration (config/app.yaml)
storage:
  data_dir: data
  retention_days_images: 30       # placeholder; set from the client agreement
  purge_on_startup: true          # run purge_images() whenever the app starts
  keep_debug_after_confirm: false # delete debug/ once a session is confirmed

`images_purge_after` = confirmed_at + retention_days_images.
Sessions that were never confirmed are purged 7 days after created_at.

### 10.6 API additions
GET    /api/sessions?status=confirmed   -> list from store.list_sessions
GET    /api/export.csv                  -> store.export_csv
DELETE /api/sessions/{id}               -> store.delete_session(reason='user_request')

### 10.7 Rules
- Do not log subject codes, measurements or image paths to the console or log files;
  log only session_id and event names.
- No network sync, cloud backup or telemetry. Backup = copying the whole data/ folder,
  documented in the README.
- README section "Data protection": the station disk must be encrypted
  (BitLocker on Windows), and the retention period must match the client's consent form.

### 10.8 Tests
- Creating, editing, confirming and deleting a session leaves no folder, no sessions row,
  no edits rows, and exactly one deletions row.
- purge_images deletes the photos but keeps body_measurements.yaml.
- A crash simulated mid-write (exception after the .tmp write) leaves the previous file intact.
- export_csv output contains no rows for unconfirmed sessions.
