# Image-based body measurement, pattern-ready export

Local web app that turns a front photograph plus at least one side photograph
into GarmentCode body parameters, using silhouette geometry only — no
machine-learned measurement model. Full specification:
[docs/AGENT_BRIEF.md](docs/AGENT_BRIEF.md).

> **These outputs have not been validated against standardised manual
> anthropometry.** Nothing here has been compared with a trained measurer using
> an agreed protocol, so no claim of accuracy or of equivalence to a tape measure
> is made or implied, and the prototype is not for any medical or diagnostic use.
> Confidence in the interface describes how sure a detector was, never how
> accurate a measurement is. The validation plan is at the end of this file.

**Status against the brief's milestones.** M0 (skeleton and storage), M1 (design
system and UI shell), M2 (vision and the thirteen visual stages), M3 (the
Montazerian core geometry) and the export half of M6 are implemented, together
with the M4b bench. **M4 (the extended 2D measurement families), M5 (the
multi-view mesh) and M7 (retention hardening) are not started**, and the brief
says to stop for approval before each. What is and is not implemented is listed
under *Where this stops* below. Measured behaviour against the five GarmentCode
reference bodies: [bench/README.md](bench/README.md).

## Tech stack

Deliberately boring: the whole pipeline is CPU-only Python behind one web server,
and the front end has no build step at all.

| Layer | What | Version / notes |
|---|---|---|
| Language | Python | 3.11 (pinned; the torch/ultralytics wheels target it) |
| Web server | FastAPI + Uvicorn | `0.115.6` / `0.34.0`; JSON API plus the static pages |
| Data validation | Pydantic | `2.10.4`, for `config/*.yaml` only |
| Config & output | PyYAML | `6.0.2`; app config, schema, and both exported files |
| Numerics | NumPy | `2.x`; opencv-python 5 requires numpy 2 |
| Vision | OpenCV | `>=4.10,<6`; **not** `-headless`, because ultralytics imports `cv2` |
| Detection & pose | Ultralytics YOLO26 | `>=8.4.52,<8.5`; `yolo26s-seg.pt` + `yolo26s-pose.pt`, CPU |
| Uploads | python-multipart | `0.0.20`, required by FastAPI for the image endpoints |
| Storage | SQLite | stdlib `sqlite3`; index, consent and audit trail only, no ORM |
| Front end | HTML, CSS, vanilla JS | no framework, no bundler, no npm; `<canvas>` overlays |
| Tests | pytest + FastAPI TestClient | drawn silhouettes with known answers in `tests/synthetic/` |
| Lint & format | ruff | `0.8.6`; `ruff check` + `pytest` in `scripts/check.sh` |
| Bench only | trimesh, plus `pygarment` from the GarmentCode clone | `requirements-bench.txt`; the PyPI wheel is unusable, see below |
| Packaging | Docker (`python:3.11-slim`) or conda | `environment.yml` pins the same requirements |

Everything runs offline: no cloud APIs, no telemetry, no network calls at
runtime. `pygarment` is imported from a `git clone` of
[GarmentCode](https://github.com/maria-korosteleva/GarmentCode) rather than PyPI,
because the published wheel installs under different top-level names, does not
ship the `BodyParameters` subclass, and pins a numpy older than opencv-python 5
accepts.

## Run it (Docker — nothing else to install)

```bash
docker compose up --build        # http://localhost:8000
docker compose down
```

Rebuild with `--build` whenever `requirements.txt` or the Dockerfile changes:
`docker compose up` on its own reuses the old image and will keep serving stale
dependencies. The whole repository is bind-mounted and uvicorn runs with
`--reload`, so every edit applies live. The first start also fetches the model
weights the checkout is missing; see *Model weights*.

Dependencies live in a venv at `/opt/venv` inside the image, already on `PATH`.
Shell in with `docker compose exec app bash` — not `bash -l`, because a login
shell resets `PATH` and you would get the system python instead.

## Lint and tests

```bash
docker compose run --rm app bash scripts/check.sh    # ruff check + pytest
```

The image carries the bench dependencies too, so the M4b bench runs in the
container as well once the GarmentCode clone is in `vendor/` (it is bind-mounted
from the host):

```bash
docker compose run --rm app python bench/run_bench.py vendor/GarmentCode/assets/bodies
```

## Local setup without Docker (conda)

Base miniconda is 3.13; this project pins **3.11**, because that is what the
ultralytics/torch wheels target at M3. So it gets its own env:

```bash
conda env create -f environment.yml    # python 3.11 + requirements.txt
conda activate bodymeasure
./scripts/check.sh                     # ruff check + pytest
uvicorn bodymeasure.api:app --reload   # http://localhost:8000
```

After changing `requirements.txt`: `conda env update -f environment.yml`.
`environment.yml` only names the python version and points pip at
`requirements.txt`, so the host env and the container install the same pins.

`src/` is put on the path by `pyproject.toml` (`pythonpath`), so nothing needs
installing in editable mode.

## Where the scale comes from

**The subject's measured height.** There is no calibration step and nothing to
stand on. Each view is scaled on its own:

```
floor_row = lowest foreground row of the mask
head_row  = highest  foreground row of the mask
mm_per_px = height_cm * 10 / (floor_row - head_row)
```

That is the whole rule. `height_cm` is measured by the tailor, shoes off, to
0.1 cm, and it scales every millimetre the engine reports. Front and side rows
are matched by the height above the floor in millimetres, never by pixel row, so
the two views do not have to be the same size in the frame.

**Loose hair breaks it.** The head row is whatever is highest in the mask, so hair
or a cap makes the body measure short. `head_row` is a draggable landmark in the
review page: drag it down to the top of the skull and every measurement from
that view is recomputed.

A one-centimetre error in the height moves every measurement by about
`1 / height` — under a centimetre on a 170 cm subject, which is why the tailor's
number and not the client's own estimate.

## Optional: correcting the lens

The station runs uncorrected by default. Frames are measured as they arrive and
`measurement_report.yaml` carries the warning `lens distortion not corrected`.

To correct it, without buying a calibration board:

```bash
python tools/calibrate_lens_on_monitor.py
```

It draws a checkerboard **on the station monitor** and asks for the screen's
visible width in millimetres, which makes the board's physical size known. It
then shows the board at 16 tilted positions, grabs a frame of each with the
camera, and `cv2.calibrateCamera` writes `camera_matrix` and `dist_coeffs` to
`config/calibration.yaml`. Run it on the station with nothing else on screen and
the room lit evenly. Nothing is written unless the fit's residual is under one
pixel.

`config/calibration.yaml` then holds only `camera_matrix`, `dist_coeffs` and
`rotation`. There is no scale in it: the scale comes from the height.

## Layout

| Path | What it holds |
|---|---|
| `config/app.yaml` | model paths and hashes, thresholds, enhancement, upload limits, the pinned GarmentCode commit, the garment profiles |
| `config/schema.yaml` | the 26 GarmentCode keys, their units, views and default confidence |
| `config/calibration.yaml` | optional lens model, written by the monitor tool; git-ignored |
| `src/bodymeasure/` | `api.py`, `session.py`, `artifacts.py`, `landmarks.py`, `measure.py`, `geometry.py`, `export.py`, and the rest |
| `tools/` | model provisioning, the GarmentCode compatibility check, the on-monitor lens calibration |
| `web/` | plain HTML/CSS/JS, no build step: capture, processing, review, sessions |
| `web/samples/` | three stored captures the capture page offers as mock input |
| `tests/` | pytest; `synthetic/` holds generated silhouettes with known answers |
| `data/` | sessions and the index — git-ignored, never committed |

## Capture procedure

Open `/` and work down the page.

1. **Subject.** Enter the subject code, sex, **measured stature**, the scale mode
   and the **garment profile**, tick consent, press *Start session*. No image is
   captured until consent is ticked; the server refuses otherwise, and it refuses
   a stature outside 100-220 cm, an unknown profile or an unknown scale mode.
   The profile decides which fields the export gate will later require.
2. **Camera.** Press *Start camera*. The browser is asked for a portrait stream
   (1080x1920), so the camera is bolted in the right way up and needs no physical
   rotation. The yellow box is where the body must sit: 8% clear top and bottom,
   filling at least 70% of the frame, and within the middle third across.
3. **Captures.** On screen, and worth reading to the subject:
   **shoes off**; **hair tied flat or a tight cap**; **fitted clothing**, no thick
   layers; **A-pose**, arms 20-45 degrees from the body and not touching it.

   Press *Capture all views* and the run takes over: front, left side, right side
   in turn, each with five seconds to get into the pose and a **three-second
   countdown** to hold still, and a shutter flash at the shot. The per-view
   capture buttons take a single view instead, and are what *Retake* leaves for
   redoing one view.

   The server segments each frame, writes stages 00 to 10, and runs the pose
   gate. A refusal arrives as a code and an instruction — `WRIST_OVERLAP: Left
   wrist overlaps the torso: move the hand away from the body` — so the subject
   is told what to change, not just that something is wrong. A refused view is
   still written, so its stages can be inspected.
4. **Process.** Enabled once the front and at least one side view pass. It opens
   `/process/<session id>`, which runs the landmark search and the measurements,
   writes stages 11 and 12, and steps through every stage with its runtime and
   parameters. From there, *Review*.

Without a webcam, every step also accepts an uploaded image, so the demo runs
away from the station.

### Stored samples

`web/samples/front.jpg`, `side_left.jpg` and `side_right.jpg` are three captured
views the page offers as *Use sample*, so the whole flow can be demonstrated with
no camera, no subject and no floor. They are committed files, dropped in by hand;
nothing reads them unless the button is pressed.

- **Replace them freely.** Drop in real photographs under the same three names
  and the buttons pick them up — the page checks for each file and only shows the
  button when it is there, so there is no list to keep in step.
- **What makes a sample usable:** the three views must be the right way up, and
  they have to pass the pose gate (8% clear top and bottom, body filling at
  least 70% of the frame, centred, A-pose with the arms 20-45° out). A sample
  that fails shows the reasons, exactly as a real capture does.
- The committed ones are renders of the GarmentCode reference body, not
  photographs of anyone, so the repository carries no image of a real person.

Uploads, samples and live frames are all sent as already upright: the browser
asks the camera for a portrait stream, so a photograph and a live frame arrive the
same way and the server turns neither. `camera.rotation` in `config/app.yaml` is
only the fallback for a camera that ignores the portrait request, and
`tools/calibrate_lens_on_monitor.py` still needs it because it grabs its own
frames.

## Capturing from another machine

The camera is opened by the browser, so the station PC browsing to
`http://localhost:8000` needs nothing extra. Browsing to the station by IP from
another machine does: browsers refuse `getUserMedia` on a plain-http origin that
is not localhost, and the *Start camera* button will report the refusal. Three
ways round it, cheapest first:

1. **Upload instead.** Every capture button has an upload beside it, so photos
   taken on a phone work without any of this.
2. **A tunnel.** `ssh -L 8000:localhost:8000 user@station` makes the station
   look like localhost to the other machine, and nothing changes in the app.
3. **HTTPS.** Generate a certificate and pass it to uvicorn:

   ```bash
   openssl req -x509 -newkey rsa:2048 -nodes -days 365 \n     -keyout certs/key.pem -out certs/cert.pem -subj "/CN=station"
   ```

   then add `--ssl-keyfile certs/key.pem --ssl-certfile certs/cert.pem` to the
   `command:` in `docker-compose.yml`. A self-signed certificate still makes the
   browser warn once; accepting it is enough for the camera to work.

## The thirteen visual stages

Every view keeps each step of its own processing as a file, listed in
`manifest.json` with the algorithm, the parameters and the earlier files it came
from. Nothing is overwritten and nothing is hidden once it is done: the raw
capture in particular is kept exactly as it arrived.

```text
data/sessions/<id>/views/<view>/
  00_raw.jpg                 what the camera or the upload gave us
  01_oriented.png            rotated upright; a side view mirrored to face +x
  02_undistorted.png         undistorted, or marked "not applied"
  03_enhanced.png            the image the vision models actually saw
  04_detection.png           the chosen person's box, class and confidence
  05_roi.png                 that person's region, cropped
  06_mask_raw.png            segmentation output, before cleanup
  07_mask_clean.png          after opening, closing and hole treatment
  08_body_isolated.png       the body on a neutral background
  09_edges.png               the contour the geometry functions read
  10_keypoints.png           named COCO-17 keypoints with confidence
  11_landmarks.png           anatomical points and measurement rows
  12_measurement_overlay.png the span measured on each row, in centimetres
```

`/process/<session id>` steps through them with a filmstrip, a comparison viewer
(opacity blend or side by side against the raw capture), zoom and fit, and a
panel giving each stage's runtime, dimensions and parameters. `/review/<session
id>` shows the same viewer beside the measurements. Both read the stage list from
the API, so the pipeline and the interface cannot drift apart.

The enhancement at stage 03 is deliberately restrained — grey-world white
balance, CLAHE on lightness only, then a bilateral filter — and a test pins that
it moves no more than 1% of the silhouette's pixels. It changes how a pixel
looks, never where the body boundary is.

## How a circumference is computed

The four torso circumferences follow Montazerian's two-equation rule
(brief 12.1), not a single formula:

1. The **front breadth** `W` is the mask run at the accepted row of the front view.
2. The **side depth** `D` is the mask run at the same *height above the floor* in
   the side view — matched in millimetres, never by pixel row.
3. The semi-axes are `a = max(W, D) / 2` and `b = min(W, D) / 2`.
4. While `a <= 3b` the perimeter uses `ellipse_equation_1`
   (`pi * [3(a+b) - sqrt((3a+b)(a+3b))]`); beyond it, the elongated-ellipse
   equation `ellipse_equation_2`. Both are exact for a circle and both are tested
   against numerical integration of the true perimeter.

Every circumference keeps `front_breadth_mm`, `side_depth_mm`, its front/back
split, the row height, the chosen equation and each contributing view's
`mm_per_px`, so a number in the table can be recomputed from its own provenance
record. A test does exactly that.

> **Which two equations Montazerian used is the one thing here taken on
> judgement.** The brief specifies the selection rule (`a <= 3b`) but not the
> formulae, so the pair above was chosen because it satisfies the rule's intent —
> an approximation that is excellent for a human torso section, and a second one
> that holds where the first drifts on a flat section — and because both are
> exact for a circle and verifiable against integration. If Montazerian's pair
> differs, change the two functions in `geometry.py`; the identifiers in every
> provenance record say which equation produced which number.

The front/back arcs (`back_width`, `waist_back_width`, `hip_back_width`) use a
different model — two half ellipses split at the reviewed balance line — and are
labelled in the report as a method-specific approximation.

## Review procedure

After processing, open `/review/<session id>`.

- Each view shows the photo with the mask outline, the keypoints and every
  landmark. Drag a point, or drag a slice line to change its height; releasing
  it recomputes and the table updates. A moved landmark turns magenta and is
  recorded with `auto: false`.
- **`head_row` and `floor_row` set that view's scale.** If loose hair or a cap
  made the body measure short, drag the head row down to the top of the skull
  and every measurement from that view follows.
- The results are shown three times over, for three different readers: the
  **Montazerian core** panel, with each circumference's breadth, depth, row and
  equation and each length's left, right and accepted mean; the **pattern-ready**
  panel grouped by torso, arms and lower body; and the full table, filterable by
  core, torso, arms, lower body, mesh, warnings or missing. Selecting a row opens
  its provenance: inputs, method, formula, views, confidence, and the fact that
  nothing is validated against a manual measurement.
- A missing value is `null` with a reason, never zero and never a population
  average. `bust_points` and `bum_points` are not visible in a silhouette at all,
  so they start as `estimated_pending_acceptance` from the GarmentCode mean body
  and **block a pattern-ready export until a person accepts or replaces them**.
- Typing a value over a measurement requires a note. It is then recorded as
  `manual_input` with `source: operator_input`, and both the value and the note
  go into the audit trail and the report.
- `Undo`, `Redo` and `Reset to automatic` are available during the review
  session. Reset throws away every landmark edit and every typed value.
- Typed values survive later landmark drags; a fresh capture of a view clears
  the stored landmarks, because they no longer describe the photo.
- *Delete session* removes the folder and every photo in it, after confirming.

## Exports and the export gate

Three files, written to the session folder and served from
`/api/sessions/<id>/export/...`:

| File | What it is | When it is written |
|---|---|---|
| `body_measurements.yaml` | the 26 primary GarmentCode keys and nothing else, in centimetres, with the pinned commit in its metadata | only when the export gate passes |
| `extended_measurements.yaml` | everything else the pipeline measured, so no unsupported key can reach the pattern file | always |
| `measurement_report.yaml` | models, thresholds, lens calibration, scale method, every provenance record, every operator edit, every artifact path, every warning | always, including for a partial or failed session |

`GET /api/sessions/<id>/export/csv` gives one row per measurement with its
status, source, method, formula, views and warnings.

The **export gate** blocks the pattern-ready file when a field the chosen garment
profile needs is `null`, `unavailable`, `invalidated`, or an estimate nobody has
accepted. A blocked export also deletes any `body_measurements.yaml` from an
earlier run, so a pattern engine can never pick up stale numbers. The profiles
are in `config/app.yaml`: `core_upper_body`, `top_shirt`, `skirt`, `trousers`,
`full`.

**The GarmentCode revision is pinned in `config/app.yaml`, and the commit there is
a placeholder.** Nothing has been checked against a real revision, because
`pygarment` is not installed here. `python tools/verify_garmentcode.py <body
file>` compares the exported keys with the keys the installed revision actually
reads and prints the exact difference; with no `pygarment` it exits 2 and says
the check could not be made, which is not the same as the check passing. Nothing
renames a key to make it pass.

## Model weights

The two YOLO26 files are git-ignored, so a fresh clone has none. **The
application never downloads anything**: a station is provisioned once, by hand,
and from then on startup only verifies the files it was told about.

```bash
python tools/provision_models.py --fetch          # download the missing files
python tools/provision_models.py --write-hashes   # print the config block to paste
```

Paste the printed `seg_sha256` and `pose_sha256` into `config/app.yaml`. While
they are `null` the weights are used as found; once set they are enforced, and a
weight file that does not match the configured hash refuses to load rather than
quietly measuring with something else. `GET /api/health` reports whether the
files were found, and the capture page shows it. Both models load once at
startup, not per request.

In Docker, `docker/entrypoint.sh` still fetches missing files into the
bind-mounted checkout on first start; a checkout that already carries its weights
is left alone and never reaches the network.

## Data protection

- **Pseudonymous by construction.** A session folder is named by a UUID4, never
  by the subject. Only the tailor's client code is stored, and only in the
  index - never in a path, a URL or an access log.
- **Consent is recorded, not assumed.** The server refuses to open a session
  without it, and stores the time and the version of the consent text shown.
- **Retention.** Photos, masks and keypoints are deleted
  `retention_days_images` after a session is confirmed, and `unconfirmed_days`
  after creation if it never was. `landmarks.json` and both YAML files survive,
  so the record stays usable without the images. The purge runs on startup when
  `purge_on_startup` is set; set the periods to match the client's consent form.
- **Deletion is logged without personal data.** `deletions` records the session
  id, time, scope and reason - nothing about the person.
- **Encrypt the station disk.** BitLocker on Windows. Nothing here encrypts the
  files themselves.
- **No network.** No sync, cloud backup or telemetry. Backup means copying the
  whole `data/` folder to encrypted media.
- `data/` and `config/calibration.yaml` are git-ignored.

## Storage

Files hold the images and the YAML the pipeline consumes; a single SQLite
database at `data/app.db` holds the index, consent and audit trail, and never an
image or a YAML body (brief 10). Every database access and every session-folder
deletion goes through [`src/bodymeasure/store.py`](src/bodymeasure/store.py).

| Table | Holds |
|---|---|
| `sessions` | id, subject code, sex, measured height, consent time and version, status, views, purge date |
| `edits` | every landmark move and typed value, with the old and new value |
| `deletions` | what was deleted, when, and why - no personal data |

Status runs `capturing` -> `measured` -> `confirmed` -> `exported`. Only
confirmed sessions reach `GET /api/export.csv`, the one-row-per-session file for
Objective 2. Every file is written to `<name>.tmp` and renamed into place, so an
interrupted write leaves the previous file intact.

## Where this stops

Implemented: the two-view core mode (front plus one or two sides), the thirteen
visual stages with their manifest, the Montazerian circumferences with their
intermediates, shoulder and arm length per side, the extended-2D fields the
existing silhouette geometry already supported, operator review with
dependency-based recomputation, the provenance contract, the three exports and
the export gate.

**Not implemented, and each needs its own milestone approval:**

- **The back view and four-view mode** (brief 4.2, 11.3). `VIEWS` is front and
  the two sides; there is no `gate_back` and no back-view landmark search.
- **The multi-view mesh** (brief 13, M5). There is no `reconstruct.py` and no
  `measure_mesh.py`. Nothing claims `mesh_visual_hull` as a source; the review
  page's mesh filter exists and will simply match nothing.
- **Most of the additional research measurements** (brief 3.3) — neck-base
  circumference, across-front and across-back, the three crotch lengths, inseam
  and outseam, knee, calf and ankle, the bust-point geometry. The ones the
  current silhouette geometry supports are exported; the rest are absent rather
  than estimated.
- **`reference_target` scale mode** (brief 4.4). Only measured stature is
  offered, and the selector shows only that.
- **`design_parameters.yaml`** (brief 3.4). No ease, allowance or style
  parameter is stored anywhere, which is the point: this code does not touch
  garment design inputs.
- **Retention hardening** (brief M7). Retention and idempotent deletion work and
  are tested, but the brief's hardening pass has not been done.
- **A performance baseline on the target CPU** (brief 20.5). Per-stage runtimes
  are recorded in the manifest for every real capture, so the baseline is
  collectable, but no numbers have been taken on a Core i5-class machine and no
  threshold has been invented.

## Validation plan

There is no ground truth here yet, so verification is split three ways, and only
the first layer is done:

1. **Mathematical** (done). Each ellipse equation is exact for a circle and
   within 0.5% of numerical integration of the true perimeter; widths, depths,
   path lengths and angles are checked on synthetic masks with known answers;
   measurements are invariant to image offset and proportional resize when the
   scale follows.
2. **Known objects** (not done). Calibrated rectangular, circular, cylindrical
   and elliptical objects, plus a dress form with independently known dimensions.
   Object results will not be generalised to human accuracy.
3. **Repeatability** (not done). At least three complete recaptures per subject
   with repositioning, reported as mean, standard deviation, range and
   coefficient of variation. Repeatability is precision, not accuracy.

Human validation comes later, with trained measurers, repeated readings, agreed
anatomical definitions and an agreement analysis. Until its collection protocol
is recorded, no manual reference value will be added to this dataset, and
`validated_against_manual` stays `false` in every record.

## Known limitations

- **The height is a measurement, and everything inherits its error.** The engine
  is scale-free in the sense that it never needs a physical calibration, but it
  is only as good as the number typed in: a centimetre off scales the whole body
  by `1 / height`. It is also the one input a dishonest or careless operator
  could choose freely.
- **Hair sets the head row.** Nothing in a silhouette says where the skull ends.
  The capture instructions say to tie it flat or wear a tight cap, and the review
  page makes `head_row` draggable; a subject with volume left in will otherwise
  measure short.
- **Accuracy.** Against the GarmentCode reference bodies, bust, underbust and
  waist land within about 3 cm, but `crotch_hip_diff`, `waist_line`,
  `arm_length` and the shoulder measurements are well out. The causes are
  written up in [bench/README.md](bench/README.md); the short version is that
  the maximum-curvature landmark search is not selective on a smooth body, and
  the crotch row a silhouette shows is not the anatomical one.
- **No real capture has been measured.** Every test uses drawn or rendered
  figures. The pose-gate thresholds have never met a real A-pose.
- Webcam capture works through the container, because the *browser* opens the
  camera, not the server: open `http://localhost:8000` on the machine with the
  webcam. Reaching the station from another machine over the LAN does not work,
  though - `getUserMedia` is refused on a plain-http origin that is not
  localhost. See *Capturing from another machine* below.
- The first start of a checkout with no weights needs the network to fetch them;
  after that they are on the host and nothing reaches out again.
- `bust_points` and `bum_points` are not visible in a silhouette at all. They
  start from the GarmentCode ratios and should be entered by tape.
