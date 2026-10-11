# Body Measurement Demo (Objective 1.1)

Local web app that turns a front photo plus at least one side photo into
GarmentCode body parameters, using silhouette geometry only — no machine-learned
measurement model. Full specification: [docs/AGENT_BRIEF.md](docs/AGENT_BRIEF.md).

**Status: complete (M0-M6 and the M4b bench).** YOLO26 segmentation and pose, the
pose gate, landmark search, the 26 GarmentCode keys, the capture and review
pages, and the bench against the five GarmentCode reference bodies. The image
scale is the subject's measured height, so there is no station calibration and no
floor mark. Measured accuracy and its limits: [bench/README.md](bench/README.md).

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
| `config/app.yaml` | model paths, thresholds, camera index and rotation |
| `config/schema.yaml` | the 26 GarmentCode keys, their units, views and default confidence |
| `config/calibration.yaml` | optional lens model, written by the monitor tool; git-ignored |
| `src/bodymeasure/` | `api.py`, `session.py`, `landmarks.py`, `measure.py`, `geometry.py`, and the rest |
| `tools/` | the optional on-monitor lens calibration, a separate milestone |
| `web/` | plain HTML/CSS/JS, no build step |
| `web/samples/` | three stored captures the capture page offers as mock input |
| `tests/` | pytest; `synthetic/` holds generated silhouettes with known answers |
| `data/` | sessions and the index — git-ignored, never committed |

## Capture procedure

Open `/` and work down the page.

1. **Subject.** Enter the id, sex and **height**, tick consent, press *Start
   session*. No image is captured until consent is ticked; the server refuses
   otherwise, and it refuses a height outside 100-220 cm.
2. **Camera.** Press *Start camera*. The browser is asked for a portrait stream
   (1080x1920), so the camera is bolted in the right way up and needs no physical
   rotation. The yellow box is where the body must sit: 8% clear top and bottom,
   filling at least 70% of the frame, and within the middle third across.
3. **Captures.** On screen, and worth reading to the subject:
   **shoes off**; **hair tied flat or a tight cap**; **fitted clothing**, no thick
   layers; **A-pose**, arms 20-45 degrees from the body and not touching it.

   Press *Start capture* in section 2 and the run takes over: front, left side,
   right side in turn, each with a five-second countdown to get into the pose and
   a five-second one to hold still, and a shutter flash at the shot. The capture
   buttons in section 3 take a single view instead, counting down from three, and
   are what *Retake* leaves for redoing one view.

   The server segments each frame and runs the pose gate: a tick means the view is
   usable, otherwise the reasons appear and *Retake* clears it.
4. **Measure.** Enabled once the front and at least one side view pass. It
   writes both YAML files and offers them for download.

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

## Review procedure

After measuring, open `/review/<session id>`.

- Each view shows the photo with the mask outline, the keypoints and every
  landmark. Drag a point, or drag a slice line to change its height; releasing
  it recomputes and the table updates. A moved landmark turns magenta and is
  recorded with `auto: false`.
- **`head_row` and `floor_row` set that view's scale.** If loose hair or a cap
  made the body measure short, drag the head row down to the top of the skull
  and every measurement from that view follows.
- The table lists all 26 values. Rows marked `estimated` or `low` are
  highlighted: `bust_points` and `bum_points` are not visible in a silhouette at
  all and start from the GarmentCode ratios, so a tape measurement belongs
  there. Typing a value marks it `edited` with `source: tailor`, and the note
  goes into the report.
- Typed values survive later landmark drags; a fresh capture of a view clears
  the stored landmarks, because they no longer describe the photo.
- *Delete session* removes the folder and every photo in it, after confirming.

## Model weights

The two YOLO26 files are git-ignored, so a fresh clone has none. In Docker
there is nothing to do: `docker/entrypoint.sh` checks `models/` on every start
and fetches only the files that are missing, into the bind-mounted checkout, so
it runs once per machine and is offline from then on. A checkout that already
carries its own weights is left alone and never reaches the network.

Run it without Docker, fetch them by hand:

```bash
python -c "from ultralytics import YOLO; YOLO('yolo26s-seg.pt'); YOLO('yolo26s-pose.pt')"
mkdir -p models && mv yolo26s-*.pt models/
```

`config/app.yaml` points at `models/`, so fine-tuned weights drop in by replacing
the files. `GET /api/health` reports whether they were found, and the capture
page shows it. Both models load once at startup, not per request.

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
