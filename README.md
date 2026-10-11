# Body Measurement Demo (Objective 1.1)

Local web app that turns a front photo plus at least one side photo into
GarmentCode body parameters, using silhouette geometry only — no machine-learned
measurement model. Full specification: [docs/AGENT_BRIEF.md](docs/AGENT_BRIEF.md).

**Status: complete (M0-M6 and the M4b bench).** Calibration, YOLO26 segmentation
and pose, the pose gate, landmark search, the 26 GarmentCode keys, the capture
and review pages, and the bench against the five GarmentCode reference bodies.
Measured accuracy and its limits: [bench/README.md](bench/README.md).

## Run it (Docker — nothing else to install)

```bash
docker compose up --build        # http://localhost:8000
docker compose down
```

Rebuild with `--build` whenever `requirements.txt` or the Dockerfile changes:
`docker compose up` on its own reuses the old image and will keep serving stale
dependencies. The whole repository is bind-mounted and uvicorn runs with
`--reload`, so every edit applies live.

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

## Calibration procedure

Open `/calibrate.html` and work top to bottom. Enter the board geometry first:
the counts are **inner** corners, so a 10x7-square board is 9x6.

1. **Intrinsics.** Start the camera, then hold the checkerboard at 15-20
   different angles, distances and positions in the frame. Grab a frame at each.
   Vary the distance - boards all held at one depth leave the distortion terms
   badly conditioned. Press *Compute intrinsics* and check `rms_px`: below ~0.5
   is good, above ~1.0 means reshoot.
2. **Scale.** Stand the board upright on the floor mark, square to the camera.
   Measure from the board's lowest inner corners down to its bottom edge and
   enter that as the bottom margin. Press *Measure scale*, check `mm_per_px_x`
   and `mm_per_px_y` agree within about 1%, then *Save calibration*.

Step 1 parks its result in `data/calib/intrinsics.yaml`; step 2 combines the two
into `config/calibration.yaml`. Nothing is written until you press save. Either
step also accepts uploaded image files instead of the webcam.

## Layout

| Path | What it holds |
|---|---|
| `config/app.yaml` | model paths, thresholds, camera index and rotation |
| `config/schema.yaml` | the 26 GarmentCode keys, their units, views and default confidence |
| `config/calibration.yaml` | written by the calibration page (M2); git-ignored |
| `src/bodymeasure/` | `config.py`, `api.py` today; vision, geometry and measurement modules per milestone |
| `web/` | plain HTML/CSS/JS, no build step |
| `tests/` | pytest; `synthetic/` holds generated silhouettes with known answers (M1) |
| `data/` | sessions and calibration captures — git-ignored, never committed |

## Capture procedure

Open `/` and work down the page.

1. **Subject.** Enter the id and sex, tick consent, press *Start session*. No
   image is captured until consent is ticked; the server refuses otherwise.
2. **Camera.** Press *Start camera*. The preview is rotated to match the
   calibrated rotation, so what you see is what gets measured. The red line is
   the floor mark and the green outline is where the body should sit.
3. **Captures.** Tight clothing, A-pose with the arms 20-45 degrees from the
   body, standing on the floor mark. Each *Capture* button counts down from
   three. The server segments the frame and runs the pose gate: a tick means the
   view is usable, otherwise the reasons appear and *Retake* clears it.
4. **Measure.** Enabled once the front and at least one side view pass. It
   writes both YAML files and offers them for download.

Without a webcam, every step also accepts an uploaded image, so the demo runs
away from the station.

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
- The table lists all 26 values. Rows marked `estimated` or `low` are
  highlighted: `bust_points` and `bum_points` are not visible in a silhouette at
  all and start from the GarmentCode ratios, so a tape measurement belongs
  there. Typing a value marks it `edited` with `source: tailor`, and the note
  goes into the report.
- Typed values survive later landmark drags; a fresh capture of a view clears
  the stored landmarks, because they no longer describe the photo.
- *Delete session* removes the folder and every photo in it, after confirming.

## Model weights

The two YOLO26 files are git-ignored and not baked into the image. Fetch them
once, into `models/`:

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
| `sessions` | id, subject code, sex, consent time and version, status, views, purge date |
| `edits` | every landmark move and typed value, with the old and new value |
| `deletions` | what was deleted, when, and why - no personal data |

Status runs `capturing` -> `measured` -> `confirmed` -> `exported`. Only
confirmed sessions reach `GET /api/export.csv`, the one-row-per-session file for
Objective 2. Every file is written to `<name>.tmp` and renamed into place, so an
interrupted write leaves the previous file intact.

## Known limitations

- **Accuracy.** Against the GarmentCode reference bodies, bust, underbust and
  waist land within 3 cm and height within 4 mm, but `crotch_hip_diff`,
  `waist_line`, `arm_length` and the shoulder measurements are well out. The
  causes are written up in [bench/README.md](bench/README.md); the short version
  is that the maximum-curvature landmark search is not selective on a smooth
  body, and the crotch row a silhouette shows is not the anatomical one.
- **No real capture has been measured.** Every test uses drawn or rendered
  figures. The pose-gate thresholds have never met a real A-pose.
- Webcam capture works through the container, because the *browser* opens the
  camera, not the server: open `http://localhost:8000` on the machine with the
  webcam. Reaching the station from another machine over the LAN does not work,
  though - `getUserMedia` is refused on a plain-http origin that is not
  localhost. See *Capturing from another machine* below.
- Model weights are not baked into the image; `models/` is bind-mounted.
- `bust_points` and `bum_points` are not visible in a silhouette at all. They
  start from the GarmentCode ratios and should be entered by tape.
