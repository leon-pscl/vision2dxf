# Instructions for the Coding Agent

## Project summary

Build a web app prototype for made-to-measure garment pattern generation from body photos. The pipeline has 10 steps:

1. Garment type
2. Capture
3. Segmentation
4. Calibration
5. Landmarks
6. Measurement (6a: direct regression, 6b: body-model fit)
7. Validation and YAML
8. Drafting
9. Production details
10. Validation (draping and toile)

The work has two phases:

- **Phase 1:** a clickable UI with MOCK adapters behind stable interfaces.
- **Phase 2:** replace the mocks for steps 2–5 with real pretrained models.

All other steps stay mocked for now.

If Phase 1 already exists in the repo, verify it against this document and update the README.md
Only proceed to phase 2 once the project owner instructs to do so AND the pretrained models are already present/ready to be imported.
### Core rule

Every model sits behind an adapter that implements a fixed contract. Swapping a model must take exactly one edit, to `adapters.yaml`.

Never change `contracts.py` to fit a model. If a real model cannot satisfy a contract, stop and report the mismatch.

---

## Global constraints

- **User images are never written to disk.** Keep them in memory only. Evaluation scripts may write outputs, because they use a separate test set.
- **Mock outputs are clearly labelled.** Every result carries `is_mock: bool`. The UI shows a persistent yellow banner, "MOCK DATA – not real measurements", whenever any result in the session is a mock.
- **Thresholds and settings live in config, not in code.** Use `config.yaml`.
- **Device is configurable.** Set `device: cpu | cuda` in `config.yaml`. The default is `cpu`.
- **This is an academic project.** Record every model's licence in `LICENSES_MODELS.md`.

---

## Stack and repository layout

- Backend: Python 3.11, FastAPI, Pydantic v2, pytest.
- Frontend: React, Vite, TypeScript, and plain CSS or Tailwind.
- No database. Session state lives in memory, in a dict keyed by `session_id`.

```
/backend
  main.py
  contracts.py
  registry.py
  adapters.yaml
  config.yaml
  adapters/base.py
  adapters/mock/...
  adapters/real/...
  tests/
/frontend
/scripts
  download_weights.py
  eval_segmentation.py
  compare_silhouettes.py
/weights            (git-ignored)
README.md
LICENSES_MODELS.md
```

---

## Contracts (`backend/contracts.py`)

- All lengths are in cm.
- Every result includes `is_mock: bool` and `warnings: list[str]`.
- Images are passed as base64 strings.

|Step|Model|Fields|
|---|---|---|
|1|`GarmentSpec`|`garment_type` (enum: shirt, trousers, skirt, dress); `required_measurements: list[str]`; `ease_cm: dict[str, float]`; `design_rules: dict`|
|2|`CaptureInput`|`front_image`; `side_image`; `height_cm`; `weight_kg` (optional)|
|2|`CaptureResult`|`distance_ok`; `pose_ok`; `full_body_visible`; `clothing_ok`; `guidance: list[str]`|
|3|`SegmentationInput`|`image`; optional `prompts`: `points: list[(x, y)]`, `box: (x1, y1, x2, y2)`, `text: str`|
|3|`SegmentationResult`|`mask_png` (base64); `boundary_quality: float` (0–1); `model_name`|
|4|`CalibrationResult`|`method` (enum: height, reference_object); `px_per_cm` per view; `relative_uncertainty`|
|5|`LandmarkResult`|`points: dict[name, {x, y, confidence, source: "model" \| "heuristic" \| "user"}]`; `confirmed_by_user: bool`|
|6|`MeasurementResult`|for each measurement: `{value_cm, lower_cm, upper_cm, method}`|
|7|`ValidationResult`|`flags: list[{measurement, rule, severity}]`; `yaml: str`|
|8|`DraftResult`|`route` (enum: block, garmentcode); `pieces: list[{name, svg}]`|
|9|`ProductionResult`|pieces with seam allowance and notches, as SVG; `export_formats: ["svg"]`|
|10|`DrapeResult`|`placeholder_image`; `toile_checklist: list[str]`|

**Steps 6a and 6b must return the identical `MeasurementResult`, including the interval fields.** Step 7 will later compute conformal intervals for either backend. Changing the result type later would mean reworking the UI.

---

## Adapters and registry

- `adapters/base.py` defines one `typing.Protocol` per step. For example:
    
    ```python
    class Segmenter(Protocol):    def run(self, inp: SegmentationInput) -> SegmentationResult: ...
    ```
    
- `adapters.yaml` maps each step to a dotted class path. For example:
    
    ```yaml
    segmentation: adapters.real.seg_sam2.Sam2Segmenter
    ```
    
- `registry.py` loads these classes at startup.
- Step 6 has two backends, `regression` and `body_model`. The user selects one in the UI.
- Mocks are deterministic: they use seeded randomness, so the same input always gives the same output. They set `is_mock=True`.

---

## API

|Method|Path|Behaviour|
|---|---|---|
|POST|`/api/session`|Returns a new `session_id`|
|POST|`/api/session/{id}/step/{n}`|Runs the adapter for step n, stores the result, returns it|
|GET|`/api/session/{id}`|Returns the full session state|
|GET|`/api/session/{id}/export.yaml`|Returns the step 7 YAML|

---

## Phase 1 – UI shell with mocks (time-box: 1 hour)

A 10-step wizard with a left-hand stepper. Each step shows a status: pending, done, or warning.

- **Step 1:** garment type selector. Show the measurements the selected garment requires.
- **Step 2:** front and side image upload, plus an optional webcam. Inputs for height and weight. A pass/fail checklist for distance, A-pose, full body visible and clothing, each with guidance text.
- **Step 3:** the image with the mask overlaid at 50% opacity. Show `boundary_quality` and the model name.
- **Step 4:** the method used, `px_per_cm` per view, and the uncertainty.
- **Step 5 (highest UI priority):** a canvas showing the image with draggable landmark points.
    - Colour each point by its source: model, heuristic or user.
    - A dragged point becomes `source="user"`.
    - A "Confirm landmarks" button sets `confirmed_by_user`.
    - Steps 6 and later stay locked until the landmarks are confirmed.
- **Step 6:** a toggle between Regression and Body model. Below it, a table of value and interval for each measurement, with the interval drawn as a horizontal bar.
- **Step 7:** a list of flags, a YAML preview, and a download button.
- **Steps 8–9:** the SVG pattern pieces in a grid. Step 8 has a route toggle: Block or GarmentCode.
- **Step 10:** a placeholder image and the toile checklist with tick boxes.

**Order of work.** If time runs short, cut from the bottom of this list:

1. Contracts, Protocols, registry and `adapters.yaml`.
2. Mocks, API, and contract tests in pytest.
3. Wizard shell with steps 1, 2, 6 and 7.
4. Step 5 landmark canvas. If dragging is behind schedule, fall back to click-to-reposition. Do not drop the step.
5. Steps 3, 4, 8, 9 and 10.
6. README section: "How to plug in a real model".

**Phase 1 is accepted when:**

- Backend and frontend start cleanly, and a user can click through all 10 steps.
- pytest passes.
- Replacing one mock class path in `adapters.yaml` with a second dummy class works with no other edits.
- No image file appears on disk after a full run.

---

## Phase 2 – Real pretrained adapters for steps 2–5

### Adapters to implement (`backend/adapters/real/`)

1. **`pose_rtmpose.py`.** RTMPose through an ONNX runtime.
    - Write COCO keypoints with confidences into `LandmarkResult` (`source="model"`).
    - Also implement the step 2 pose check:
        - A-pose: shoulder–elbow–wrist angles fall within ranges set in config.
        - Full body visible: nose and both ankles are detected above a confidence threshold.
        - Distance: the person's bounding-box height, as a fraction of image height, falls within a band set in config.
2. **`det_yolo.py`.** An Ultralytics YOLO person detector. Return the highest-confidence box for the "person" class.
3. **`seg_sam2.py`.** SAM 2, using the smallest checkpoint.
    - Prompt it with a box.
    - The config option `sam2_prompt_source: pose | yolo_box` selects where the box comes from. A box from pose keypoints gets padding set in config.
4. **`seg_sam3.py`.** SAM 3 with the text prompt `"person"`. If several instances come back, keep the highest-scoring one that overlaps the detector or pose box.
5. **`seg_yolo.py`.** Ultralytics YOLO segmentation, using the smallest `-seg` checkpoint.
    - Use the "person" class only and keep the highest-confidence instance.
    - Expose `retina_masks: true | false` in config.
6. **`calib_height.py`.**
    - Per view: `px_per_cm = (lowest mask row − top mask row) / height_cm`.
    - `relative_uncertainty` comes from a pixel-error value set in config.
7. **`landmarks_contour.py`.** Heuristic tailoring landmarks derived from the mask and keypoints, with `source="heuristic"`.
    - Waist: the narrowest front-mask width between the hip and shoulder keypoints.
    - Hip: the widest front-mask width below the waist and above mid-thigh.
    - Bust or chest: the widest front-mask width between the shoulders and the waist.
    - The search bands for these are set in config.

### Engineering rules

- Load each model lazily, once per process.
- `scripts/download_weights.py` downloads weights into `/weights` and verifies checksums. `/weights` is git-ignored.
- Log inference time for every adapter call.
- Keep all mocks selectable. Do not delete them.

### Evaluation harness: `scripts/eval_segmentation.py`

Inputs:

- A folder of test images (front and side).
- Hand-corrected ground-truth masks.
- Height in cm for each subject, so pixel errors can be converted to centimetres.

Evaluate these variants, each as its own row:

- `sam2 + pose_box`
- `sam2 + yolo_box`
- `sam3_text`
- `yolo_seg`
- `yolo_seg_retina`

Metrics for each variant:

- Mask IoU.
- Boundary IoU.
- Mean and 95th-percentile boundary error, in px and in cm.
- Runtime per image on the configured device.

Outputs:

- A CSV of the metrics.
- A PNG grid for each image: original | ground truth | each variant's mask.

### Silhouette compatibility check: `scripts/compare_silhouettes.py`

Inputs:

- Masks produced by a chosen segmenter.
- A sample of the training silhouettes used by the measurement regressor.

Steps and outputs:

- Normalise both sets to the regressor's input size.
- Report an edge-sharpness statistic, the aspect-ratio distribution, and the fraction of mask area above the head keypoint (to show whether hair is included).
- Produce a side-by-side PNG grid.

**Do not run or modify the regressor in this phase.**

### Licences: `LICENSES_MODELS.md`

Record one row per model: model, version, licence, and URL. Cover at least RTMPose, SAM 2, SAM 3 and Ultralytics YOLO.

Then add a section to `README.md` that states:

- Ultralytics YOLO is AGPL-3.0.
- The project is for academic use.
- If the app is served over a network, the full source of the application must be made available under AGPL-compatible terms.

### Tests

- pytest checks that each real adapter's output validates against its contract on 3 sample images.
- Switching `adapters.yaml` between mock and real requires no other edits.
- An end-to-end test runs steps 1–5 with real adapters and 6–10 with mocks. It confirms that no user image is written to disk.

**Phase 2 is accepted when:**

- All tests pass.
- Both scripts run on the sample test set and produce their CSV and PNG outputs.
- `LICENSES_MODELS.md` is complete.
- The UI shows real masks and landmarks for steps 3–5, and the MOCK banner remains for steps 6–10.

---

## Trained-model artifacts (loader for models trained outside this repo)

The project owner trains the step 6a regressors (V-HW, V-H) in a separate notebook. They hand them to this app as model packages. This repo **never trains anything**; it only loads and serves models.

### Model package format

Each package is a folder under `/models/<model_name>/<version>/`, git-ignored, containing:

|File|Purpose|
|---|---|
|`model.pkl` (or `.joblib`, `.onnx`, `.skops`)|The inference model|
|`manifest.json`|See the schema below|
|`golden_inputs.npz`|10–20 input samples, already preprocessed exactly as in the notebook|
|`golden_outputs.npz`|The notebook's predictions for those inputs|
|`requirements.lock`|Exact library versions used to save the model|

The `manifest.json` schema is defined as a Pydantic model in `contracts.py`. Its fields:

- `name`
- `version`
- `step` (e.g. `"6a"`)
- `format` (enum: `pickle`, `joblib`, `onnx`, `skops`)
- `sha256` of the model file
- `input_spec`: `{views: ["front", "side"], image_size: [h, w], tabular: ["height_cm", "weight_kg"], feature_order: [...], units}`
- `preprocessing`: the name and version of the function in the shared `core` package
- `output_spec`: `{measurements: [...], units: "cm"}`
- `conformal`: `{method: "split", alpha, quantiles: {measurement: q_hat}}`, or `null` if not yet calibrated
- `training_data`: a short description
- `metrics`: `{measurement: {mae_cm, ...}}`
- `library_versions`: `{python, sklearn, torch, numpy, ...}`

### Shared preprocessing package: `/core`

- Create a small installable package, `core/`, with `pyproject.toml`. It holds the silhouette normalisation, resizing, feature construction, and any custom model classes.
- The app imports preprocessing **only** from `core`. The project owner's notebook will `pip install -e` the same package.
- Any class that ends up inside a pickle must be defined in `core`. A class defined in the notebook gets pickled under the notebook's `__main__` module and will fail to load in the app.

### Loader: `backend/adapters/real/measure_regression.py`

1. Read `manifest.json` and validate it against the Pydantic model.
2. Verify the model file's SHA-256 against the manifest. **Refuse to load on a mismatch.**
3. Compare `library_versions` with the installed versions:
    - Different major or minor versions: log a warning and add it to `warnings`.
    - Mismatched versions for sklearn pickles: refuse to load unless `allow_version_mismatch: true` is set in config.
4. Load by format:
    - ONNX: onnxruntime.
    - skops: `skops.io.load` with trusted types listed in config.
    - pickle and joblib: allowed only from `/models`. Never load these from user uploads or from URLs.
5. **Golden-parity check at startup.** Run the model on `golden_inputs.npz` and compare with `golden_outputs.npz`.
    - Tolerance `parity_atol_cm` is set in config; default 0.01 cm.
    - On failure, mark the adapter as unavailable, fall back to the mock, and show the reason in the UI.
6. Prediction: apply `core` preprocessing, run the model, then fill the intervals:
    - If `conformal` is present: `[value − q_hat, value + q_hat]` for each measurement.
    - Otherwise: lower and upper are `null`, and add the warning "uncalibrated: no interval".

### Model selection

- `adapters.yaml` gets a key `measurement_regression_model: <model_name>/<version>`.
- `GET /api/models` lists the installed packages with their manifest summaries and parity status.
- Step 6 of the UI shows the model name, version, parity status, and whether intervals are calibrated.

### Notebook export helper: `scripts/export_template.py`

Write a reference function, `export_model_package(model, X_golden, out_dir, manifest_fields)`, for the project owner to copy into the notebook. It must:

- save the model;
- compute the SHA-256;
- run predictions on `X_golden` and save the golden inputs and outputs;
- write `requirements.lock` with `pip freeze`;
- write `manifest.json`.

### Tests

- A package with a tampered model file is refused (SHA-256 mismatch).
- A package whose golden outputs have been altered fails the parity check, and the app falls back to the mock.
- A valid dummy package loads, passes parity, and returns a `MeasurementResult` with intervals.

---

## Reporting back

When each phase is done, report:

- What was built and what was cut.
- Any contract mismatch with a real model.
- Runtime per adapter on the configured device.
- For Phase 2: the CSV from `eval_segmentation.py`.

**Do not choose the final segmenter yourself.** The selection is made by the project owner from the evaluation results.

## Extras
To make sure that the codebase runs on all devices, utilize containers via docker. 