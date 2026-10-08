AGENT BRIEF — BodyM measurement model (Kaggle notebook)
Version: 2026-10-07. This brief supersedes all earlier instructions.

====================================================================
0. PURPOSE AND CONTEXT
====================================================================
Build one Kaggle notebook that trains, evaluates, documents and exports a
body-measurement model for STEP 6 of a 10-step made-to-measure pattern pipeline:
  1 garment type → 2 capture (front A-pose + side) → 3 person segmentation →
  4 scale/perspective calibration → 5 landmarks → 6 MEASUREMENT ESTIMATION →
  7 schema validation + YAML → 8 pattern drafting → 9 production details/export →
  10 virtual drape + physical toile vs tape-measured ground truth.

The exported model receives a front silhouette, a side silhouette, height (cm),
and optionally weight (kg). It returns 14 body measurements (cm), each with a
prediction interval. Step 7 uses the intervals for plausibility gating; step 4
uses them in its error budget.

The notebook is also the permanent documentation of how the model was produced.

====================================================================
1. HARD CONSTRAINTS
====================================================================
- Data: BodyM only. It is already downloaded and attached under /kaggle/input/.
  BodyM contains binary silhouettes, height, weight, gender and 14 measurements.
  It has NO RGB images. Do not train or fine-tune any segmentation model here.
- Do not add Ultralytics/YOLO as a dependency.
- Licence: CC BY-NC 4.0. The AWS registry entry links to the CC BY legal code;
  record this inconsistency and treat the data as non-commercial.
- Test-A and Test-B are used exactly once, in the final evaluation section.
  No model selection, tuning, calibration or threshold choice may use them.
- Never claim ISO 20685 compliance. Report errors against tolerance thresholds only.
- Do not assume file or folder names. Inspect the attached data and write loaders
  from what you find.

====================================================================
2. PHASE 0 — SETUP AND DATA INTEGRITY
====================================================================
- Print environment: Python and package versions, GPU type, internet on/off.
- Set and log global seeds (Python, NumPy, PyTorch; deterministic flags where feasible).
- Load all hyperparameters from config.yaml (write it from the notebook if absent).
- Inspect the BodyM directory tree and CSV headers; print a summary.
- Verify counts against Ruiz et al. (2022):
    train 2,018 subjects / 6,134 silhouettes
    testA    87 subjects / 1,684 silhouettes
    testB   400 subjects / 1,160 silhouettes
  Raise a clear error on any mismatch.
- Verify that every photo maps to a subject, every subject has measurements and
  height/weight, and that front/side mask pairs exist.

====================================================================
3. PHASE 1 — EXPLORATORY DATA ANALYSIS
====================================================================
- Subjects and silhouettes per split; photos per subject.
- Sex distribution; BMI (computed from height/weight) in bands
  <18.5, 18.5–25, 25–30, 30–40, >40.
- Measurement distributions and their correlation matrix.
- Silhouette image sizes, person pixel-height distribution, and 10–20 example
  mask pairs, including visible segmentation artefacts.
- Write an interpretation: the population limits that bound generalisation
  (notably high-BMI scarcity).

====================================================================
4. PHASE 2 — SPLITS
====================================================================
- From TRAIN only, split BY SUBJECT ID into:
    fit (~80%), validation (~10%), conformal calibration (~10%).
  Stratify by sex and BMI band where feasible.
- Assert zero subject overlap between all splits.
- Save the split assignments to CSV for reproducibility.

====================================================================
5. PHASE 3 — BASELINES (in this order)
====================================================================
B0 Non-visual: ridge regression and gradient boosting predicting the 14
   measurements from height + weight + sex. This is the bar every vision model
   must beat.
B1 Geometric: per-slice ellipse model.
   - Scale = height_cm / silhouette pixel height.
   - Slice heights from proportional heuristics (document each).
   - Front width → one axis, side depth → the other; Ramanujan perimeter.
   - Per-measurement linear calibration fitted on the fit split only.
   - Measurements not expressible as slices: state how they are derived or
     mark them unsupported.
B2 Learned: BMnet-style CNN (Ruiz et al., 2022).
   - Front and side silhouettes, each 640×480, concatenated spatially to 640×960.
   - Constant-valued height and weight channels concatenated depth-wise.
   - Backbone: MNASNet (or ResNet if justified). ImageNet weights attached as a
     Kaggle Model/dataset if internet is off; document the source. If the first
     conv layer is adapted to the channel count, document how.
   - MLP head → 14 outputs. L1 loss. Checkpoint every epoch to /kaggle/working
     so a session timeout does not lose the run.
Report validation metrics (Section 9) for B0, B1 and B2 side by side.

====================================================================
6. PHASE 4 — BOUNDARY-SENSITIVITY EXPERIMENT (key deliverable)
====================================================================
Purpose: derive the boundary accuracy that step 3 (segmentation) must achieve.
On validation silhouettes, apply independently to front and side masks:
  - morphological erosion/dilation, k = −6 … +6 px;
  - random boundary jitter;
  - downsample → upsample at factors 2, 4 and 8 (simulates low-resolution
    prototype masks).
Re-run B1 and B2. Plot ΔMAE per measurement against perturbation size.
Output the maximum tolerable boundary error (px, and mm via the height-derived
scale) for chest, waist and hip at tolerance thresholds of 9, 15 and 25 mm.
Save the results to sensitivity.csv.

====================================================================
7. PHASE 5 — PRODUCTION MODEL
====================================================================
- Architecture: B2.
- Train two variants:
    V-HW: height + weight channels.
    V-H : height only. Remove the weight channel; do not zero-fill it.
- Augmentation, applied independently per view: erosion/dilation with
  magnitudes taken from the Phase 4 results, boundary jitter,
  downsample→upsample (×2, ×4), rotation ±3°, small scale/translation.
  Rationale: deployment masks come from a different segmenter than BodyM's
  DeepLabv3+-derived masks.
- Select the checkpoint on validation MAE only.
- For any measurement where the variant does not beat B0 on validation, flag it
  in the model card. Do not hide or drop it.

====================================================================
8. PHASE 6 — UNCERTAINTY (split conformal prediction)
====================================================================
- On the calibration split, compute absolute residuals per measurement per variant.
- Store the conformal quantiles for 80% and 90% target coverage
  (Angelopoulos & Bates, 2021).
- Document that the coverage guarantee assumes exchangeability; it weakens under
  segmenter, device or population shift. The intervals must be recalibrated on
  tape-measured data from step 10.

====================================================================
9. METRICS (used in Phases 3, 5 and 7)
====================================================================
Per measurement, per model/variant:
  - MAE (mm), mean bias (mm)
  - TP50 / TP75 / TP90 of absolute error (Ruiz et al., 2022)
  - Bland–Altman mean difference and 95% limits of agreement
  - % of predictions within 9, 15 and 25 mm
  - Conformal empirical coverage (80%, 90%) and mean interval width
Stratify by sex and BMI band; always print n per stratum; mark strata with
n < 30 as unreliable.

Add visualizations whenever necessary
====================================================================
10. PHASE 7 — FINAL EVALUATION (run once)
====================================================================
- Freeze the selected V-HW and V-H models and their conformal files.
- Evaluate on Test-A and Test-B with all Section 9 metrics.
- Present B0, B1, V-HW and V-H in one comparison table.
- Interpret Test-A versus Test-B. Test-B photos were taken in less controlled
  conditions, so the gap estimates sensitivity to capture conditions.

====================================================================
11. PHASE 8 — EXPORT (/kaggle/working/export/, then zip)
====================================================================
- model_vhw.onnx, model_vh.onnx, plus PyTorch state_dicts.
  Verify ONNX and PyTorch outputs match within 1e-4 on 20 samples.
- preprocess.py:
    silhouette_to_tensor(front_mask, side_mask, height_cm, weight_kg=None)
  reproduces the training preprocessing exactly (binarisation, cropping,
  aspect handling, resizing, channel construction). Include a unit test
  against saved reference tensors.
- infer.py: loads model + conformal file and selects the variant by whether
  weight is supplied. Returns:
    {measurement: {value_cm, lo80, hi80, lo90, hi90}}.
- conformal.json: per-measurement, per-variant quantiles.
- schema_map.yaml: BodyM measurement names → pattern-engine keys
  (leave TODO placeholders for the user to fill), units (cm), definition note
  ("SMPL-registered mesh vertex-path length; not ISO tape measurement"), and an
  explicit list of pattern keys NOT provided (e.g. bust points, underbust,
  armscye depth, crotch depth).
- model_card.md: training data, licence, input contract, intended use and
  out-of-scope use, all final metrics with strata, known failure regimes
  (high BMI), sensitivity results, measurements flagged as not beating B0.
- DECISIONS.md: dated decision log. Each entry: decision, alternatives,
  evidence ([Ref]/[Exp]/[Assumption]), consequences, open questions.
- README.md: how a downstream pipeline step loads and calls infer.py, with a
  minimal working example.
- references.md: full APA list of every citation used.
- requirements.txt (pinned), config.yaml, split CSVs, sensitivity.csv.

====================================================================
12. DOCUMENTATION REQUIREMENTS
====================================================================
Notebook structure
- Section 0 at the top: purpose, place in the 10-step pipeline, inputs, outputs,
  how to reproduce (Kaggle "Save Version → Save & Run All"), expected runtime,
  hardware, table of contents.
- One numbered section per phase above.
- Before every non-trivial code cell, a markdown cell with:
    WHAT   — what the cell does (1–3 sentences).
    WHY    — the rationale, tagged as exactly one of:
             [Ref] APA in-text citation (open-access, 2021–2026 preferred);
             [Exp] pointer to an experiment cell in this notebook;
             [Assumption] explicit unverified assumption + how it will be tested.
    OUTPUT — variables, files or figures produced.
    CHECK  — the assertion or sanity check that confirms success.
- After every results cell, an interpretation cell. Refer to printed variables;
  never hard-code numbers in markdown.

Explanations that must appear
- Subject-level splitting (multiple silhouettes per subject → leakage).
- Single use of Test-A/Test-B.
- Purpose of B0.
- Choice of augmentation magnitudes (linked to Phase 4).
- Why two variants are exported (height-only vs height+weight trade-off; see the
  input ablation in Ruiz et al., 2022).
- What conformal intervals guarantee and when that breaks.
- Mesh-defined vs tape-measured ground truth and its implication for step 10.
- Licence status and the registry inconsistency.

Code quality
- NumPy-style docstrings for every function (parameters, units, shapes,
  returns, raises).
- Type hints throughout; units in variable names (height_cm, weight_kg, err_mm).
- Reusable functions live in src/ (written from the notebook) and are imported.
- No hidden state: the notebook must pass "Restart & Run All".
- Optionally run a notebook linter (e.g. pynblint; Quaranta et al., 2022) and
  record the outcome. Treat it as advisory.

====================================================================
13. FINAL CELL
====================================================================
Print a reproducibility summary: package versions, seeds, data file counts,
GPU type, wall-clock time per phase, and paths of all exported files.

====================================================================
14. OPEN DECISIONS FOR THE USER (do not resolve silently)
====================================================================
- Whether the capture protocol (step 2) will collect weight, which determines
  the default variant.
- Pattern-engine key names for schema_map.yaml.
- Tolerance thresholds the pattern engine will accept per measurement.
Leave clearly marked TODOs and list them in DECISIONS.md.

====================================================================
REFERENCES (copy into references.md)
====================================================================
Angelopoulos, A. N., & Bates, S. (2021). A gentle introduction to conformal
  prediction and distribution-free uncertainty quantification
  (arXiv:2107.07511). https://arxiv.org/abs/2107.07511
Pimentel, J. F., Murta, L., Braganholo, V., & Freire, J. (2021). Understanding
  and improving the quality and reproducibility of Jupyter notebooks.
  Empirical Software Engineering, 26, 65.
  https://doi.org/10.1007/s10664-021-09961-9
Quaranta, L., Calefato, F., & Lanubile, F. (2022). Pynblint: A static analyzer
  for Python Jupyter notebooks. In Proceedings of CAIN '22.
  https://arxiv.org/abs/2205.11934
Ruiz, N., Bellver, M., Bolkart, T., Arora, A., Lin, M. C., Romero, J., &
  Bala, R. (2022). Human body measurement estimation with adversarial
  augmentation (arXiv:2210.05667). https://arxiv.org/abs/2210.05667