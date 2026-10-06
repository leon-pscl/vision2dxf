FIX BRIEF — model_training_measurement.ipynb
Version: 2026-10-07 (supersedes all earlier fix instructions). The original agent
brief (2026-10-07) still governs everything not changed here.

====================================================================
0. WORKING PROTOCOL (read first)
====================================================================
You will NOT run this notebook. The user runs it on Kaggle. Therefore:

- Every "CHECK" below must be implemented as code inside the notebook: an
  `assert` with an informative message, plus a `print` of the evidence (the value
  being checked). A CHECK that exists only in your report does not count.
- Add two config.yaml keys:
    smoke_test: true      # 1 epoch, ≤50 subjects per split (fit/val/calib),
                          # 10 sensitivity subjects, ≤2 photos per subject
    run_final_eval: false # when false, Section 8 (Test-A/Test-B) and everything
                          # that depends on its results is skipped with a printed notice
  Every downstream cell must handle both settings (e.g. the model card writes
  "final evaluation not run" instead of failing).
- Review your code statically before delivery:
  * every name used is defined or imported in that cell's scope or an earlier cell;
  * every module written with %%writefile imports everything it uses;
  * tensor shapes are traced by hand through MeasurementNet.forward for both
    variants, with a shape comment on each line of forward();
  * no f-string contains dict or set literals inside {...};
  * no keyword argument is repeated in a call;
  * every DataFrame column accessed is created earlier, by name.
  If your environment allows, run `python -m py_compile` on each module and on the
  notebook's code cells (syntax only).
- Markdown must not contain claims that depend on run results. Interpretation
  cells either refer to printed values ("see the table above") or say what to look
  for ("if X, then Y"). Never state an observed outcome.

DELIVERABLES
1. The corrected notebook.
2. CHANGELOG.md: one row per fix ID → cell id(s) changed → the assertion/print
   that verifies it → what the user should see in the output when it passes.
3. A list of anything you could not implement, with the reason.

USER DECISIONS (defaults applied; the user may override before you start)
- U1 Scalar injection: concatenate height/weight to pooled image features (B1).
- U2 Conformal calibration unit: one randomly chosen photo pair per calibration
  subject (B5).
- U3 Pre-2021 references: keep as labelled "foundational-method exceptions" (D5).

====================================================================
PART A — CRASHES (fix in this order)
====================================================================
A1. B1 intercept (§4.2, bodym_baselines.py)
    fit() stores only Ridge.coef_; predict() returns Z @ coef with no intercept.
    Fix: store the fitted Ridge objects and predict with .predict(Z). Remove the
    stray line `self.scaler_ = Ridge`.
    CHECK: the existing "worse than train-mean" assertion passes.

A2. Backbone name (config.yaml, bodym_cnn.py)
    "mnasnet1_0_3_0" is not a torchvision model. Fix: backbone: mnasnet1_0.
    On failure, print the available mnasnet names.
    CHECK: build_backbone prints its weights source.

A3. Missing imports
    - bodym_training.py and bodym_sensitivity.py: `from tqdm.auto import tqdm`.
    - Notebook setup cell: `from torch.utils.data import DataLoader`.

A4. predict_index NameError (§6): the comprehension uses `use_weight`; the local
    is `use_w`. Fix the name.

A5. `.groupby("stratum").n()` (§8, model card): replace every occurrence with
    `.groupby("stratum")["n"].first()`.

A6. EXPORT_DIR undefined (§9): define `EXPORT_DIR = PATHS["export"]` at the start
    of §9.

A7. §9.2 copies src/preprocess.py, which does not exist. Delete that line.

A8. Missing §9.1 — weights and ONNX are never saved. Add §9.1 BEFORE §9.2 (with
    WHAT/WHY/OUTPUT/CHECK markdown):
    - torch.save each variant to export/model_vhw.pt and model_vh.pt containing
      {"model": state_dict, "backbone", "hidden", "dropout", "targets",
       "use_weight", "img_height", "img_width", "mask_threshold"}.
    - torch.onnx.export each variant (dynamic batch axis).
    - Verify with onnxruntime on 20 REAL validation tensors (not random).
    CHECK: max |onnx − torch| < cfg["onnx_atol"], printed per variant;
    REQUIRED_EXPORTS assertion passes.

A9. Model card f-string (§9.4): `{{'80%': ...}}` inside an f-string expression is
    a set containing a dict → TypeError. Build the coverage dict in a variable
    first, then interpolate json.dumps(coverage_dict, indent=2).

A10. Linter cell (§11): capture_output passed twice (SyntaxError at compile);
    `_sp` undefined. Use `subprocess`, pass capture_output once. If the .ipynb is
    not found in /kaggle/working, print "skipped".

====================================================================
PART B — SILENT CORRECTNESS ERRORS (invalidate results)
====================================================================
B1. Dead scalar path (bodym_cnn.py)
    scalar_proj and scalar_fuse are BOTH zero-initialised and applied in series,
    so both weight gradients are zero forever; only scalar_fuse.bias learns, which
    ignores the scalars. Both variants are effectively silhouette-only. The
    injection is also shape-incompatible (1280 channels at full resolution added
    to a 32-channel, half-resolution stem output).
    Fix (U1): remove spatial scalar injection. Concatenate normalised scalars to
    the pooled image features before the MLP head:
        feat = pool(features(mask))              # (B, 1280)
        z = cat([feat, scalars_normalised], 1)   # (B, 1280 + n_scalars)
        head(z)
    Keep the mean-reduced 1-channel stem. Remove the unused "ones" channel (update
    to_tensor, N_CHANNELS constants and preprocess.py accordingly) or justify it
    in DECISIONS.md. If a projection layer is kept anywhere, zero-initialise only
    its LAST layer.
    CHECKS (in code):
    - After the first optimiser step, assert the head's weights for the scalar
      inputs are not all zero.
    - After training, assert that +10% height_cm changes mean girth predictions
      (|Δ| > 0) for both variants; print the Δ per girth.
    - Delete the old "delta < 1e-6 at init" test.

B2. Aspect ratio transposed (config.yaml, to_tensor)
    img_height=480, img_width=640 squashes portrait silhouettes. Ruiz et al.
    (2022) use 640 (H) × 480 (W) per view.
    Fix: set the per-view size to preserve the native mask aspect ratio measured
    in §2 (if native is 4:3 portrait, use 640×480). Fix the forward() docstring
    ("each resized to H/2 x W" is wrong).
    CHECK: print native vs model aspect ratio; assert they match within 1%.

B3. B1 slice rows inverted (_row_for)
    Returns y1 − (1 − f)·span, so f=0 (floor) maps to the top row.
    Fix: row = y1 − f·span.
    CHECK: save and display artifacts/fig_b1_slices.png with every slice row
    overlaid and labelled on one front mask; assert the ankle row index > chest
    row index (ankle is lower in the image).

B4. B1 widths include arms / both legs
    Fix: for each slice row, use the connected foreground run containing the
    body's midline column (median x of the mask). For thigh and calf, measure one
    leg: the run nearest the midline on one side. Restrict each measurement's
    calibration to its own features (its perimeter, width, depth, mm_per_px,
    height_cm), or correct the docstring if you keep all features.
    CHECK: the B3 overlay figure also draws the measured runs.

B5. Evaluate and calibrate PER PHOTO PAIR, not per subject average
    predict_index/predict_split average all photos of a subject (≈3 per subject
    in train and Test-B, ≈19 in Test-A). Deployment predicts from ONE pair, so
    intervals calibrated on averaged predictions under-cover, and the Test-A vs
    Test-B gap is confounded by the averaging count.
    Fix:
    - Primary metrics, checkpoint selection and conformal calibration use
      per-photo-pair predictions against the subject's targets.
    - Subject-averaged metrics: secondary table only, clearly labelled.
    - Conformal (U2): compute calibration scores from ONE randomly chosen pair per
      calibration subject (seeded), so scores are exchangeable with single-pair
      deployment. Record in conformal.json: "unit": "photo_pair",
      "pairs_per_subject": 1.
    CHECK: print photos-per-subject per split next to each metric table; assert
    the conformal unit field exists.

B6. Jitter is dilation-only (bodym_perturb.py)
    `grown |= fg` never removes foreground; both np.where branches are identical.
    Fix: new_fg = (signed_distance − field·sigma) < 0, so the boundary moves both
    in and out.
    CHECK: on 50 masks, print mean and range of foreground-area change; assert
    |mean| < 0.5% and both signs occur.

B7. Circular sensitivity self-check (§5)
    ΔMAE at "none" is zero by construction. Replace with:
    - "none" MAE on the sensitivity subjects equals the per-pair B2 MAE on the
      same subjects computed independently (assert within 1e-6);
    - morph k=0 MAE equals "none" MAE (assert).

B8. Tolerable-error definition (max_tolerable_boundary_error)
    - Criterion = TOTAL error: baseline MAE + ΔMAE ≤ tolerance. Report this and the
      Δ-only figure, clearly labelled.
    - Contiguity: tolerable magnitude = largest |k| such that every magnitude from
      0 to k on that side passes. Report erosion (k<0) and dilation (k>0)
      separately.
    - Downsample: report the largest passing FACTOR; never convert it to mm.
    - If nothing passes at a tolerance, output "none tolerable" instead of 0.

B9. Sensitivity on the SHIPPED models
    The sweep ran on B2; the model card filters for V-HW and is empty.
    Fix: after §6, re-run the sweep for V-HW and V-H (same grid and subjects). The
    step-3 requirement in the model card comes from V-HW. The B2 sweep remains
    only as the source of augmentation magnitudes (it must precede training);
    state this in DECISIONS.md.
    CHECK: assert the model card's sensitivity table is non-empty and labelled V-HW.

B10. Augmentation magnitude rule (§5)
    Current rule picks the largest magnitude where ANY measurement passes; the
    markdown says ALL. Fix: require all measurements except `height` (input
    passthrough) to pass, with the B8 contiguity rule. Print the binding
    measurement.

B11. DataLoader worker RNG (training_augmenter)
    One default_rng is copied into all workers → identical augmentation streams.
    Fix: create the RNG lazily inside each worker, seeded from
    torch.utils.data.get_worker_info().id, the epoch and cfg seed.
    CHECK: draw samples via a 2-worker DataLoader; assert augmented masks from
    different workers differ.

B12. Smaller fixes
    - Conformal half-width: q is already the half-width → mm = q*10, not q*20
      (§7 tables, §11 summary).
    - §7 plausibility demo: use an actual V-HW prediction for the first
      validation pair, not the mean of the B0 feature vector.
    - §6: delete the `... or True` assertion; assert FLAGS has the expected columns
      and every flagged row truly fails vs B0.
    - Seeds: after loading config, set SEED = cfg["seed"] and call
      set_global_seeds(SEED) again.
    - §2 sex labels: .reindex([0, 1]) before renaming to female/male.
    - infer.py: if clamping pushes lo above or hi below the point estimate, keep
      the value inside the interval and add "clamped": true to that measurement's
      output.
    - Final message: the zip is in /kaggle/working, not /kaggle/outputs.

====================================================================
PART C — TRAINING BUDGET
====================================================================
C1. epochs_b2=6 and epochs_prod=8 are ~1% of the iterations Ruiz et al. (2022)
    used (150k iterations, batch 22). Keep epochs configurable; log validation MAE
    per epoch; print total optimiser steps per model in the §11 summary; label the
    B0-vs-vision flag table PROVISIONAL with the step count printed beside it.
C2. Add a cell that times 100 augmenter calls and prints ms/sample. If >50 ms,
    cache distance transforms or simplify, and log the decision in DECISIONS.md.
C3. Replace the hard-coded runtime table in §0.5 with "measured runtimes: see §11".

====================================================================
PART D — DOCUMENTATION INTEGRITY
====================================================================
D1. Remove claims the code does not support:
    - "ONNX checks already automated and passing": only true after A8; then refer
      to the printed result.
    - preprocess self-test "compares the full tensor": implement it. Save the
      reference input masks plus height/weight, recompute tensors in the
      self-test, and assert np.allclose(..., atol=1e-6) against saved tensors.
D2. Remove every hard-coded number from markdown and DECISIONS.md, including:
    "up to 45"/"up to 27" silhouettes per subject; "max waist 164 cm vs median
    88 cm"; "the single subject id shared between Test-A and Test-B"; "960x720";
    "<50 ms per subject"; "clean masks score below 0.05"; the runtime table.
    Replace with printed values or delete.
D3. Make the stem description consistent everywhere (module and class
    docstrings, §4.3 markdown, DECISIONS D5): mean-reduced 3→1 channel stem;
    scalars enter at the head (B1).
D4. Remove invented content:
    - DEFINITIONS: every entry becomes "BodyM measurement; vertex-path length on an
      SMPL-registered mesh (Ruiz et al., 2022). Exact path not published."
    - Do not call height_cm "self-reported"; say "height from hwg_metadata.csv
      (collection method not stated)".
D5. References (U3):
    - Correct Ramanujan to: Ramanujan, S. (1914). Modular equations and
      approximations to π. Quarterly Journal of Mathematics, 45, 350–372.
    - Drillis & Contini (1966): the Human Factors citation is likely wrong. Remove
      it and label the slice-height table [Assumption], unless you can give a
      verifiable source.
    - Put Bland & Altman (1986) and Ramanujan (1914) under a subsection
      "Foundational-method exceptions (outside the 2021–2026 open-access rule)".
    - Add a cell that regex-checks every in-text citation against references.md
      and prints mismatches in both directions.
D6. Display every saved figure inline as well as saving it.
D7. Add dated DECISIONS.md entries for: head-level scalar injection (B1),
    per-pair evaluation and single-pair calibration (B5), total-error budget
    (B8), sensitivity re-run on shipped models (B9), provisional training budget
    (C1).

====================================================================
PART E — RUN PLAN (executed by the user, not by you)
====================================================================
Write this into a markdown cell in §0.4:
E1. Smoke run: smoke_test=true, run_final_eval=false. Every assertion must pass.
E2. Full training run: smoke_test=false, run_final_eval=false. Review validation
    results, sensitivity tables and the B1/B5/B9 checks.
E3. Final run, once: run_final_eval=true, "Save Version → Save & Run All". No
    training code changes after this. If a later fix to Sections 1–7 is needed,
    log it in DECISIONS.md with a statement that Test-A/B results have been seen.

====================================================================
REFERENCES (for references.md)
====================================================================
Angelopoulos, A. N., & Bates, S. (2021). A gentle introduction to conformal
  prediction and distribution-free uncertainty quantification (arXiv:2107.07511).
  https://arxiv.org/abs/2107.07511
Pimentel, J. F., Murta, L., Braganholo, V., & Freire, J. (2021). Understanding
  and improving the quality and reproducibility of Jupyter notebooks. Empirical
  Software Engineering, 26, 65. https://doi.org/10.1007/s10664-021-09961-9
Ruiz, N., Bellver, M., Bolkart, T., Arora, A., Lin, M. C., Romero, J., & Bala, R.
  (2022). Human body measurement estimation with adversarial augmentation
  (arXiv:2210.05667). https://arxiv.org/abs/2210.05667
Foundational-method exceptions:
Bland, J. M., & Altman, D. G. (1986). Statistical methods for assessing agreement
  between two methods of clinical measurement. The Lancet, 1(8476), 307–310.
Ramanujan, S. (1914). Modular equations and approximations to π. Quarterly
  Journal of Mathematics, 45, 350–372.