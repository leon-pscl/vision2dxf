# CHANGELOG — model_training_measurement.ipynb

Fix brief: `ipynb/fixbrief.md` (2026-10-07). Original brief: `ipynb/instructions.md` (2026-10-07).

Applied with `python patch_notebook.py`. The patch sources are in `patch_cells/`; the notebook goes
from 136 to 142 cells (51 replaced, 6 inserted, 1 deleted, 107 nbformat keys normalised).

Cell numbers below are the indices in the **patched** notebook. To map back, run
`python patch_notebook.py --dry-run`, which prints the original-index plan.

Every CHECK is implemented as an `assert` plus a `print` of the evidence in the notebook itself.

---

## Part A — crashes

| ID | Cells | Fix | Assertion / print | What you should see when it passes |
|----|-------|-----|-------------------|----------------------------------|
| A1 | 38, 44 | `B1Geometric` stores the fitted `Ridge` objects and predicts with `.predict(Z)`; the intercept is no longer dropped. Stray `self.scaler_ = Ridge` removed. | 44 asserts `all(np.isfinite(b1.intercept(c)))`; 44 asserts the "worse than train-mean" check passes; both print the intercept table. | `B1 calibration intercepts (cm):` followed by 13 rows, then `B1 measurements worse than a constant train-mean predictor: none`. |
| A2 | 8, 47, 50 | `backbone: "mnasnet1_0"`. `build_backbone` lists the available MNASNet names on failure. | 50 asserts `hasattr(_tvm, cfg["backbone"])` and `cfg["backbone"] in _avail_mnas`; prints the available list. | `available torchvision MNASNet models: ['mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3']`. |
| A3 | 3, 52, 62 | `from torch.utils.data import DataLoader` in cell 3; `from tqdm.auto import tqdm` in `bodym_training.py` and `bodym_sensitivity.py`. | 3 asserts `DataLoader is not None`; module import in 54 and 64 would raise otherwise. | `DataLoader bound: torch.utils.data.DataLoader` then `all required packages present`. |
| A4 | 62 | `sweep_model` uses the local `use_w`, not `use_weight`. | Sweep runs; `B2 sweep done: (N, 5)`. | The B2 sweep completing without a `NameError`. |
| A5 | 95, 117 | `.groupby("stratum").n()` → `.groupby("stratum")["n"].first()` (4 sites). Also `M.stratum_sizes()` helper added to `bodym_metrics.py`. | 95 prints per-stratum `n`; 117 builds the strata lines. | `n per stratum: {...}` in the stratified report; strata lines in `model_card.md`. |
| A6 | 97 | `EXPORT_DIR = PATHS["export"]` at the top of §9.1. | Every later cell uses it. | `EXPORT_DIR = /kaggle/working/export`. |
| A7 | 102 | Deleted `shutil.copy2(PATHS["src"] / "preprocess.py", ...)` — `src/preprocess.py` never existed. | — | The mirror cell completing without `FileNotFoundError`. |
| A8 | 97, 108, 125 | New §9.1 saves `model_{vhw,vh}.pt` with full architecture metadata and `model_{vhw,vh}.onnx` with a dynamic batch axis; verified against PyTorch on 20 **real** validation tensors. | Asserts `max \|onnx − torch\| < cfg["onnx_atol"]/10` per variant; asserts the checkpoint round-trips to 1e-6; 125 asserts all four files exist via `REQUIRED_EXPORTS`. | `max \|onnx - torch\| = 0.00e+00 cm (… ) over 20 samples` for both variants, then `all 16 required brief section 11 artefacts present`. |
| A9 | 117 | Model-card coverage dict hoisted out of the f-string; `json.dumps(coverage_dict, indent=2)` interpolated. | — | The model card writing (the old f-string was a **SyntaxError** at compile time, so this cell never ran). |
| A10 | 132 | `subprocess` instead of undefined `_sp`; `capture_output` passed once; missing `.ipynb` reports "skipped". | Wrapped in try/except; prints the report either way. | `=== advisory linter (pynblint, Quaranta et al. 2022) ===` followed by findings or `skipped: …`. |

## Part B — silent correctness errors

| ID | Cells | Fix | Assertion / print | What you should see |
|----|-------|-----|-------------------|--------------------|
| B1 | 47, 50, 52, 74 | Spatial scalar injection removed. `z = cat([pool(features(mask)), scalars]); head(z)`. `scalar_proj`/`scalar_fuse` deleted. Constant `ones` channel removed. | 52 asserts the head's scalar-input weight norm `> 0` after the **first** optimiser step; 50 asserts changing height/weight **does** change the output at init (the old test asserted the opposite); 74 asserts `\|Δ\| > 0` for chest/waist/hip under +10% stature, for both variants. | 52: `head weights on scalar input columns after step 1: 2.8e-02`. 50: `max \|delta output\| when only height/weight channels change, at init: 2.5e-03`. 74: a per-girth `\|Δ\|` table, all non-zero. |
| B2 | 8, 47, 50 | Per-view size 480×640 → **640×480** (portrait, matching the native mask aspect). Canvas 640×960. `batch_size` 8 → 6. `forward()` docstring corrected. | 50 asserts the model's per-view aspect matches the measured native aspect within 1%. | `native aspect (H/W): [0.75…]`, `model per-view aspect (H/W): 1.3333`, `relative aspect error: [0.0…]` and the assertion passes. |
| B3 | 38, 44 | `_row_for` returns `y1 − f·span`. The old formula mirrored every slice (chest was read at upper-thigh height, ankle at the crown). | 44 asserts `ankle_row > chest_row` in image coordinates; prints the row table. | `ankle row 864 > chest row 251: slice order is correct`. |
| B4 | 38, 44 | Slice widths use the connected foreground run **containing the midline**; thigh/calf/knee/ankle use the one **leg** run nearest the midline. Each measurement's calibration is restricted to its own features via `features_for()`. | 44 asserts ≥1 slice row has >1 foreground run; prints the run table and the per-measurement feature list. | `slices whose row has >1 foreground run: 5/13`, plus `chest uses ['chest_perim_mm', 'chest_width_mm', 'mm_per_px', 'height_cm']`. |
| B5 | 52, 62, 74, 81, 89, 92, 95, 117 | All primary metrics, checkpoint selection, sweeps and conformal calibration are **per photo pair**. Subject averages are a labelled secondary table. Conformal uses one seeded pair per calibration subject. | 81 asserts exactly one prediction per calibration subject; 84 asserts `conformal.json` records `unit='photo_pair'` and `pairs_per_subject=1`; 74/89/95 print photos-per-subject next to every table. | 81: `conformal calibration unit: 1 photo pair per subject (200 pairs from 200 subjects)`. 84: `conformal.json records unit='photo_pair', pairs_per_subject=1`. |
| B6 | 59, 69 | `boundary_jitter` thresholds `signed_distance − field·sigma`, moving the boundary both ways. The dead duplicated `np.where` branch and the `grown \|= fg` OR are gone. | 69 asserts `\|mean area change\| < 0.5%` over 50 masks **and** that masks both grew and shrank. | `jitter area change over 50 masks: mean +0.142%, range [-1.13%, +1.56%], shrank 19, grew 31`. |
| B7 | 64, 76 | The circular "ΔMAE at `none` is 0" check is replaced by an independent recomputation of the unperturbed MAE (plain `DataLoader` vs the sweep), for B1, B2, V-HW and V-H; `morph k=0` asserted equal to `none`. | 64 asserts `max \|diff\| < 1e-6` for B1 and B2; 76 repeats it for both variants. | `B2 identity sweep MAE vs independent per-pair MAE, max \|diff\| (mm): 0.00e+00`. |
| B8 | 62, 66, 76, 117 | Criterion is **total** error (Δ-only also published, labelled); contiguity rule walks outward from 0; erosion and dilation separate rows; downsample reports a **factor** with NaN px/mm; `"none tolerable"` instead of `0.0`. | 66 asserts downsample px/mm are NaN and both erosion and dilation rows exist. | `erosion stops at k=-2, dilation at k=+4 despite k=+5 failing`; `unit=['factor']`; `rows where NO magnitude passes … 'none tolerable', not 0.0`. |
| B9 | 75, 76, 117 | Sweep re-run on V-HW and V-H after training; model card publishes V-HW's budget. B2's sweep remains the source of the augmentation magnitudes. | 76 asserts the V-HW budget is non-empty and labelled V-HW; 117 asserts the card's table has ≥1 row. | `=== the deliverable: maximum tolerable boundary error for V-HW ===` followed by a populated table, and a B2-vs-V-HW comparison. |
| B10 | 62, 66 | Augmentation magnitudes require **all** measurements except `height` to pass, with B8 contiguity. | 66 prints the binding (first-failing) measurement; asserts it is a string or None. | `binding measurement: chest` and `rule: every measurement except ['height'] must stay within 25 mm`. |
| B11 | 59, 69 | `WorkerRNG` builds its generator lazily per worker from `(seed, worker_id, epoch)`. | 69 feeds the **same** photo through a 2-worker `DataLoader` and asserts the outputs differ; also asserts `set_epoch` changes the stream. | `byte-identical across workers: False` and `per-worker RNG: distinct augmentation streams confirmed`. |
| B12 | 46, 79, 81, 84, 112, 117, 134 | Half-width is `q × 10`, not `q × 20`; §7 demo uses a real V-HW prediction; the `… or True` assertion is replaced by a two-directional flag check; seeds re-set from config after it loads; §2 sex labels `.reindex([0, 1])`; `infer.py` keeps the point estimate inside a clamped interval and reports `"clamped": true`; final message says `/kaggle/working`. | 79 asserts `q*10`; 84 asserts every interval contains its own point estimate after clamping; 112 asserts interval nesting and the `clamped` key; 74 asserts flag consistency in both directions. | 79: `half-width in mm = q*10 = 50 mm (the old code printed q*20 = 100 mm)`. 84: `every interval still contains its own point estimate after clamping`. |

## Part C — training budget

| ID | Cells | Fix | Assertion / print | What you should see |
|----|-------|-----|-------------------|--------------------|
| C1 | 0, 72, 74, 117, 134 | Optimiser steps computed per model and printed in §11 and beside the flag table; the B0 flag table is labelled **PROVISIONAL**. | 117 interpolates `TRAIN_STEPS` into the card; 74 prints the step count with the table. | `optimiser steps: {"B2": 1275, "V-HW": 1700, "V-H": 1700}` and `this run is 0.85% of that budget at minimum`. |
| C2 | 70 | New cell times 20 augmenter calls and prints ms/sample against `cfg["augmenter_budget_ms"]`. | Prints the median and the budget; prints remediation options if over. | `augmenter: median NN.N ms/sample over 20 calls … budget 50 ms/sample`. |
| C3 | 0 | §0.5 runtime table replaced with "measured runtimes: see §11". | — | The header reads *"Measured runtimes are printed per phase in §11 for the run that produced them."* |

## Part D — documentation integrity

| ID | Cells | Fix | Assertion / print | What you should see |
|----|-------|-----|-------------------|--------------------|
| D1 | 98, 103, 117, 126 | ONNX claim now refers to the printed result; the preprocess self-test is a **full-tensor** equivalence check — §9.1 saves the reference input masks and scalars, and `self_test` recomputes and compares at 1e-6. | 103 asserts the subprocess printed `PASS`; also asserts each scalar channel is constant. | `preprocess self-test: PASS (4 references recomputed and compared at atol=1e-06, IMG 640x480, threshold 127)`. |
| D2 | 119 | All hard-coded run numbers removed from markdown and DECISIONS.md ("45/27 silhouettes", "164 cm", "88 cm", the shared-id claim, "960x720", "<50 ms", "clean masks below 0.05", the runtime table). | 119 asserts five forbidden strings are absent. | The patch applies cleanly; no assertion fires. |
| D3 | 47, 50, 119 | One consistent stem description: mean-reduced 3→1 stem, scalars at the head. | 50 asserts `stem_in_channels() == 1`. | `stem: mean-reduced 3->1 input channels (MNASNet trunk, stem index 0); trunk in_channels=1`. |
| D4 | 10, 115, 119 | `DEFINITIONS` is one uniform statement for all 14 columns; `height_cm` is "from `hwg_metadata.csv` (collection method not stated)". | — | Every `schema_map.yaml` measurement carries the same `definition` string. |
| D5 | 121 | Ramanujan corrected to *Q. J. Math.* 45, 350–372; the unverifiable anthropometric citation removed and the slice table labelled `[Assumption]`; Bland & Altman + Ramanujan under a "foundational-method exceptions" heading; new bidirectional citation cross-check cell. | 121 asserts no in-text surname is absent from `references.md`. | `in-text surnames absent from references.md entirely: none`. |
| D6 | 3, 16, 66, 92, 95, 136 | `%matplotlib inline` in cell 3; every figure now `plt.show()`ed before `plt.close()`. | — | Figures render inline in the notebook output. |
| D7 | 119 | Five new dated entries: D15 head-level scalars, D16 per-pair evaluation, D18 total-error budget, D17 shipped-model sensitivity, D19 provisional budget. | — | `  22 dated entries` printed. |

## Part E — run plan

| ID | Cells | Fix | What you should see |
|----|-------|-----|--------------------|
| E1–E3 | 0 | §0.4 now carries the three-run plan, including the clause requiring a DECISIONS.md entry stating that Test-A/B results have already been seen if §1–7 is fixed after the final run. | The header's *Run plan* section. |

## Run protocol

| ID | Cells | Fix |
|----|-------|-----|
| — | 8, 54, 64, 87, 89, 92, 95, 112, 115, 117, 123, 125, 132, 134, 136 | `smoke_test` and `run_final_eval` config keys, honoured by every dependent cell. A smoke run caps subjects/photos/epochs and prints `SMOKE RUN`. With `run_final_eval: false`, §8 prints a skip notice, `infer.py`'s smoke test falls back to a **validation** subject, and the model card, `schema_map.yaml`, the summary figure and §11 all report "final evaluation not run" instead of failing. |

---

## Bugs found beyond the fix brief

These were not in `fixbrief.md`. Each would have crashed or silently corrupted the run.

| # | Cells | Bug | Fix |
|---|-------|-----|-----|
| X1 | 47 | `MeasurementNet` used `net.features`, but **MNASNet has no `.features` attribute** — its trunk is `net.layers`. So even after fixing the backbone name, construction raised `AttributeError`. (For ResNet-family models `features` does exist, so this only bites once A2 is fixed.) | `_trunk()` resolves `features` or `layers`; `_replace_stem()` handles both a bare `Conv2d` stem (MNASNet) and a wrapped stem block. Cell 50 notes it. |
| X2 | 47 | `to_tensor` built `mask_ch` as 3-D but the scalar channels as 2-D, so `np.concatenate(..., axis=0)` raised `ValueError: all the input arrays must have same number of dimensions`. This fired on the first `to_tensor` call in cell 54. | `mask_ch` kept 2-D and the channels combined with `np.stack`. |
| X3 | 47 | `N_CHANNELS = 5` / `N_CHANNELS_VH = 4` did not match what `to_tensor` actually built (4 and 3, including the `ones` channel). Every channel-count assertion in §4, §6 and §9 therefore disagreed with the tensor it was checking. | Corrected to the true counts after removing `ones`: **`N_CHANNELS = 3`, `N_CHANNELS_VH = 2`**. |
| X4 | 10, 38 | `bodym_data.py` and `bodym_baselines.py` used `Sequence` / `Optional` in annotations without importing them. Survived only because of `from __future__ import annotations`; would break `typing.get_type_hints` and any static check. | Both imports fixed. |
| X5 | 34, 89 | `stratified_metrics` used `pd.Series.groupby(s).groups`, whose index semantics changed in pandas 2.2; cell 89 built `y_true` from `subject_table` order but compared it against `b1_matrix`'s **sorted** order, so its assert only passed because the ids happened to be pre-sorted. | `stratified_metrics` enumerates `pd.unique(labels)` directly; `evaluate_b0_b1` indexes `subject_table` by the requested order explicitly. |
| X6 | 62, 106 | `max_tolerable_boundary_error` blanked `max_tolerable_px`/`_mm` for downsample rows but put the factor in those same columns — so the answer was NaN'd away. `infer.py` also defaulted `backbone` to the dead `mnasnet1_0_3_0`. | New `max_tolerable` + `unit` columns carry the value; `infer.py` raises `KeyError` if the checkpoint records no backbone. |
| X7 | all | The notebook failed `nbformat.validate` — code cells lacked `id`, `execution_count` and `outputs`. | The patcher normalises them (107 keys added). |
| X8 | 47 | The MNASNet stem is a bare `nn.Conv2d`, which `setattr(parent, int_key)` rejects (`nn.Sequential` is indexable, not attribute-settable by int). | `_replace_stem` branches on the parent type. |

## Verification performed

The patch was checked without BodyM, using synthetic silhouettes:

- every patch source parses (`patch_notebook.py --check`)
- `nbformat.validate` passes on the patched notebook (142 cells)
- every code cell's source parses with `ast.parse`
- `src/` modules were executed against synthetic A-pose masks:
  - **B3** slice rows come out monotone in the correct direction (ankle 864 > calf 750 > crotch 476 > waist 341 > chest 251 > shoulder 163)
  - **B4** chest width 259 px excludes the arms (torso run `(221, 479)`, arms extend to ±300); thigh reads one leg (59 px) not both (118 px); `features_for("chest")` and `features_for("leg-length")` are disjoint
  - **A1** `predict()` equals `Ridge.predict()` for every derived target, and intercepts are finite and non-trivial
  - **B6** jitter over 50 masks: mean +0.142 %, range [−1.13 %, +1.56 %], 19 shrank / 31 grew; and it differs from `dilate(+4)` (added 4122 px, removed 4428 px)
  - **B1** scalars change the output at init (2.5e-03); the head's scalar-input weight norm is 2.9e-02 after one optimiser step; +10 % stature moves every girth for both variants
  - **B8** on a deliberately non-monotone curve, erosion stops at k=−2 and dilation at k=+4; the downsample factor survives as `max_tolerable=2.0, unit='factor'` with px/mm NaN; `"none tolerable"` is emitted when magnitude 0 fails
  - **B10** the binding measurement is named (`chest`) and `height` is excluded
  - **B12** `q × 10 = 50 mm` vs the old `q × 20 = 100 mm`; a too-tight envelope leaves the point estimate inside its interval and sets `clamped`

Not executed here (no BodyM, no GPU): the data-integrity suite, the EDA figures, B0/B1 fitting on
real masks, CNN training, the sweeps, ONNX export and the subprocess tests. Those run on Kaggle.

## Addendum A - execution, persistence and resume

Second patch, applied by `patch_addendum.py` after the fix-brief patch. 142 -> 150 cells
(8 inserted: 3 markdown, 5 code).

| ID | Requirement | Implementation |
|----|-------------|----------------|
| A0 | Phases 0-3 run on CPU | `phase_wrap.guard()` wraps CPU-only phases in `run_state.cpu_guard()`, which raises on any `Tensor.to('cuda')` / `.cuda()`. Trips are asserted in smoke mode, so the claim is executable rather than printed. |
| A1 | Working-directory file/GB budget | `run/ckpt/{variant}/` holds exactly `last.pt` and `best.pt`; nothing is written per epoch. `check_working_budget()` asserts `max_files_working` (200) and `max_gb_working` (15.0) against Kaggle's ~500 files / 20 GB, with the ten largest files named. Regenerable caches go to `/kaggle/temp`. |
| A2 | `config_hash` gating | 16-hex sha256 over the sorted YAML config, excluding only `hash_exclude`. `smoke_test` and `run_final_eval` are **included** - both change what is computed. A phase is reusable only when status is `done` **and** the hash matches **and** every listed artefact exists; all three are checked independently and each is proven to fail on its own in smoke mode. |
| A3 | Resume across sessions | `adopt_resume_source()` looks for `<working>/run/manifest.json` first, then `/kaggle/input/*/run/manifest.json`, prefers the working directory, and prints which source it used. A corrupt manifest degrades to a fresh one rather than aborting. |
| A4 | Resume must be deterministic | Three properties, each exercised: step-based cosine LR over the full `total_steps` budget (stored in `last.pt` and reused on resume); per-epoch `(seed, epoch)` shuffle generator; Python/NumPy/torch/CUDA RNG restored from `last.pt`. `best.pt` loads only on a fresh start. Early stopping (patience 3, min_delta 1 mm) with the counter persisted. Session-time guard saves, marks `partial`, and calls back instead of raising. |
| A5 | Single-use test sets | `evaluate_test_lock()` compares the SHA-256 of each `best.pt` against the previous `final_eval.json`: `run` / `reuse` (same weights - test sets not re-read) / `blocked` (changed weights). On `blocked` the phase is marked `blocked`, the reason printed, **and the export still runs** with the absence recorded on the model card. |
| A6 | Atomic writes | `atomic_save()` / `atomic_write_bytes()` write to a `.tmp<pid>` sibling then `os.replace`. Verified: no `.tmp` residue, and no orphan `.npz` (numpy appends the suffix to a path that lacks it - the writer hands it a file object instead). |
| A7 | Machine-checkable gates | `RunState.validate()` schema-checks the manifest at startup; `check_working_budget()`; `cpu_guard()`; the determinism/early-stopping/time-guard cell. |

### Verification performed here

- `patch_addendum.py --dry-run` then apply; a second run refuses with exit 3 rather than corrupting
  the file (indices shift once cells are inserted - this was found and fixed after it happened once)
- `nbformat.validate` passes on all 150 cells
- every code cell parses with `ast.parse`
- `_test_rs.py` - 33 checks on `run_state`: hash stability and `hash_exclude`; all three reuse
  conditions failing independently; payload round-trip (DataFrame/ndarray/JSON/meta); manifest schema
  validation; corrupt-manifest recovery; `maybe_skip` / `BLOCKED` / `partial` blocking; the
  test-set lock's four branches; the CUDA tripwire; the budget cap; atomic writes with no residue.
  **All passed.**
- `_test_train.py` - resume determinism, early stopping, the time guard, checkpoint contents.
  **All passed.** Headline: `|straight - resumed| = 0.000e+00 mm` against the addendum's 1e-3
  tolerance, with identical per-epoch validation history and identical per-epoch learning rates.

### Two design points worth flagging

**The determinism test interrupts with `max_epochs_this_call`, not with a smaller `epochs`.**
Passing `epochs=1` for the first session changes `total_steps` from 24 to 12, so the cosine schedule
differs from step 0 and the two runs diverge for a reason unrelated to resume. That was the first
version of the test and it failed - the failure was in the test, not the code. A real session ends
via the time guard, which leaves the configured budget untouched. The cell now asserts the
per-epoch learning rates match, so this cannot regress unnoticed.

**Early stopping makes `TRAIN_STEPS` a result, not an input.** With `epochs_prod: 8` and
`patience: 3`, variants will often stop around epoch 4-5. Every step-count comparison against a
published budget must therefore quote the printed `epochs_completed` / `total_steps_run` for that
variant. Stated in the markdown at the training cell rather than buried here.

### Deliberately not applied

- Early stopping to B0's `HistGradientBoosting`. sklearn's internal validation split would make the
  non-visual bar stochastic, so every "B0 beats vision" comparison would hinge on a coin flip. B1's
  ridge is closed-form and needs no such treatment. Both stay deterministic.
- A hard raise on A5 `blocked`. Sections 9-11 still produce a bundle whose model card records the
  absence and the reason.

## Not implemented

| Item | Reason |
|------|--------|
| 50 *real* masks for the B6 jitter statistics | Needs BodyM; the cell samples 50 from `IDX["train"].photos`. |
| The linter's findings | `pynblint` runs on Kaggle; the cell skips cleanly when the `.ipynb` is absent. |
| D5's citation cross-check completeness | The regex catches `Author (year)` patterns only, so `Angelopoulos & Bates (2021)` is listed for manual review rather than silently passing. The hard assert is on surnames, which does cover it. |