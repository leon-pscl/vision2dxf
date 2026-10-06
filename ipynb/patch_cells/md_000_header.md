# BodyM Measurement Model — Training, Evaluation & Export

**Notebook version** 2.0 · **Fix brief version** 2026-10-07 · **Pipeline step** 6 of 10

---

## 0.1 Purpose

This notebook trains, evaluates, documents and exports the **body-measurement estimation model**
for step 6 of a 10-step made-to-measure (MTM) pattern pipeline:

| # | Step | # | Step |
|---|------|---|------|
| 1 | Garment type | 6 | **MEASUREMENT ESTIMATION ← this notebook** |
| 2 | Capture (front A-pose + side) | 7 | Schema validation + YAML |
| 3 | Person segmentation | 8 | Pattern drafting |
| 4 | Scale / perspective calibration | 9 | Production details & export |
| 5 | Landmarks | 10 | Virtual drape + physical toile vs tape-measured ground truth |

**Input contract.** front silhouette (binary mask), side silhouette (binary mask), `height_cm`,
optionally `weight_kg`.
**Output contract.** 14 body measurements in cm, each with 80% and 90% prediction intervals.

Step 7 consumes the intervals for plausibility gating; step 4 consumes them in its error budget.

**Evaluation unit: one photo pair.** That is what `infer.py` receives, so every primary metric,
checkpoint decision and conformal score is computed per front/side pair. Subject-averaged metrics
exist as a clearly-labelled secondary table.

## 0.2 Why this notebook is also documentation

Every artefact the pipeline needs later (schema map, model card, decision log, references, pinned
requirements, trained weights) is written **by this notebook**, so the provenance of the exported
model is never separated from the code that produced it. The notebook must pass *Restart & Run All*
with no hidden state.

## 0.3 Data

**BodyM** (Ruiz et al., 2022) — binary silhouettes + height + weight + sex + 14 measurements.
**No RGB images.** Therefore **no segmentation model is trained here**: step 3 of the pipeline owns
that. This notebook only consumes masks.

Licence **CC BY-NC 4.0** — **non-commercial use only**. See §10.1 for the registry inconsistency.

## 0.4 How to reproduce

1. Upload BodyM as a Kaggle Dataset and attach it to the notebook (GPU accelerator recommended).
2. Kaggle menu → **Save Version** → **Save & Run All**.
3. Collected artefacts land in `/kaggle/working/export/` and are zipped to
   `/kaggle/working/export.zip`. (Kaggle's `/kaggle/outputs` is only populated on Save Version;
   during Run All the zip must be downloaded from the notebook's output pane.)

`config.yaml` is written on first run if absent, then **read back** as the single source of
hyperparameters. Change a value there, re-run, and every phase follows.

### Run plan

Three runs, in this order. Do not skip to the third.

**E1 — Smoke run.** `smoke_test: true`, `run_final_eval: false`. One epoch, at most 50 subjects per
split, at most 2 photos per subject, 10 sensitivity subjects. Every assertion in the notebook still
runs. Purpose: catch crashes in minutes instead of hours. Expected wall clock: minutes.
**Every assertion must pass.** If one fails, fix it before E2 — a smoke run that ignores a failure
teaches you nothing.

**E2 — Full training run.** `smoke_test: false`, `run_final_eval: false`. Real splits, real epochs.
Review the validation tables, the sensitivity tables, and specifically the B1 slice-row check, the
B6 jitter area check, the B11 two-worker RNG check and the B9 V-HW sensitivity budget. Purpose:
confirm the science before the single-use evaluation budget is spent. Expected wall clock: hours —
the boundary sweep now runs three times (B2 for the augmentation magnitudes, then V-HW and V-H), and
the input is 640×480 per view.

**E3 — Final run, once.** `run_final_eval: false` → `true`, then **Save Version → Save & Run All**.
**No training-code changes after this.** Test-A and Test-B are each read once, in §8.

> **If a fix to sections 1–7 is needed after E3 has run**, apply it, and log it in `DECISIONS.md`
> **with an explicit statement that Test-A/Test-B results have already been seen** — because at that
> point the single-use protocol is no longer intact and any change made in response to those results
> is selection on the test set.

## 0.5 Hardware & expected runtime

Measured runtimes are printed per phase in §11 for the run that produced them. There is no
predicted-runtime table here: the previous one was a guess written before the code ran, and it was
wrong about both the total and the per-phase split.

| Item | Value |
|------|-------|
| GPU | 1× T4 / P100 (16 GB). CPU-only runs, but the sweep dominates and becomes impractical |
| Disk | < 1 GB dataset, < 500 MB artefacts |
| RAM | ≥ 8 GB (dataset is streamed, not cached) |
| Smoke run | minutes |
| Full run | hours; dominated by the three sensitivity sweeps and by §9's ONNX verification |

Knobs that trade accuracy for time live in `config.yaml` (`epochs_b2`, `epochs_prod`, `img_height`,
`img_width`, `batch_size`, `backbone`, `n_sensitivity_subjects`, `n_onnx_check`).

Note that `epochs_b2` and `epochs_prod` are a small fraction of the training budget in Ruiz et al.
(2022). Every "beats the non-visual baseline" statement this notebook makes is labelled **PROVISIONAL**
and printed beside the optimiser step count. See §6 and DECISIONS.md D19.

## 0.6 Table of contents

- **§1 Phase 0** — setup, environment, seeds, data integrity
- **§2 Phase 1** — exploratory data analysis
- **§3 Phase 2** — subject-level splits
- **§4 Phase 3** — baselines B0 (non-visual), B1 (geometric), B2 (BMnet-style CNN)
- **§5 Phase 4** — boundary-sensitivity experiment (key deliverable)
- **§6 Phase 5** — production model, variants V-HW and V-H, plus the shipped-model sweep
- **§7 Phase 6** — split conformal prediction
- **§8 Phase 7** — final evaluation on Test-A / Test-B (run **once**)
- **§9 Phase 8** — export bundle
- **§10** — licence, required explanations, open decisions
- **§11** — reproducibility summary