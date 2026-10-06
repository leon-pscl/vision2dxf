_TODAY = time.strftime("%Y-%m-%d")
_ARTEFACT_P50 = float(np.percentile(EDA["artefact_scores"]["artefact_score"], 50))
_ARTEFACT_P99 = float(np.percentile(EDA["artefact_scores"]["artefact_score"], 99))
_MAX_PPS = int(EDA["split_summary"]["photos_per_subject_max"].max())
_SHARED_AB = integrity_report[integrity_report.check == "leakage/testA->testB"]
_SHARED_AB_N = int(_SHARED_AB.detail.iloc[0].split()[0]) if len(_SHARED_AB) else 0

DECISIONS = f"""# DECISIONS.md — dated decision log

Every entry: decision, alternatives considered, evidence, consequences, open questions.
Evidence tags: **[Ref]** published reference · **[Exp]** experiment cell in this notebook ·
**[Assumption]** stated, unverified assumption.

---

## D1 — {_TODAY} — Split by subject, not by photo

**Decision.** All three splits (fit/val/calib) are assigned by `subject_id`, stratified by
sex x BMI band, drawn from the BodyM `train` split only.

**Alternatives.** (a) photo-level random split; (b) subject-level split without stratification.

**Evidence.** **[Exp]** Section 2 printed the per-subject silhouette count; the maximum observed
is {_MAX_PPS} (`EDA["split_summary"]`). **[Exp]** The stratification table in section 3 shows the
per-stratum counts in all three splits.

**Consequences.** Primary metrics are per **photo pair**, so a heavily photographed subject
contributes proportionally more rows — which is correct, because deployment also sees one pair
per capture. Validation contains fewer distinct capture conditions per subject than the raw photo
count suggests, which slightly under-samples Test-A-style variation; Test-A is reported
separately for that reason.

**Open questions.** None.

---

## D2 — {_TODAY} — Test-A and Test-B evaluated exactly once

**Decision.** Both test splits are touched only in section 8. No selection, hyperparameter,
augmentation magnitude, calibration or threshold uses them. A protocol audit asserting zero
subject-id and zero photo-id overlap with the training splits runs immediately before the
evaluation.

**Alternatives.** (a) use Test-A for checkpoint selection; (b) k-fold CV on train and ignore both.

**Evidence.** **[Ref]** Brief section 1. **[Exp]** The printed audit in section 8 shows the
overlap for both id types on both splits.

**Consequences.** Model selection rests on {len(ids_val)} validation subjects, so selection noise
is larger than it would be with a larger selection set. The `infer.py` smoke test in section 9
falls back to a **validation** subject when `run_final_eval` is false, so the single-use budget is
never spent on a smoke test.

**Open questions.** The integrity report records {_SHARED_AB_N} subject id(s) shared between
Test-A and Test-B. That does not affect either split's independence from training, but it does mean
the two test splits are not independent of *each other*, so the A-to-B gap should be read as a
sensitivity estimate rather than as two independent samples. Whether this is a genuine duplicate
capture or an id collision in the source registry is unknown.

---

## D3 — {_TODAY} — B0 is a mandatory bar, and failing it is reported

**Decision.** B0 (height + weight + sex, ridge and gradient boosting) is computed first and every
vision model is compared against it per measurement. Measurements where a variant fails to beat B0
are flagged in the model card, not dropped.

**Alternatives.** (a) compare only aggregate MAE; (b) report only the best model per measurement.

**Evidence.** **[Ref]** Brief section 5. **[Exp]** `validation_mae_comparison.csv` and the flag
table in section 6.

**Consequences.** Some measurements may be better served by the non-visual formula. This is stated
rather than hidden, and pipeline step 7 can act on it. **The flag table is PROVISIONAL**: see D19.

**Open questions.** Whether step 7 should switch measurement-by-measurement between the vision
model and B0, or keep one model for all 14.

---

## D4 — {_TODAY} — Two height definitions ship with BodyM; `height_cm` is the model input

**Decision.** `hwg_metadata.csv:height_cm` is the scalar input and the basis for the mm/px scale.
`measurements.csv:height` (mesh vertex-path stature) is retained as a 14th target when
`include_height_as_target` is true. The dataset does not state how `height_cm` was collected, so
it is not described as self-reported.

**Alternatives.** (a) use `measurements.height` as the input; (b) average the two.

**Evidence.** **[Exp]** `verify_integrity` quantifies the disagreement per split; the printed
mean and max |diff| and correlation are in the integrity report.

**Consequences.** The exported model requires a stature value from `hwg_metadata.csv`, which the
capture protocol (step 2) must supply. The predicted `height` output is close to an identity
function and should not be treated as an independent measurement.

**Open questions.** Whether the mesh-derived `height` target adds value, or should be dropped via
`include_height_as_target: false`.

---

## D5 — {_TODAY} — Mean-reduced 3→1 stem; scalars enter at the head

**Decision.** The ImageNet stem's weights are **mean-reduced over the input-channel axis** to
`(out, 1, k, k)`, the standard grayscale adaptation, preserving every pretrained output filter
with full spatial structure. Height and weight are read out of their constant channels and
concatenated to the average-pooled image features immediately before the MLP head. Nothing is
zero-initialised, so the head's scalar-input weights receive gradient from the first optimiser
step.

**Alternatives.** (a) re-initialise the stem; (b) duplicate the mask into three channels;
(c) feed the scalars as spatial channels summed at the second convolution; (d) zero-initialise a
scalar projection at the stem.

**Evidence.** **[Exp]** Section 4.3 asserts the stem is 1-input-channel after construction, that
the scalar readout is exact, and that changing height/weight **does** change the output at
initialisation. **[Exp]** `train_model` asserts the head's scalar-input weights are non-zero after
one optimiser step. **[Exp]** Section 6 asserts a +10% height change moves every key girth
prediction for both variants.

**Consequences.** (c)/(d) were not merely worse — they were **dead**. Two zero-initialised
projections in series give zero gradient to both weight matrices, so only a bias could learn, and
that bias cannot see the scalars. Both variants were effectively silhouette-only, and the
V-HW-vs-V-H ablation that section 6 reports measured nothing. The replacement costs one extra
input column on the head's first linear layer.

**Open questions.** Whether a FiLM-style multiplicative conditioning would beat concatenation.
Not tested here.

---

## D6 — {_TODAY} — V-H removes the weight channel rather than zero-filling it

**Decision.** V-H has 3 input channels (the `ones` channel having been removed as uninformative);
the weight channel is absent. `infer.py` selects the variant by whether `weight_kg` is supplied,
and raises if the pairing is inconsistent.

**Alternatives.** (a) zero-fill the weight channel; (b) fill with the training mean.

**Evidence.** **[Ref]** Brief section 7. **[Ref]** The input ablation in Ruiz et al. (2022)
motivates exporting both variants. **[Exp]** The channel-count assertions in sections 4.4 and 6,
and the end-to-end variant-selection test in section 9.

**Consequences.** Zero-filling would tell the network "weight = 0", an out-of-distribution value it
could learn to treat as a flag. Removal is the honest encoding of "not measured".

**Open questions.** Which variant is the pipeline default — see D8.

---

## D7 — {_TODAY} — Augmentation magnitudes come from the Phase-4 sweep

**Decision.** Erosion, dilation and jitter magnitudes are the largest values at which **every**
measurement except `height` (an input passthrough, not a prediction) still satisfies the
{MAX_TOLERABLE_BOUNDARY.tolerance_mm.max():.0f} mm threshold under the *total-error* criterion, walked
outward from zero so the pass range is contiguous. The binding (first-failing) measurement is
`{AUG_MAG["binding_measurement"]}`. Downsample augmentation uses the smallest swept factor only.

**Alternatives.** (a) the magnitudes used by Ruiz et al. (2022); (b) arbitrary values; (c) a broad
range to maximise robustness; (d) the previous rule, which required only *any one* measurement to
pass while the surrounding text claimed *all* did.

**Evidence.** **[Exp]** `AUG_MAG` is derived programmatically from `sensitivity.csv` in section 5
and the binding measurement is printed, so the rule is checkable rather than claimed.
**Rationale for the families:** **[Ref]** brief section 7.

**Consequences.** The production model is trained to be robust to exactly the boundary errors the
sweep found tolerable, and no more. Rule (d) was too generous: one easy measurement could carry a
magnitude that eleven others fail, which is why the binding measurement is now named.

**Open questions.** Whether the thresholds themselves should be tightened once the pattern engine
states its per-measurement tolerances (D9).

---

## D8 — {_TODAY} — Default variant left open for the user

**Decision.** Both variants are exported and `infer.py` selects by weight availability. No default
is hard-coded.

**Alternatives.** (a) make V-HW the default and require weight collection; (b) make V-H the
default so weight is optional.

**Evidence.** **[Exp]** `weight_gain_mm` in `validation_mae_comparison.csv` quantifies the cost of
dropping the weight channel, per measurement.

**Consequences.** The pipeline works either way, and the accuracy cost of not collecting weight is
a number rather than an assumption. **[Exp]** The scalars demonstrably reach the predictions now
(D5), so that number is meaningful for the first time.

**Open questions.** **TODO(user)** Does the capture protocol (step 2) collect weight? This is an
open decision for the user and is deliberately not resolved here.

---

## D9 — {_TODAY} — Pattern-engine keys and tolerances left open for the user

**Decision.** `schema_map.yaml` ships `TODO(user)` pattern keys and the reporting thresholds
{cfg["tolerance_mm"]} mm are labelled as reporting-only. The plausibility envelope in `infer.py`
and `conformal.json` is a placeholder marked `TODO(user)`.

**Alternatives.** (a) invent plausible key names; (b) omit the file and let step 7 discover the
names at runtime.

**Evidence.** **[Ref]** Brief sections 11 and 14.

**Consequences.** Step 7 cannot run unattended until the keys are filled in. Inventing names would
produce a bundle that looks complete and fails at integration.

**Open questions.** **TODO(user)** (1) pattern-engine key names; (2) which tolerance the pattern
engine accepts per measurement; (3) the plausibility envelope.

---

## D10 — {_TODAY} — Conformal intervals are marginal, conditional on exchangeability, and need recalibration

**Decision.** Split conformal quantiles per measurement per variant, finite-sample corrected, with
the coverage guarantee and all its breaking conditions written into `conformal.json`,
`model_card.md` and this log.

**Alternatives.** (a) Gaussian residual intervals; (b) quantile regression; (c) conformal with a
per-subject (Mondrian) partition.

**Evidence.** **[Ref]** Angelopoulos & Bates (2021). **[Exp]** The hand-checked quantile
self-test in section 7 pins the finite-sample correction, and a second self-check pins that q is a
half-width (so mm = q×10, not q×20). **[Exp]** Empirical coverage on the held-out splits is
reported in section 8 rather than assumed.

**Consequences.** Coverage is marginal over pairs, not per-subject, and it weakens under
segmenter, device and population shift. A Mondrian partition by BMI band would improve conditional
coverage but shrinks each calibration set below a size where the quantile is reliable.

**Open questions.** Recalibration on tape-measured ground truth is deferred to pipeline step 10 and
is a hard prerequisite for using these intervals in production gating.

---

## D11 — {_TODAY} — Licence treated as non-commercial despite the registry inconsistency

**Decision.** BodyM is handled as **CC BY-NC 4.0, non-commercial**. The AWS registry entry links to
the CC BY legal code; the discrepancy is recorded rather than resolved in favour of the more
permissive reading.

**Alternatives.** (a) treat as CC BY (commercial permitted) because the registry says so;
(b) refuse to use the data.

**Evidence.** **[Ref]** brief section 1.

**Consequences.** Commercial deployment of a model trained on this data needs written clarification
from the rights holders. Model weights trained on NC data may inherit the restriction; that is a
legal question, not a modelling one.

**Open questions.** **TODO(user)** Whether to seek commercial licensing, or to retrain on a
commercially-licensed dataset.

---

## D12 — {_TODAY} — B1's slice table is an explicit assumption, and its slice rows are verified

**Decision.** B1 uses a documented table of anatomical slice heights as a fraction of stature
measured **from the floor**, labelled **[Assumption]** with no published source claimed. The
per-measurement linear calibration is fitted on the fit split only, using **only that
measurement's own features** (its perimeter, its width, the scale terms, and for lengths its own
row difference).

**Alternatives.** (a) learn slice heights; (b) measure them from the SMPL fit; (c) cite a standard
anthropometric table.

**Evidence.** **[Exp]** The overlay figure `fig_b1_slices.png` shows every slice row on one mask
and the assertion `ankle_row > chest_row` verifies the vertical mapping. **[Exp]** The coefficient
probe shows which geometry each prediction leans on. **[Exp]** B1 cannot beat a constant
train-mean predictor — asserted, because a failure there means a calibration bug rather than a
weak heuristic.

**Consequences.** The previous slice-row formula mapped the floor to the *top* image row, so every
girth was read at the wrong height; and the previous widths summed the whole foreground row,
including both arms and both legs, so every girth was systematically too large. Both are fixed and
both are checked. Restricting each calibration to its own features stops B1 borrowing, say, the
chest perimeter to predict the wrist — which is how it was quietly converging on B0.

**Open questions.** The citation offered for the slice heights could not be verified and was
removed (D21). Whether B1 earns a place in the exported bundle at all — it is retained as a
diagnostic and is not shipped.

---

## D13 — {_TODAY} — Input size preserves the native mask aspect ratio

**Decision.** Each view is resized to {cfg["img_height"]}×{cfg["img_width"]} (H×W) and the front and
side are concatenated horizontally into a {cfg["img_height"]}×{2*cfg["img_width"]} canvas. The
measured native mask size is
{list(EDA['image_stats'][["height_px", "width_px"]].drop_duplicates().itertuples(index=False, name=None))}
(H × W).

**Alternatives.** (a) train at native resolution; (b) the 480×640 in the previous version;
(c) follow the source brief's figure literally.

**Evidence.** **[Exp]** Section 2 measured the mask geometry directly. The assertion in section
4.3 requires the model's per-view aspect ratio to match the native one within 1%.

**Consequences.** The previous 480×640 transposed the aspect and squashed portrait silhouettes.
Matching the source brief's 640×960 combined canvas required H > W, i.e. 640×480 per view. That is
{cfg["img_height"]*cfg["img_width"]/(640*480):.2f}× the pixels of the previous setting, which is
why `batch_size` was lowered to {cfg["batch_size"]} and why the runtime in §11 is longer than the
old §0.5 estimate.

**Open questions.** Whether native-resolution training improves accuracy enough to justify the
extra compute.

---

## D14 — {_TODAY} — Segmentation model is out of scope for this notebook

**Decision.** No segmentation model is trained or fine-tuned. Ultralytics/YOLO is not a
dependency.

**Alternatives.** (a) train a segmenter on BodyM; (b) accept RGB input.

**Evidence.** **[Ref]** Brief section 1; BodyM has no RGB images, so there is nothing to train a
segmenter on here.

**Consequences.** The model depends entirely on step 3's mask quality, which is why section 5
quantifies the requirement rather than assuming it, and why the shipped budget comes from the
production model (D17).

**Open questions.** Which segmenter step 3 will use, and whether its boundary error falls inside
the tolerances in `max_tolerable_boundary_error.csv`.

---

## D15 — {_TODAY} — Scalars are concatenated to pooled features, not injected spatially

**Decision.** The zero-initialised spatial scalar projection is removed. Scalars are read out of
their constant channels, concatenated to the average-pooled image features, and passed to the MLP
head: `z = cat([pool(features(mask)), scalars_normalised])`, `head(z)`.

**Alternatives.** (a) keep the spatial injection but zero-initialise only its **last** layer;
(b) FiLM-style multiplicative conditioning; (c) feed the scalars through a small MLP first.

**Evidence.** **[Exp]** The previous design had two zero-initialised convolutions in series, so
both weight matrices received exactly zero gradient and only a bias could learn — a constant that
cannot see the scalars. It was also shape-incompatible: a 1280-channel full-resolution tensor
added to a 32-channel half-resolution stem output. **[Exp]** Section 6 now asserts that +10%
height moves every key girth for both variants.

**Consequences.** Both variants are genuinely different models for the first time, so the
V-HW-vs-V-H ablation in section 6 is interpretable rather than a comparison of two identical
networks. The constant `ones` channel was removed at the same time: it carried no information.

**Open questions.** Whether a small MLP on the scalars before concatenation would help.

---

## D16 — {_TODAY} — Primary evaluation is per photo pair; subject averages are secondary

**Decision.** All primary metrics, checkpoint selection and conformal calibration operate on **one
row per front/side photo pair**. Subject-averaged metrics are computed and reported as a
clearly-labelled secondary table. Conformal calibration uses **one seeded random pair per
calibration subject**, recorded in `conformal.json` as `unit: photo_pair`,
`pairs_per_subject: 1`.

**Alternatives.** (a) keep per-subject averaging everywhere; (b) sample several pairs per subject
and average the scores.

**Evidence.** **[Ref]** Brief section 11 fixes the `infer.py` signature at one front mask and one
side mask. Deployment therefore predicts from one pair. **[Exp]** The two test splits have very
different photos-per-subject counts, so per-subject averaging made the Test-A vs Test-B gap partly
a measurement of the averaging count rather than of capture quality.

**Consequences.** Reported errors are higher than the previous version's, and that is the correct
number: they are what a single capture produces. Conformal scores are exchangeable with what
deployment computes. The cost is that a heavily photographed subject now contributes more rows,
which is the right weighting for this deployment.

**Open questions.** None.

---

## D17 — {_TODAY} — The published boundary budget comes from V-HW, not B2

**Decision.** The sensitivity grid is run three times: on B2 (before training, because it is the
source of the augmentation magnitudes), and again on V-HW and V-H after they are trained. The
model card and `max_tolerable_boundary_error.csv` report **V-HW**.

**Alternatives.** (a) report B2's numbers and note the caveat; (b) report the best of the three;
(c) skip the post-training sweep.

**Evidence.** **[Exp]** The previous version swept B2 only, so the model card's
`model == "V-HW"` filter returned an empty table and the step-3 requirement in the shipped
documentation was blank.

**Consequences.** The sweep cost roughly doubles. In exchange, the number step 3 must satisfy is
the number the *shipped* model actually satisfies, which is the only version of that number anyone
can act on. V-HW's tolerable error should be at least B2's, since the variants were trained on
exactly these perturbations; the cell prints that comparison so the claim is checkable.

**Open questions.** None.

---

## D18 — {_TODAY} — The tolerable-error budget is defined on total error, with contiguity

**Decision.** A magnitude is tolerable when the **total** error (the model's own MAE plus the
perturbation-induced ΔMAE) stays within the threshold. The Δ-only criterion is reported alongside
and labelled. Erosion and dilation are separate budgets, each walked outward from zero so only a
contiguous range of passing magnitudes counts. `downsample` reports the largest passing **factor**
and is never converted to millimetres. When nothing passes, the cell prints `none tolerable`
rather than `0`.

**Alternatives.** (a) Δ-only; (b) total error without the contiguity rule; (c) convert the
downsample factor to mm using the nominal scale.

**Evidence.** **[Exp]** The previous rule tested ΔMAE only, so a model already at 20 mm on the
chest would have been reported as tolerating unlimited boundary error at a 25 mm threshold. It
also took `max(|magnitude|)` over all passing rows, so a non-monotone curve could report k=6 as
tolerable when k=3 failed. A factor is a ratio, not a distance; multiplying it by mm/px produces a
number with no physical meaning.

**Consequences.** The published budgets are smaller and more honest. `max_tolerable_px` is NaN for
downsample rows, with the answer in a `unit` column.

**Open questions.** Whether a real deployment should also budget for *simultaneous* families
rather than one at a time. Not tested.

---

## D19 — {_TODAY} — The training budget is far below the reference, and every flag is provisional

**Decision.** `epochs_b2` and `epochs_prod` stay configurable and the per-epoch validation MAE is
logged. The total optimiser steps for each model are printed in section 11 and beside the B0 flag
table, and the flag table is labelled **PROVISIONAL**.

**Alternatives.** (a) raise the epochs to something closer to the reference; (b) present the flags
as final; (c) drop the flag table.

**Evidence.** **[Ref]** Ruiz et al. (2022) train for 150k iterations at batch 22. This run performs
{min(TRAIN_STEPS.values())}–{max(TRAIN_STEPS.values())} steps. **[Exp]** The training curve shows
whether validation MAE was still falling at the last epoch.

**Consequences.** A flag means "did not beat B0 at this budget", which is a statement about the
run and not about the architecture. Presenting it as final would over-claim.

**Open questions.** Whether the budget is sufficient for the flag table to be treated as final.

---

## D20 — {_TODAY} — Boundary jitter is two-sided, and DataLoader workers get distinct RNG streams

**Decision.** `boundary_jitter` moves the boundary in both directions, by thresholding the signed
distance minus the displaced field. The training augmenter constructs its generator lazily inside
each worker, seeded from the worker id, the epoch and the master seed.

**Alternatives.** (a) keep the seed-based `|= fg` OR; (b) build the generator in the enclosing
scope; (c) build it once per worker from `worker_init_fn` alone.

**Evidence.** **[Exp]** The previous jitter OR-ed the original foreground back in, so no pixel
could ever be removed: it only dilated, and both `np.where` branches were identical, making the
displacement itself a no-op. The sweep's jitter axis was therefore re-measuring dilation. The
assertion over 50 masks now requires `|mean area change| < 0.5%` and that masks both grow and
shrink. **[Exp]** A two-worker DataLoader fed the *same* photo must produce different outputs;
a forked worker inherits a copy of the parent's generator state, so the streams were identical.

**Consequences.** The jitter axis of section 5 measures something new. The augmenter now costs two
distance-transform passes per view per sample, which is why section 6 times it.

**Open questions.** Whether caching the distance transform per sample is worth the memory; the
timing cell prints the budget comparison.

---

## D21 — {_TODAY} — Measurement definitions and citations state only what is verifiable

**Decision.** Every `DEFINITIONS` entry carries one identical statement: a BodyM measurement,
defined as a vertex-path length on an SMPL-registered mesh, with the exact path not published.
`height_cm` is described as coming from `hwg_metadata.csv` with its collection method not stated.
References are restricted to verifiable sources; the unverifiable anthropometric-table citation was
removed and the slice table is labelled an assumption instead. Bland & Altman (1986) and Ramanujan
(1914) are kept under an explicit "foundational-method exceptions" heading because they predate the
2021–2026 open-access preference. A cell regex-checks in-text citations against `references.md` in
both directions.

**Alternatives.** (a) keep the anatomical descriptions as written; (b) keep the pre-2021
citations without a heading; (c) drop the slice-heights table entirely.

**Evidence.** **[Ref]** Ruiz et al. (2022) do not publish per-column vertex paths. **[Exp]** The
citation cross-check cell reports any in-text citation with no reference entry, and any reference
entry never cited.

**Consequences.** The schema map is less informative and more honest. Pattern-engine mapping
still requires the user to supply the anatomical definitions they need.

**Open questions.** **TODO(user)** the pattern engine's own anatomical definitions per key.

---

## D22 — {_TODAY} — `config.yaml` carries the run protocol, and stale overrides are reported

**Decision.** `smoke_test` and `run_final_eval` are config keys. Every downstream cell honours both.
Smoke mode caps subjects, photos per subject, sensitivity subjects and epochs. `run_final_eval:
false` skips section 8 and makes every consumer of its results say "final evaluation not run".
Four runtime keys that were the subject of hard bugs (`backbone`, `img_height`, `img_width`,
`batch_size`) are reported loudly when a pre-existing `config.yaml` overrides them.

**Alternatives.** (a) edit the notebook between runs; (b) separate smoke notebooks;
(c) silently accept config overrides.

**Evidence.** **[Ref]** Brief section 0. **[Exp]** The merge is `{{**DEFAULT_CFG, **cfg}}`, so a
`config.yaml` from an earlier run keeps any key it already contains — which is how the
non-existent backbone name and the transposed image size survived unnoticed.

**Consequences.** The smoke run exercises every assertion in minutes, so a crash is caught before
a multi-hour run. The stale-override banner makes it obvious why a config change had no effect.

**Open questions.** None.
"""
(EXPORT_DIR / "DECISIONS.md").write_text(DECISIONS, encoding="utf-8")
print("written:", EXPORT_DIR / "DECISIONS.md", f"({len(DECISIONS)} chars)")
_entries = [l for l in DECISIONS.splitlines() if l.startswith("## D")]
print(f"  {len(_entries)} dated entries")
# fix D2: no hard-coded run numbers survive in the log
for _forbidden in ("45 silhouettes", "27 silhouettes", "164 cm", "88 cm", "960x720"):
    assert _forbidden not in DECISIONS, f"hard-coded '{_forbidden}' still in DECISIONS.md (fix D2)"