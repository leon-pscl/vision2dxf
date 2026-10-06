require("p05_b1")

def _metrics_block(split: str, model: str = "V-HW") -> str:
    """Render one markdown table of test metrics.

    Parameters
    ----------
    split : {'testA', 'testB'}
        Test split name.
    model : str, default 'V-HW'
        Model name.

    Returns
    -------
    str
        Markdown table, or a "not run" notice when the final evaluation was skipped.
    """
    if FINAL_COMPARISON is None:
        return ("_Final evaluation not run (`config.yaml: run_final_eval: false`). "
                "Set it to true and Save & Run All once to populate this section._")
    t = RESULTS[split]["metrics"][model][[
        "measurement", "n", "mae_mm", "bias_mm", "tp50_mm", "tp75_mm", "tp90_mm",
        "ba_mean_diff_mm", "ba_loa_lo_mm", "ba_loa_hi_mm"] +
        [f"pct_within_{int(t)}mm" for t in cfg["tolerance_mm"]] +
        ["coverage_80", "coverage_90", "width_90_mm"]]
    head = "| " + " | ".join(t.columns) + " |"
    sep = "|" + "|".join(["---"] * len(t.columns)) + "|"
    body = ["| " + " | ".join(
        (f"{v:.2f}" if isinstance(v, (float, np.floating)) else str(v))
        for v in row) + " |" for row in t.itertuples(index=False)]
    return "\n".join([head, sep] + body)


def _markdown_table(tbl: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table.

    Written by hand rather than via ``DataFrame.to_markdown`` because that method requires the
    optional ``tabulate`` package, which is not guaranteed to be present on a Kaggle image. The
    model card is a required deliverable, so its formatting must not depend on an undeclared
    dependency.

    Parameters
    ----------
    tbl : DataFrame
        Table to render.

    Returns
    -------
    str
        Markdown table with a header row and a separator row.
    """
    def fmt(v) -> str:
        if isinstance(v, (float, np.floating)):
            return "n/a" if pd.isna(v) else f"{v:.3f}"
        return str(v)

    head = "| " + " | ".join(str(c) for c in tbl.columns) + " |"
    sep = "|" + "|".join("---" for _ in tbl.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in tbl.itertuples(index=False)]
    return "\n".join([head, sep] + body)


# fix A5: df.groupby("stratum").n() raises TypeError ('SeriesGroupBy' object is not callable).
# Use bracket indexing, and build the dict in a variable rather than a dict/set literal inside
# an f-string expression.
strata_lines = []
if RESULTS:
    for split in ("testA", "testB"):
        for vname in ("V-HW", "V-H"):
            for key, label in (("", "sex"), ("_bmi", "BMI band")):
                t = RESULTS[split]["stratified"][vname + key]
                n_by = t.groupby("stratum")["n"].first().to_dict()
                rel = t.groupby("stratum")["reliable"].first().to_dict()
                rel_map = {str(k): bool(v) for k, v in rel.items()}
                strata_lines.append(
                    f"- **{split} / {vname} / by {label}** — n per stratum: {n_by}; "
                    f"reliable (n >= {cfg['min_stratum_n']}): {rel_map}")
else:
    strata_lines.append("_Final evaluation not run (`run_final_eval: false`); no strata to "
                        "report._")

# fix B9: the sensitivity table the model card publishes must be V-HW's, and it must not be empty.
_VHW_BUDGET = MAX_TOLERABLE_BOUNDARY[(MAX_TOLERABLE_BOUNDARY.model == "V-HW")
                                     & (MAX_TOLERABLE_BOUNDARY.criterion == cfg["max_tolerable"])]
assert not _VHW_BUDGET.empty, (
    "the model card's segmentation budget is empty: no V-HW sensitivity rows (fix B9)")
sensitivity_md = _markdown_table(
    _VHW_BUDGET[["measurement", "family", "side", "tolerance_mm",
                 "max_tolerable_px", "max_tolerable_mm", "unit"]])
_BUDGET_LABEL = (f"total error (baseline MAE + dMAE) <= tolerance"
                 if cfg["max_tolerable"] == "total" else "dMAE only <= tolerance")

flag_md = (M.format_metric_table(FLAGS) if len(FLAGS)
           else "_No measurement failed to beat B0 on validation._")

# fix A9: `{{'80%': ...}}` inside an f-string expression is a SET containing a DICT, which is a
# TypeError, and the doubled braces are a SyntaxError at compile time in this position. Build
# the dict as a variable, dump it, and interpolate the string.
if COVERAGE_SUMMARY is not None:
    coverage_dict = {f"{r['split']} {r['variant']} (per photo pair, n={r['n']})":
                     {"80%": round(r["empirical_80"], 1),
                      "90%": round(r["empirical_90"], 1),
                      "mean width 90 (mm)": round(r["mean_width_90_mm"], 1)}
                     for r in COVERAGE_SUMMARY.to_dict("records")}
    coverage_md = json.dumps(coverage_dict, indent=2)
else:
    coverage_md = ("_Final evaluation not run (`run_final_eval: false`); held-out coverage "
                   "cannot be estimated until the single evaluation pass is done._")

# fix D2: no hard-coded run results. Everything below is interpolated from objects computed
# in this run.
_pps_note = photos_per_subject_note(IDX["train"], ids_val)
_height_rng = (INTEGRITY_INDEX["train"].subjects.height_cm.min(),
               INTEGRITY_INDEX["train"].subjects.height_cm.max())

MODEL_CARD = f"""# Model Card — BodyM Measurement Estimator

## Purpose and scope
Estimates {len(TARGETS)} body measurements in cm from a front silhouette, a side silhouette and
stature (optionally mass), for step 6 of a 10-step made-to-measure pattern pipeline. Each
measurement is returned with 80% and 90% split-conformal prediction intervals.

**Evaluation unit: one photo pair.** That is what `infer.py` receives. Subject-averaged metrics
exist as a clearly-labelled secondary table (`subject_averaged_comparison.csv`) and are *not*
the headline number.

**No ISO 20685 compliance is claimed or implied.** Errors are reported against the reporting
thresholds of {cfg["tolerance_mm"]} mm only.

## Training data
- **Dataset**: BodyM (Ruiz et al., 2022). Binary silhouettes, height, weight, sex and
  {len(MEASUREMENT_COLUMNS)} mesh-derived measurements. **No RGB images** — no segmentation model
  was trained or fine-tuned here.
- **Split used for training**: `fit` = {len(ids_fit)} subjects (BodyM `train` split only),
  stratified by sex x BMI band, selected by **subject id** so that a subject's multiple
  silhouettes can never appear in two splits.
- **Validation**: `val` = {len(ids_val)} subjects ({_pps_note}). Checkpoint selection used
  validation MAE only.
- **Conformal calibration**: `calib` = {len(ids_calib)} subjects, never seen in fitting, and
  **one photo pair per subject** (`unit: photo_pair`, `pairs_per_subject: 1`).
- **Held out**: Test-A and Test-B were evaluated **exactly once**, in section 8.
  {'Both were evaluated in this run.' if RUN_FINAL_EVAL else '**Not evaluated in this run** (`run_final_eval: false`).'}
- **Population** (train): {int(EDA['sex_counts'].loc['female', 'train'])} female /
  {int(EDA['sex_counts'].loc['male', 'train'])} male; BMI band counts
  {EDA['bmi_bands']['train'].to_dict()}.

## Training budget — read this before trusting any comparison
| model | epochs | optimiser steps | best val MAE |
|---|---|---|---|
| B2 | {EPOCHS_B2} | {TRAIN_STEPS['B2']} | {hist_b2.val_mae_mm.min():.2f} mm |
| V-HW | {EPOCHS_PROD} | {TRAIN_STEPS['V-HW']} | {hist_vhw.val_mae_mm.min():.2f} mm |
| V-H | {EPOCHS_PROD} | {TRAIN_STEPS['V-H']} | {hist_vh.val_mae_mm.min():.2f} mm |

Ruiz et al. (2022) train for 150k iterations at batch 22. This run is
{min(TRAIN_STEPS.values())/150000:.2%} of that. **Every "beats B0" statement below is
PROVISIONAL at this budget**: a flag means "did not beat B0 here", not "cannot beat B0".
Increase `epochs_b2` / `epochs_prod` in `config.yaml` before treating any flag as final.

## Licence
**CC BY-NC 4.0 — non-commercial use only.** The AWS registry entry for BodyM links to the CC BY
legal code, which is inconsistent with the CC BY-NC terms the dataset is released under. This
inconsistency is recorded here and the data is treated as **non-commercial** throughout. Any
commercial deployment requires written clarification from the rights holders.

## Input contract
| field | type | unit | required | notes |
|---|---|---|---|---|
| front_mask | ndarray (H, W) | binary | yes | values > {cfg["mask_threshold"]} are foreground |
| side_mask | ndarray (H, W) | binary | yes | same person, side view |
| height_cm | float | cm | yes | from `hwg_metadata.csv`; collection method not stated by the dataset |
| weight_kg | float | kg | no | when supplied, the V-HW variant is used automatically |

Output: `{{measurement: {{value_cm, lo80, hi80, lo90, hi90, clamped}}}}`. `clamped` is true when
the plausibility envelope truncated that measurement's interval; the point estimate is always
kept inside its own interval.

## Intended use
Estimating body measurements for made-to-measure pattern drafting, where the inputs come from a
controlled capture (step 2) and a person-segmentation model of the quality characterised in the
sensitivity section below.

## Out-of-scope use
- Any clinical, medical, ergonomic-safety, sizing-for-occupation or identity purpose.
- Populations outside the training distribution — notably high BMI (see failure regimes).
- Subjects shorter or taller than the BodyM training height range ({_height_rng[0]:.0f}–{_height_rng[1]:.0f} cm).
- Commercial use, under the licence above.
- Using the conformal intervals as *measurement* uncertainty: they are calibrated against
  mesh-derived ground truth and must be recalibrated on tape measurements (pipeline step 10).

## Final metrics — V-HW, per photo pair
### Test-A
{_metrics_block('testA')}

### Test-B
{_metrics_block('testB')}

### Conformal coverage (empirical, held out)
{coverage_md}

## Metrics stratified by sex and BMI band
{chr(10).join(strata_lines)}

Strata with n < {cfg['min_stratum_n']} are printed for completeness and marked unreliable. They
must not be quoted as accuracy figures.

## Measurements that do not beat the non-visual baseline
B0 (ridge/gradient boosting on height + weight + sex) is the bar a vision model must clear.
These measurements did not clear it on validation for at least one variant, **at the training
budget stated above**:

{flag_md}

## Known failure regimes
- **High BMI.** The training population thins sharply at the top of the BMI range. Measured MAE
  by band is in `bmi_stratum_errors.csv` and `fig_bmi_failure_mode.png`. Outside the training
  range the model extrapolates and the prediction intervals are the only safeguard.
- **Sparse strata.** Sex-stratified numbers on Test-A rest on the printed per-stratum `n`; read
  the `reliable` flag before quoting any of them.
- **Degraded capture.** The Test-A to Test-B gap in section 8 quantifies sensitivity to
  uncontrolled photography.
- **Height extrapolation.** Outside {_height_rng[0]:.0f}–{_height_rng[1]:.0f} cm.
- **Segmenter change.** See the sensitivity results below.
- **Pose / capture-condition variation.** Each calibration score is one photo pair, so the
  intervals reflect single-capture error, not a subject average.

## Segmentation boundary sensitivity (requirement on pipeline step 3)
Nominal scale used for px → mm: **{MM_PER_PX_NOMINAL:.3f} mm/px**
(IQR {np.percentile(EDA['mm_per_px'], 25):.3f}–{np.percentile(EDA['mm_per_px'], 75):.3f} mm/px).
Criterion: **{_BUDGET_LABEL}**. Erosion and dilation are separate budgets. `downsample` rows
report a **factor**, never millimetres — a resolution ratio is not a distance.

Maximum tolerable boundary error for **V-HW**:

{sensitivity_md}

Full sweep: `sensitivity.csv` (models B1, B2, V-HW, V-H), `max_tolerable_boundary_error.csv`,
`fig_sensitivity_morph.png`, `fig_sensitivity_other.png`.

## Uncertainty guarantee and its limits
Split conformal prediction per measurement, calibrated on **one photo pair per calibration
subject**: **marginal** coverage over calibration and test pairs jointly, conditional on
exchangeability of the two samples. It is **not** conditional on an individual subject or an
individual capture, so a specific subject may fall outside its 90% interval.

The guarantee weakens, and should be assumed invalid, under:
1. **Segmenter shift** — the quantiles were fitted on BodyM's DeepLabv3+-derived masks; deployment
   masks come from a different model.
2. **Device / capture shift** — a different camera, resolution or distance changes the nominal
   mm/px scale, biasing every measurement.
3. **Population shift** — bodies outside the training BMI/height range.
4. **Ground-truth shift** — the targets are SMPL mesh vertex-path lengths, not tape measurements.

**These intervals must be recalibrated on tape-measured ground truth from pipeline step 10 before
they are used for plausibility gating in production.**

## Reproducibility
- Master seed: {cfg["seed"]}; split assignments in `splits_train.csv`; the chosen calibration
  pair per subject in `calibration_photo_pairs.csv`.
- `config.yaml` holds every hyperparameter. Environment: Python {ENV['python']},
  torch {ENV['torch']}, torchvision {ENV['torchvision']}, GPU
  {ENV['gpu']['name'] or 'none'}.
- Full environment log, decision log and references: `DECISIONS.md`, `references.md`,
  `requirements.txt`.

## Open items for the user
1. Whether the capture protocol (step 2) collects weight — this determines the default variant.
2. Pattern-engine key names in `schema_map.yaml` (currently `TODO(user)`).
3. Which tolerance threshold the pattern engine accepts per measurement.
4. The plausibility envelope in `infer.py` / `conformal.json` (`TODO(user)`).
5. Commercial licensing clearance, given the registry inconsistency above.
6. Whether the training budget is sufficient before treating the B0 flag table as final.
"""
(EXPORT_DIR / "model_card.md").write_text(MODEL_CARD, encoding="utf-8")
print("written:", EXPORT_DIR / "model_card.md", f"({len(MODEL_CARD)} chars)")
_sens_rows = len([l for l in sensitivity_md.splitlines() if l.startswith("|")]) - 1
print(f"  sensitivity table: {_sens_rows} V-HW budget rows, criterion '{_BUDGET_LABEL}'")
assert _sens_rows > 0, "the model card's sensitivity table is empty (fix B9)"
print("mark_phase('model card') =", mark_phase("model card"), "s")