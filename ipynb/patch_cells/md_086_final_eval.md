### §8 Phase 7 — Final evaluation on Test-A and Test-B (run **once**)

> **Single-use protocol.** Test-A and Test-B are evaluated here and nowhere else. No model
> selection, hyperparameter, augmentation magnitude, calibration or threshold in this notebook was
> chosen using them. Everything above this line used `train` only.
>
> **Skip switch.** With `config.yaml: run_final_eval: false` this section prints a notice and every
> consumer of its results reports "final evaluation not run" rather than failing — including the
> model card, `schema_map.yaml`, the summary figure and §11. The `infer.py` smoke test in §9.3 also
> falls back to a *validation* subject in that mode, so a smoke run never spends the single-use
> budget.

**WHAT** — With V-HW, V-H and their conformal files frozen, evaluate B0, B1, V-HW and V-H on Test-A
and Test-B with the full §4.1 metric suite, **per photo pair** for the vision models, stratified by
sex and by BMI band with `n` and reliability flags. Present all four in one comparison table, then
interpret the Test-A vs Test-B gap.

**WHY**
- [Ref] Brief section 10: interpret the two test sets together. Ruiz et al. (2022) describe Test-B
  photographs as taken under less controlled conditions, so the A-to-B degradation **is** the estimate
  of capture-condition sensitivity — a measurement, not a defect.
- [Ref] Brief section 1: *"Never claim ISO 20685 compliance. Report errors against tolerance thresholds
  only."* Every table below reports pass rates against 9/15/25 mm and states `n`; none reports a
  compliance claim.
- [Exp] **Evaluation is per photo pair** (fix B5). The previous version averaged each subject's photos.
  Because Test-A has far more photos per subject than Test-B, that made the A-to-B gap partly a
  measurement of the averaging count rather than of capture quality — which is precisely the
  quantity this section exists to estimate.
- [Exp] Stratification is reporting-only. Selection was already done on validation, so quoting thin
  strata cannot be a selection leak — but marking `n < 30` unreliable keeps a 6-subject BMI>40 stratum
  from being read as evidence.
- [Exp] `df.groupby("stratum").n()` raises `TypeError: 'SeriesGroupBy' object is not callable` on any
  pandas where the frame has an `n` column (fix A5), so the per-stratum counts use bracket indexing.

**OUTPUT** — `RESULTS` (per split: predictions, metric tables, stratified tables, subject-averaged
secondary tables, stratum labels), `FINAL_COMPARISON`, `COVERAGE_SUMMARY`,
`SUBJECT_AVG_COMPARISON`, `BMI_ERRS`, `fig_testA_vs_testB.png`.

**CHECK** — asserts no test subject id appears in any training split, printed immediately before the
results; asserts every split has all 14 measurements covered; asserts coverage columns exist for both
variants; asserts `FINAL_COMPARISON` records the evaluation unit per row.