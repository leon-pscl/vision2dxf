#### §6.1 Re-run the sweep on the shipped models

**WHAT** — Repeat the exact same perturbation grid and the exact same subject set for `net_vhw`
and `net_vh`, and append the rows to `SENSITIVITY`.

**WHY**
- [Exp] The earlier sweep ran on B2 only. The model card then filtered for V-HW and produced an
  empty table, so the segmentation boundary budget in the shipped documentation was blank. The
  B2 sweep is retained as the *source of the augmentation magnitudes* — it has to precede
  training, because the production model is trained to be robust to exactly the perturbations
  the sweep found tolerable — but the deliverable number for step 3 comes from V-HW.
- [Ref] Brief section 6: the tolerable boundary error is the key deliverable, and brief section
  7 makes V-HW the production architecture. A budget computed on a different model does not
  transfer automatically: the augmented variants are trained *to be* less sensitive, so their
  tolerable error should be at least as large.

**OUTPUT** — `SENSITIVITY` covering `{B1, B2, V-HW, V-H}`, refreshed
`MAX_TOLERABLE_BOUNDARY`, refreshed `sensitivity.csv`.

**CHECK** — assert the V-HW rows exist for every grid point; assert the V-HW tolerable-error
table is non-empty; print the B2-vs-V-HW comparison so the effect of augmentation is visible.