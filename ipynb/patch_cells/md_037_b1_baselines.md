"""Markdown for §4.2 B0/B1.

WHAT — State the B0/B1 definitions and the ground rules both obey.

WHY
- [Ref] Brief section 5: B0 is the bar. Stated here so the comparison table in section 8 is
  read correctly.
- [Ref] Ruiz et al. (2022) report the same height+weight input ablation that motivates
  exporting both a height-only and a height+weight variant in section 6.
- [Exp] Two corrections in this section are load-bearing for every B1 number:
  - **Slice rows were vertically mirrored.** `_row_for` returned `y1 - (1 - f) * span`, so
    f = 0 (the floor) mapped to the *top* image row and f = 1 (the crown) to the bottom. Every
    slice was reading the wrong height: the chest slice was reading upper-thigh height and the
    ankle slice was reading the crown. The corrected mapping is `row = y1 - f * span`, and the
    check below asserts that the ankle row index exceeds the chest row index, which is only true
    if image row indices grow downward and f is measured from the floor.
  - **Slice widths included arms and both legs.** Summing every foreground pixel in a row gave
    arm-tip-to-arm-tip at chest height and outer-thigh-to-outer-thigh at thigh height, so every
    derived girth was systematically too large. The corrected measurement takes the single
    connected foreground run containing the body midline column; for thigh, calf, knee and
    ankle it takes the one leg run nearest the midline instead.

CHECK — the overlay figure is saved and displayed; `ankle_row > chest_row` is asserted; the
derived/unsupported partition is asserted.
"""

### §4.2 Baselines B0 and B1

**WHAT** — Two non-learned-from-images baselines:
- **B0 (non-visual)** — ridge and gradient boosting predicting all 14 measurements from
  `height_cm`, `weight_kg` and sex. *This is the bar every vision model must beat.*
- **B1 (geometric)** — per-slice ellipse model: scale = `height_cm` / person pixel height;
  horizontal slices at documented heuristic heights; the torso foreground run (or one leg run)
  as the front semi-axis and the side run as the depth; Ramanujan perimeter for girths; true
  row differences × scale for lengths; one per-measurement linear calibration fitted on the
  **fit split only**, on that measurement's own features.

**WHY**
- [Ref] Brief section 5: B0 is the bar. Stated here so the comparison table in §8 is read
  correctly.
- [Ref] Ruiz et al. (2022) report the same height+weight input ablation that motivates
  exporting both a height-only and a height+weight variant in §6.
- [Assumption] B1's slice heights are proportional anthropometric heuristics, **not** BodyM
  ground truth, and no published source is claimed for them. *Tested by* the per-measurement
  calibration residual printed below: a badly placed slice shows up as a systematically wrong
  girth whose error is worst where body-shape variance is highest.
- [Assumption] The person's pixel height in the mask equals stature. *Tested by* §2's mm/pixel
  spread. In deployment this assumption is owned by step 4 (scale/perspective calibration), not
  by this model — which is precisely why §5 measures sensitivity to it.
- [Exp] Measurements that are lengths rather than girths (`leg-length`, `shoulder-to-crotch`,
  `arm-length`) are derived as row differences, not guessed. `height` is a model *input*, so B1
  echoes it back and §6 excludes it from "beats B0" comparisons. The implementation asserts
  that derived ∪ unsupported partitions the 14 targets, so nothing is silently missing.
- [Exp] Each measurement's calibration is restricted to its own features (fix B4). Feeding every
  feature to every measurement let the ridge borrow, say, the chest perimeter to predict the
  wrist, which is how B1 quietly converged on B0 while appearing to be a shape model. The
  printed coefficient probe now shows which geometry each prediction actually leans on.

**OUTPUT** — `src/bodym_baselines.py`, fitted `b0`, `b1`, `metrics_B0_*`, `metrics_B1`,
`artifacts/fig_b1_slices.png`, per-measurement calibration coefficient probe.

**CHECK** — assert the derived/unsupported partition; assert the fitted calibration predicts
finite values with the right shape; assert `ankle_row > chest_row` in image coordinates;
display the slice overlay showing every measured foreground run.