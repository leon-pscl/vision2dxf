### §5 Phase 4 — Boundary-sensitivity experiment (key deliverable)

**Purpose.** Derive the boundary accuracy that **step 3 (segmentation) must achieve** for this
measurement model to remain usable. This is the deliverable that couples two pipeline steps: it
tells the segmentation team how many pixels of silhouette-boundary error are tolerable before the
measurements stop being pattern-grade.

**WHAT** — On validation silhouettes, apply **independently to front and side**:
- morphological erosion / dilation for k = −6 … +6 px (negative shrinks the person, positive grows);
- random boundary jitter;
- downsample → upsample at factors 2, 4, 8 (low-resolution prototype masks).

Re-run **B1** and **B2** under each perturbation, plot ΔMAE per measurement against perturbation
size, and report the maximum tolerable boundary error in px *and* mm for chest, waist and hip at
tolerance thresholds of 9, 15 and 25 mm.

**WHY**
- [Ref] Brief section 6 makes this the key deliverable and prescribes the perturbation families.
  The three families are complementary: systematic erosion/dilation models a *bias* in the
  segmenter's output, jitter models *roughness*, and downsample → upsample models *resolution
  loss* — which is what a prototype capture at low resolution actually produces.
- [Ref] Ruiz et al. (2022) train with adversarial silhouette augmentation precisely because
  "in-the-wild" masks are imperfect; the magnitudes they use are not derived from measurement
  error, so deriving them here is the contribution.
- [Assumption] The mm/px scale used to convert pixels to millimetres is the nominal
  `height_cm / pixel height` from §2. *Tested by* the printed IQR of that scale: if it is wide,
  the mm figures carry that much extra uncertainty and the px figures are the trustworthy ones.
- [Exp] Perturbations are applied to **validation** silhouettes only. Applying them to Test-A or
  Test-B would consume the single-use evaluation budget (§8) on a diagnostic, which brief §1
  forbids.
- [Exp] **The jitter family is now genuinely two-sided** (fix B6). The previous implementation
  OR-ed the original foreground back into its output, so no pixel could ever be removed: it only
  dilated. Both `np.where` branches were also identical, so the displacement itself was a no-op
  before the OR. The jitter axis was therefore re-measuring dilation. The new implementation
  thresholds the signed distance minus the displaced field, and the CHECK requires that masks both
  grow and shrink and that the mean area change is below 0.5%.
- [Exp] **The sweep is a per-photo-pair measurement** (fix B5), because deployment predicts from
  one pair. Averaging a subject's photos before scoring would understate the error, and would do
  so unevenly across splits with different photos-per-subject counts.
- [Exp] **The harness self-check is no longer circular** (fix B7). ΔMAE at `('none', 0)` is zero by
  construction, because ΔMAE is the difference from that very row. The replacement computes the
  unperturbed MAE twice by independent code paths — once through the sweep, once through a plain
  `DataLoader` over the same subjects — and asserts they agree to 1e-6. `morph k=0` is asserted
  equal to `none` as well.

**OUTPUT** — `sensitivity.csv`, `MAX_TOLERABLE_BOUNDARY` DataFrame,
`fig_sensitivity_*.png`, augmentation magnitudes for §6.

**CHECK** — asserts the CSV covers every (model × perturbation family × measurement) combination;
asserts the identity condition matches an independent recomputation within 1e-6; asserts
`morph k=0` equals the identity condition.