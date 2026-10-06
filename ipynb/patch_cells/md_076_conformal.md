### §7 Phase 6 — Uncertainty: split conformal prediction

**WHAT** — On the `calib` split (never seen in fitting or checkpoint selection), compute absolute
residuals per measurement per variant, and store the conformal quantiles for 80% and 90% target
coverage in `conformal.json`.

**WHY**
- [Ref] Angelopoulos & Bates (2021): split conformal prediction gives **marginal**, not conditional,
  coverage — "over the calibration and test points jointly, the interval contains the truth at least
  80% of the time". It holds under *exchangeability* of calibration and test points and needs no
  distributional assumption.
- [Exp] **The calibration unit is one photo pair per subject** (fix B5, decision U2). The previous
  code averaged a calibration subject's photos before computing residuals, so the scores were
  exchangeable with a quantity the model never produces at inference — averaging several noisy views
  of one body gives a lower-variance estimate than a single capture, and the intervals under-cover
  for exactly that reason. Now one pair is drawn per subject with a fixed seed, and `conformal.json`
  records `unit: photo_pair` and `pairs_per_subject: 1`; a CHECK asserts those fields shipped.
- [Exp] The assumption is what breaks in deployment. Three distinct shifts are visible in this
  project: (a) **segmenter shift** — §5's whole point is that a different segmenter produces different
  boundaries, and residuals from BodyM's DeepLabv3+ masks understate that error; (b) **device/capture
  shift** — a different camera, resolution or lighting changes the nominal scale in §2; (c)
  **population shift** — the BMI tail is thin even in BodyM (§2), and high-BMI bodies are exactly
  where the residuals are largest. So the guarantee is documented as conditional and the model card
  states that intervals **must be recalibrated on tape-measured ground truth from step 10**.
- [Ref] The quantile is the ⌈(n+1)·coverage⌉-th order statistic of absolute residuals, computed with
  the `higher` interpolation so it is a true order statistic and never interpolates below a value
  actually observed in calibration. Using the 80th/90th percentile naively would slightly
  *under*-cover, because it ignores the finite-sample correction that is the entire point.
- [Exp] **`q` is a half-width, so millimetres are `q × 10`, not `q × 20`** (fix B12). The previous
  tables and the §11 summary doubled it, reporting an interval twice as wide as it is. The self-check
  now pins the conversion explicitly.
- [Ref] Brief section 8 requires conformal files per variant, frozen before §8.

**OUTPUT** — `conformal.json`, `CONFORMAL_QUANTILES`, `calibration_photo_pairs.csv`,
empirical-coverage check on the calibration split itself.

**CHECK** — asserts empirical coverage on the calibration set is at least the nominal level (a
finite-sample-corrected quantile must cover there); asserts quantiles are positive and monotone in
coverage; asserts exactly one calibration pair per subject; asserts `unit`/`pairs_per_subject`
shipped into the JSON.