#### §6.2 Evaluate both production variants

**WHAT** — Evaluate V-HW and V-H on validation with the full metric suite, **per photo pair**, then
compare against B0 and B1 and produce the flag table required by brief §7.

**WHY**
- [Ref] Brief section 7: *"For any measurement where the variant does not beat B0 on validation,
  flag it in the model card. Do not hide or drop it."* The flag table below is that content,
  computed rather than asserted, and it feeds §9's model card directly.
- [Exp] The flag table is verified in both directions: every flagged row must have a variant whose
  MAE is at or above B0's, and every unflagged row must have both variants strictly below. The
  previous assertion was `... or True`, which asserted nothing at all.
- [Exp] The V-HW vs V-H difference per measurement is the empirical answer to the open decision in
  §10 about whether the capture protocol must collect weight.
- [Exp] A subject-averaged table is printed alongside and labelled **secondary**, with the
  difference between it and the per-pair table shown explicitly. That difference is the optimism
  per-pair evaluation removes; quoting the averaged figure would understate the error a single
  capture actually sees.

**OUTPUT** — `metrics_VHW`, `metrics_VH`, `FLAGS`, `SCALAR_DELTA`, `SUBJECT_AVG_*`,
`variant_comparison` figure and table.

**CHECK** — asserts the flag table covers all 14 measurements in both directions; asserts the
scalar sensitivity of both variants; asserts subject-averaged metrics have the expected shape.