**Interpretation — how to read the Test-A vs Test-B gap.**

Test-B photographs were taken under less controlled conditions (Ruiz et al., 2022), and §2 printed
how many photos per subject each split carries. So a **degradation from Test-A to Test-B is the
expected result, not a red flag** — and its size is the most useful number in this notebook for step
2 of the pipeline, because it converts "capture quality matters" into a budget. Because evaluation is
now per photo pair, that gap is no longer confounded by a difference in how many photos were averaged.

Three distinct causes are separable in the printed tables:

- **Scale drift.** If the mean bias shifts coherently between A and B on the girths while TP50 barely
  moves, the nominal mm/px scale drifted (different camera distance, lens or resolution). That is
  step 4's calibration problem, and it shows up as bias, not as spread.
- **Boundary noise.** If TP90 and the Bland–Altman limits widen while the bias stays near zero, the
  *shape* information degraded — worse segmentation or more pose variation. That is step 3's problem,
  and it is what §5's boundary-sensitivity curve predicts.
- **Population shift.** If the gap is concentrated in the high-BMI strata, the two splits simply cover
  different populations. Check the per-stratum `n` before attributing anything to the model.

Note also that B0 will degrade too if the problem is population shift, and will degrade **not at all**
if the problem is segmentation — because B0 never looks at an image. Comparing B0's A-vs-B gap against
V-HW's is therefore the cleanest way to attribute the difference.

**The coverage table is the honest check on §7.** Empirical coverage on a genuinely held-out split is
the real test of the conformal claim, and it will not equal nominal exactly — `n` is small and coverage
is estimated per measurement then averaged, so a few points of noise is expected. If empirical coverage
sits *well below* nominal on one split, the assumption that has failed is exchangeability, and §7's
`assumptions_and_breaks` list is where to look first. Remember what the guarantee covers: pairs
jointly, not subjects individually, and not conditional on any particular body.