**Interpretation — the deliverable.**

**Read the table as a specification for step 3, and read the criterion column first.** A magnitude
is tolerable when the **total** error — the model's own baseline MAE plus the perturbation-induced
ΔMAE — stays within the threshold. The Δ-only figure is published alongside and labelled, because
the two answer different questions: "total" is what a caller would actually observe, while "delta"
is the error attributable to the segmentation. A model already at 20 mm on the chest would, under a
Δ-only rule, be reported as tolerating unlimited boundary error at a 25 mm threshold.

**Erosion and dilation are separate rows.** A segmenter biased tight and one biased loose are
different failure modes with different consequences, and the sweep is not symmetric in them. Each
side is walked outward from zero and stops at the first failure, so only a *contiguous* range of
passing magnitudes counts. A non-monotone curve cannot produce a larger budget by skipping its own
interior failures.

**`downsample` rows report a factor, and their px/mm columns are empty.** A resolution ratio is not
a distance. Multiplying a factor by mm/px produces a number with no physical meaning, so it is not
reported. Where nothing passes at a threshold the cell prints `none tolerable` rather than `0`,
because `0` is indistinguishable from "only the identity condition passed".

**The B1-vs-B2 asymmetry is the interesting part.** B1 re-fits its calibration under every
perturbation, so a *global* boundary bias (uniform erosion) is largely absorbed: what survives is
only the local change. B2 has no such mechanism, so it degrades on both. That asymmetry tells you
something actionable — B1's per-measurement calibration does real work against systematic scale
error, and if step 3's segmenter has a systematic bias, step 4's calibration stage can recover part
of it. Where a CNN's tolerable error is much smaller than B1's, the CNN has learned absolute
silhouette shape rather than a scale-normalised abstraction, which is the more fragile of the two
under mask-quality change.

**Why the magnitudes for §6 are what they are.** `AUG_EROSION_PX`, `AUG_DILATION_PX` and
`AUG_JITTER_PX` are the largest magnitudes at which **every** measurement except `height` (an input
passthrough, not a prediction) still passes the loosest threshold under the total-error criterion,
walked contiguously from zero. The cell prints the **binding measurement** — the first one to
fail — so the rule is checkable rather than asserted. The previous rule required only *any one*
measurement to pass while the text claimed *all* did; one easy measurement could therefore carry a
magnitude that eleven others fail.

`AUG_DOWNSAMPLE` is capped at the smallest swept factor: downsample→upsample at the largest factor
is a catastrophic mask, and training on it would teach the network to expect holes it will never
see in deployment.

**One caveat to carry forward.** These tolerances are conditional on the nominal scale holding. If
step 4's calibration is off by ±X%, every millimetre figure scales by (1 ± X) while the pixel
figures do not. Both are in the CSV so the downstream step can pick the right unit.

**And the budget that ships is V-HW's, not B2's** — see §6.1. The table above is computed from the
B2 sweep because that sweep has to run first; the augmentation magnitudes come from it, and the
model card's published requirement comes from the sweep repeated on the trained production model.