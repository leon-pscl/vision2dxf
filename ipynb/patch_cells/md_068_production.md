### §6 Phase 5 — Production model: variants V-HW and V-H

**WHAT** — Train the same architecture twice on the `fit` split with the Phase-4-derived
augmentation:
- **V-HW** — height **and** weight channels.
- **V-H** — height only. The weight channel is **removed**, not zero-filled.

Select the checkpoint on **validation MAE only**, measured per photo pair. Flag, in the model card,
every measurement where a variant fails to beat B0 on validation.

**WHY**
- [Ref] Brief section 7 requires both variants, and requires the weight channel to be *removed* for
  V-H rather than zero-filled. Zero-filling would be a lie to the network: it would tell the model
  "weight = 0", which is out of distribution, and the network could learn to treat it as a flag
  rather than as missing data. Channel removal is the honest encoding of "not measured".
- [Ref] The input ablation in Ruiz et al. (2022) is the justification for exporting both: weight
  helps girth and little else, so a capture protocol that does not collect weight still gets a
  usable model at a known cost. §10 records which one should be the default as an **open decision
  for the user** — the notebook does not resolve it silently.
- [Ref] Augmentation rationale from brief section 7: deployment masks come from a different
  segmenter than BodyM's DeepLabv3+-derived masks, and the magnitudes come from §5, not guesswork.
- [Exp] Selection on validation MAE only. Test-A and Test-B are touched exactly once, in §8.
- [Exp] **Validation MAE is measured per photo pair** (fix B5), because deployment predicts from
  one pair. The previous subject-averaged selection rewarded models that were good on the mean of a
  subject's photos.
- [Exp] **The scalars demonstrably reach the predictions** (fix B1). The previous design summed a
  zero-initialised projection of the scalar channels at the second convolution — and because a
  *second* zero-initialised projection followed it, both weight matrices received exactly zero
  gradient. Only the bias could learn, and a bias cannot see height or weight. Both variants were
  therefore silhouette-only models, and the ablation this section reports measured nothing. The
  scalars now enter at the head, and two CHECKs pin it: the head's scalar-input weights must be
  non-zero after the first optimiser step, and a +10% stature change must move every key girth
  prediction after training.
- [Exp] **The B0 flag table is PROVISIONAL.** `epochs_prod` is a small fraction of the reference
  training budget, and the total optimiser steps are printed beside the table and in §11. A flag
  means "did not beat B0 at this budget", which is a statement about the run and not about the
  architecture.

**OUTPUT** — `net_vhw`, `net_vh`, `hist_vhw`, `hist_vh`, `FLAGS` DataFrame of
measurements-not-beating-B0, `TRAIN_STEPS`, checkpoints.

**CHECK** — asserts V-H has 3 input channels and V-HW has 4; asserts the augmentation callable is
non-identity, two-sided and per-worker-distinct; asserts both histories contain one row per epoch;
asserts +10% stature moves every key girth for both variants; asserts the flag table's columns and
that every flagged row genuinely fails against B0 in both directions.