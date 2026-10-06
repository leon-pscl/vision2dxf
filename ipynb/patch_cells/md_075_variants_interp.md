**Interpretation.** Three results to read here.

**1. The flag table is a correctness check on the whole exercise — and a statement about this run.**
A measurement where V-HW does not beat B0 means the silhouette added nothing beyond height, weight
and sex: the CNN is redrawing the non-visual baseline with extra steps. That belongs in the model
card as a stated limit, and step 7 should be told that for that measurement it is better off using
the non-visual formula. But read it against the printed step count. At a small fraction of the
reference training budget, "did not beat B0" is a fact about the run, not about the architecture —
which is why the table is labelled PROVISIONAL and why the training curve matters: if validation MAE
was still falling at the last epoch, the correct response is more epochs in `config.yaml`, not a
claim that the measurement is unrecoverable from a silhouette.

**2. `weight_gain_mm` is the input ablation, measured here rather than cited — and it is only
meaningful now.** Positive means the weight channel helps. Read it split by `kind`: if girths gain
and lengths do not, then weight is carrying body *mass* information the outline cannot express, and
the capture protocol's weight measurement is what buys accuracy on the girths. That is the concrete
evidence behind the open decision in §10. Before fix B1 this number was meaningless: the scalar path
never received gradient, so V-HW and V-H were the same model with different channel counts and the
difference was noise. The `SCALAR_DELTA` block printed above is the direct check that the scalars now
reach the predictions at all.

**3. The per-pair versus subject-averaged gap is the size of the optimism removed.** The secondary
table shows what the previous version reported. Where the two agree, averaging was not doing much;
where they differ, per-subject averaging was hiding real single-capture error, and any downstream
plausibility gate tuned on the averaged number would be too lenient.

**4. The B1 column is a sanity check on the geometry.** If B1 tracks B0 closely on the lengths, B1
is mostly re-deriving stature and the interesting content is in the girths. If B1 is uniformly
terrible, the slice heuristics are misplaced and B1 should be treated as a negative result in
DECISIONS.md rather than quietly omitted.