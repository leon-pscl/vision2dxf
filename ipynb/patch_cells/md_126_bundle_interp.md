**Interpretation — what a reviewer should check in this bundle.**

Four checks are automated and run in this notebook's own output. They are the ones that historically
break in production:

1. **ONNX ↔ PyTorch agreement within 1e-4 on 20 real validation tensors.** Not random tensors — real
   masks, with their constant scalar channels. A transposed channel order, a dropped scalar or a
   missing pooling step still "runs" on random input, so random input would have passed a broken
   export. The measured max difference is printed per variant, and the cells assert the tolerance.
2. **The checkpoint round-trips.** Each saved `.pt` is reloaded into a freshly constructed
   `MeasurementNet` and must reproduce the same outputs, and must carry `backbone`, `hidden`,
   `dropout`, `targets`, `use_weight`, `img_height`, `img_width` and `mask_threshold`. `infer.py`
   reads those rather than assuming defaults — it previously fell back to a backbone name that does
   not exist in torchvision.
3. **`preprocess.py`'s self-test against training reference tensors.** It now *recomputes* each
   reference tensor from the saved input masks and scalars and compares with `np.allclose(atol=1e-6)`
   — a full-tensor equivalence check rather than a shape check. It runs in a subprocess with only the
   export directory importable, so nothing can depend on a notebook global.
4. **`infer.py` executed in a subprocess** with only the export directory importable, on a real
   subject, returning all 14 measurements with intervals for both variants. Every interval must nest
   (`lo90 ≤ lo80 ≤ value ≤ hi80 ≤ hi90`), including after clamping.

**The V-HW sensitivity budget is non-empty and is the number step 3 must satisfy.** The earlier
version swept B2 only, so the model card's V-HW filter returned nothing and the segmentation
requirement shipped blank. The published budget now comes from the sweep repeated on the trained
production model.

**Deliberately left incomplete:** the `TODO(user)` markers in `schema_map.yaml`, the plausibility
envelope, and the per-measurement tolerances. Those are pipeline-design decisions, not modelling
ones, and §10 lists them as open. Shipping invented pattern-engine key names would produce a bundle
that passes every automated check and fails at integration.

**Two things a reviewer should read with the budget in mind.** First, the training budget is a small
fraction of the reference, so the B0 flag table is labelled PROVISIONAL: a flag means "did not beat
B0 at this step count", and §11 prints the step count beside it. Second, the licence position: the
data is CC BY-NC 4.0, the AWS registry links to CC BY, and this notebook treats it as non-commercial.
Weights trained on NC data may inherit that restriction — a legal question for the user, not one this
notebook can settle.