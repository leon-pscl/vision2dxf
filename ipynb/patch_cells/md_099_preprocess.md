### §9.2 `preprocess.py` — the canonical preprocessing, with a unit test

**WHAT** — Emit a standalone `preprocess.py` into the bundle whose `silhouette_to_tensor` wraps the
*training* implementation, then run its self-test against reference tensors saved from the exact
tensors the training pipeline produced.

**WHY**
- [Ref] Brief section 11 requires `silhouette_to_tensor(front_mask, side_mask, height_cm,
  weight_kg=None)` and "include a unit test against saved reference tensors."
- [Exp] The module **imports** `to_tensor` from `bodym_cnn.py` rather than reimplementing
  binarisation, cropping, aspect handling, resizing and channel construction. A reimplementation would
  be a second source of truth that drifts silently; a wrapper cannot. The bundle ships `bodym_cnn.py`
  alongside so the import resolves offline.
- [Exp] **The self-test is a full-tensor equivalence check** (fix D1). The previous version built a
  synthetic rectangle and compared only shape, dtype, the mask range and the two normalisation
  constants. That is a contract test: it cannot detect a changed resize filter, an inverted mask
  channel, or a swapped pair order — any of which yields plausible-but-wrong measurements in
  production. §9.1 now saves the reference **input masks and scalars** next to the tensors, and the
  self-test recomputes from those and asserts `np.allclose(..., atol=1e-6)`. It also asserts each
  scalar channel is constant across the image, which catches a channel built from image data.
- [Exp] Image size and threshold are read from `config.yaml` at import time rather than baked in, so
  there is exactly one place that defines them.

**OUTPUT** — `preprocess.py` in the bundle, `reference_tensors.npz`, `reference_inputs.npz`,
`reference_meta.json`.

**CHECK** — the module's `__main__` self-test runs in a subprocess and must print PASS; it compares
the channel count, the constant-channel property and the full tensor against saved references at
1e-6.