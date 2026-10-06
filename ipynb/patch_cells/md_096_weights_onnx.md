#### §9.1 Weights, ONNX, and the equivalence check

**WHAT** — Save both variants as PyTorch checkpoints carrying everything needed to rebuild the
architecture, export both to ONNX with a dynamic batch axis, and verify the ONNX graph against
the PyTorch model on 20 real validation tensors.

**WHY**
- [Ref] Brief section 11 requires `model_vhw.onnx`, `model_vh.onnx`, plus PyTorch `state_dict`s,
  with ONNX↔PyTorch agreement within 1e-4 on 20 samples.
- [Exp] The previous version wrote no weights at all. `infer.py` was emitted into the bundle and
  then `REQUIRED_EXPORTS` asserted on `model_vhw.onnx`, so the export section could not complete
  and `infer.py` had nothing to load.
- [Exp] The check uses **real validation tensors**, not random input. A transposed channel order,
  a dropped scalar, or a missing pooling step still "runs" on random input — it only diverges on
  real masks, because random input has no constant scalar channels and no silhouette structure
  for the resize to act on. Random input would have passed a broken export.
- [Exp] The checkpoint stores `backbone`, `hidden`, `dropout`, `use_weight` and `targets`, and
  `infer.py` reads them rather than assuming defaults. It previously fell back to a hard-coded
  backbone name that no longer exists.

**OUTPUT** — `export/model_vhw.pt`, `export/model_vh.pt`, `export/model_vhw.onnx`,
`export/model_vh.onnx`, `EXPORT_VERIFY` (DataFrame).

**CHECK** — assert `max |onnx − torch| < cfg["onnx_atol"]` for both variants, printing the
value; assert the saved state dict reloads into a fresh `MeasurementNet` and reproduces the
same output; assert all four files exist.