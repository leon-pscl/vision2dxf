### §9 Phase 8 — Export bundle (`/kaggle/working/export/`)

> **The bundle is `/kaggle/working/export.zip`.** Kaggle's `/kaggle/outputs` directory is only
> populated when you Save Version; during a Run All the zip sits in the working directory and must be
> downloaded from the notebook's output pane.

**WHAT** — Write the weights (PyTorch + ONNX), the preprocessing and inference entry points, the
conformal file, the schema map, and the five documentation artefacts, all generated from the tables
this run computed. Then verify every required file exists, write a manifest, and zip.

**WHY**
- [Ref] Brief section 11 lists the required contents; the `REQUIRED_EXPORTS` assertion turns that
  list into an executable check, so a missing file fails the run rather than being discovered by
  whoever downloads the bundle.
- [Exp] **Weights are actually saved.** The previous version emitted `infer.py` and then asserted on
  `model_vhw.onnx`, which had never been written — so the export section could not complete and
  `infer.py` had nothing to load. §9.1 writes both variants with a self-describing checkpoint and
  verifies ONNX against PyTorch on **real validation tensors**.
- [Exp] **The preprocessing self-test now compares the full tensor** (fix D1). It previously built a
  synthetic rectangle and checked shape, dtype and the scalar constants — a contract test, not an
  equivalence test. A changed resize filter or an inverted mask channel passes that. §9.1 now saves
  the reference *input masks and scalars* alongside the tensors, and the self-test recomputes and
  compares with `np.allclose(atol=1e-6)`.
- [Exp] **The model card's numbers are all interpolated**, including the training budget, the
  per-stratum `n`, the flag table and the V-HW sensitivity budget. Nothing is typed by hand, so a
  re-run with different hyperparameters produces a card describing *that* run — and a stale number
  from a previous run cannot survive in it.
- [Exp] `MANIFEST.json` records every file and its size, which makes a truncated download detectable;
  that matters because the weights are the one artefact a partial zip silently breaks.

**OUTPUT** — `/kaggle/working/export/` containing weights, ONNX, `preprocess.py`, `infer.py`,
`conformal.json`, `schema_map.yaml`, `model_card.md`, `DECISIONS.md`, `README.md`,
`references.md`, `requirements.txt`, `config.yaml`, all CSVs and figures, `src/*.py`,
`MANIFEST.json`; then `/kaggle/working/export.zip`.

**CHECK** — `assert not missing` over the full brief §11 list; `preprocess.py`'s own self-test run in
a subprocess must print PASS; `infer.py` executed in a subprocess on a real subject must return all 14
measurements with properly nested intervals for both variants.