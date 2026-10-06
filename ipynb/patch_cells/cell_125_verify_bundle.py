# --- verify every brief section 11 item exists --------------------------------------------
REQUIRED_EXPORTS = [
    "model_vhw.onnx", "model_vh.onnx", "model_vhw.pt", "model_vh.pt",
    "preprocess.py", "infer.py", "conformal.json", "schema_map.yaml",
    "model_card.md", "DECISIONS.md", "README.md", "references.md",
    "requirements.txt", "config.yaml", "splits_train.csv", "sensitivity.csv",
]
missing = [f for f in REQUIRED_EXPORTS if not (EXPORT_DIR / f).is_file()]
assert not missing, f"missing required exports: {missing}"
print(f"all {len(REQUIRED_EXPORTS)} required brief section 11 artefacts present")
for _f in REQUIRED_EXPORTS:
    print(f"    {_f:36s} {(EXPORT_DIR / _f).stat().st_size/1024:9.1f} KiB")

# The bundle must actually load, not merely contain the files.
assert (EXPORT_DIR / "bodym_cnn.py").is_file(), "bundle is missing bodym_cnn.py"
assert (EXPORT_DIR / "reference_tensors.npz").is_file(), \
    "bundle is missing reference_tensors.npz (the preprocess self-test would SKIP)"
assert (EXPORT_DIR / "reference_inputs.npz").is_file(), \
    "bundle is missing reference_inputs.npz; the preprocess self-test cannot recompute tensors"

EXPORT_MANIFEST = {
    "generated_by": "model_training_measurement.ipynb",
    "seed": cfg["seed"],
    "smoke_test": SMOKE,
    "run_final_eval": RUN_FINAL_EVAL,
    "train_steps": TRAIN_STEPS,
    "files": sorted(p.name for p in EXPORT_DIR.iterdir() if p.is_file()),
    "counts": {},
}
for p in sorted(EXPORT_DIR.iterdir()):
    if p.is_file():
        EXPORT_MANIFEST["counts"][p.name] = p.stat().st_size
(EXPORT_DIR / "MANIFEST.json").write_text(json.dumps(EXPORT_MANIFEST, indent=2))
print(f"\nMANIFEST.json: {len(EXPORT_MANIFEST['files'])} files, "
      f"{sum(EXPORT_MANIFEST['counts'].values())/1e6:.1f} MB total")

if cfg["zip_export"]:
    zip_path = shutil.make_archive(str(PATHS["working"] / "export"), "zip", root_dir=EXPORT_DIR)
    print("zipped:", zip_path, f"({Path(zip_path).stat().st_size/1e6:.1f} MB)")
    print("  the zip is written to /kaggle/working/export.zip (Kaggle's /kaggle/outputs is "
          "created only on Save Version; download the file from the notebook output pane)")
else:
    print("zip_export disabled in config")

print(f"\nexport directory: {EXPORT_DIR}")
print(f"{len(list(EXPORT_DIR.iterdir()))} files")
print("mark_phase('export') =", mark_phase("export"), "s")