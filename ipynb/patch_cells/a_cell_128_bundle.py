# --- requirements (pinned), config, split CSVs, sensitivity ----------------------------------
def pinned(module_name: str) -> str:
    """Return a pinned requirement line for an installed module.

    Parameters
    ----------
    module_name : str
        Import name, e.g. ``'numpy'``.

    Returns
    -------
    str
        ``name==version``, or ``name`` if the version cannot be determined.
    """
    import importlib.metadata as md
    try:
        return f"{module_name}=={md.version(module_name)}"
    except Exception:
        return module_name


PINNED = [pinned(m) for m in ["numpy", "pandas", "scikit-learn", "scipy", "matplotlib",
                              "pillow", "pyyaml", "torch", "torchvision", "tqdm",
                              "onnx", "onnxruntime"]]
(EXPORT_DIR / "requirements.txt").write_text("\n".join(PINNED) + "\n", encoding="utf-8")
print("written: requirements.txt (pinned from the live environment)")
print("  " + ", ".join(PINNED))

# every CSV this notebook actually wrote, so the bundle carries them without a hand-maintained list
_CSV_NAMES = ["splits_train.csv", "sensitivity.csv", "max_tolerable_boundary_error.csv",
              "validation_mae_comparison.csv", "flagged_measurements.csv",
              "final_comparison.csv", "coverage_summary.csv", "bmi_stratum_errors.csv",
              "subject_averaged_metrics.csv", "subject_averaged_comparison.csv",
              "calibration_photo_pairs.csv", "onnx_verification.csv",
              "history_b2.csv", "history_vhw.csv", "history_vh.csv"]
for _n in _CSV_NAMES:
    _src = ART / _n
    if _src.is_file():
        shutil.copy2(_src, EXPORT_DIR / _n)
    elif _n in ("final_comparison.csv", "coverage_summary.csv", "bmi_stratum_errors.csv",
                "subject_averaged_comparison.csv"):
        print(f"  (skipped {_n}: not produced because run_final_eval is false)")
    else:
        raise FileNotFoundError(f"expected artefact missing: {_src}")

shutil.copy2(ART / "augmentation_magnitudes.yaml", EXPORT_DIR / "augmentation_magnitudes.yaml")
shutil.copy2(CFG_PATH, EXPORT_DIR / "config.yaml")
_cfg_b = yaml.safe_load((EXPORT_DIR / "config.yaml").read_text(encoding="utf-8"))
_cfg_b["preprocess_atol"] = 1e-6
(EXPORT_DIR / "config.yaml").write_text(yaml.safe_dump(_cfg_b, sort_keys=True), encoding="utf-8")

_FIG_NAMES = ["fig_eda_population.png", "fig_eda_measurements.png", "fig_eda_correlation.png",
              "fig_eda_imagery.png", "fig_eda_artefacts_worst.png", "fig_eda_artefacts_typical.png",
              "fig_b1_slices.png", "fig_b2_training.png", "fig_variants_validation.png",
              "fig_sensitivity_morph.png", "fig_sensitivity_other.png",
              "fig_testA_vs_testB.png", "fig_bmi_failure_mode.png"]
_copied_figs = []
for _n in _FIG_NAMES:
    if (ART / _n).is_file():
        shutil.copy2(ART / _n, EXPORT_DIR / _n)
        _copied_figs.append(_n)
    else:
        print(f"  (figure {_n} not produced in this run)")

# the source modules are part of the provenance record
_SRC_NAMES = ["bodym_data.py", "bodym_metrics.py", "bodym_baselines.py", "bodym_cnn.py",
              "bodym_training.py", "bodym_conformal.py", "bodym_perturb.py",
              "bodym_sensitivity.py", "bodym_eda.py"]
for _n in _SRC_NAMES:
    if not (PATHS["src"] / _n).is_file():
        raise FileNotFoundError(f"source module not written: {PATHS['src'] / _n}")
    shutil.copy2(PATHS["src"] / _n, EXPORT_DIR / _n)

print(f"copied: config, {len(_CSV_NAMES)} CSVs, {len(_copied_figs)} figures, "
      f"{len(_SRC_NAMES)} source modules")
print("mark_phase('docs export') =", mark_phase("docs export"), "s")