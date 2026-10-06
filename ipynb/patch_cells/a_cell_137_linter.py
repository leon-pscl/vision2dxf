# --- optional advisory linter ---------------------------------------------------------------
# fix A10: the previous cell passed capture_output twice to subprocess.run, which is a
# SyntaxError at compile time, and referenced an undefined name `_sp`. Either one stops the
# whole cell from executing. The .ipynb is also not necessarily present in /kaggle/working on
# Kaggle, in which case the step is skipped rather than failing.
LINT_REPORT = "pynblint not installed and could not be installed; linter skipped (advisory only)"
_nb_candidates = [Path("model_training_measurement.ipynb"),
                  Path(__file__).resolve().parent / "model_training_measurement.ipynb"
                  if "__file__" in dir() else Path("."),
                  PATHS["working"] / "model_training_measurement.ipynb"]
_nb_path = next((p for p in _nb_candidates if p.is_file()), None)
if _nb_path is None:
    LINT_REPORT = ("skipped: model_training_measurement.ipynb is not present in the working "
                   "directory (expected on Kaggle). The linter is advisory only.")
    print("=== advisory linter (pynblint, Quaranta et al. 2022) ===")
    print(LINT_REPORT)
else:
    try:
        import importlib.util as _iu
        if _iu.find_spec("pynblint") is None:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pynblint"],
                           check=False, capture_output=True, text=True)
        if _iu.find_spec("pynblint") is not None:
            _lr = subprocess.run(
                [sys.executable, "-m", "pynblint", "--fail-level", "warning", str(_nb_path)],
                capture_output=True, text=True, timeout=300)
            LINT_REPORT = (_lr.stdout.strip()[-4000:] or _lr.stderr.strip()[-2000:]
                           or f"no findings (exit {_lr.returncode})")
        else:
            LINT_REPORT = "pynblint is not importable after install attempt; skipped (advisory)"
    except Exception as _exc:
        LINT_REPORT = f"pynblint unavailable: {_exc} (advisory only)"
    print(f"=== advisory linter (pynblint, Quaranta et al. 2022) on {_nb_path.name} ===")
    print(LINT_REPORT)