# Mirror the training module into the bundle so it is self-contained offline.
shutil.copy2(PATHS["src"] / "bodym_cnn.py", EXPORT_DIR / "bodym_cnn.py")
print("written:", EXPORT_DIR / "bodym_cnn.py", "(preprocess.py is written by this notebook's"
      " own %%writefile cell, so there is nothing to copy)")

# Run preprocess.py's own self-test in a subprocess, exactly as a downstream user would.
_r = subprocess.run([sys.executable, str(EXPORT_DIR / "preprocess.py")],
                    capture_output=True, text=True, cwd=str(EXPORT_DIR))
print("preprocess.py self-test ->")
print((_r.stdout or _r.stderr).strip())
assert _r.returncode == 0, (
    f"preprocess.py self-test failed (exit {_r.returncode}).\n"
    f"stdout: {_r.stdout}\nstderr: {_r.stderr}")
assert "PASS" in (_r.stdout or ""), (
    f"preprocess.py self-test did not print PASS:\n{_r.stdout}\n{_r.stderr}")
print("preprocess.py reproduces the training tensors exactly, verified from a subprocess "
      "with no notebook globals in scope.")
print("mark_phase('preprocess export') =", mark_phase("preprocess export"), "s")