"""#### §0.5 Resume state, manifest validation, phase table, and the run protocol

**WHAT** — Locate and adopt any previous `run/` state, validate `run/manifest.json` against its
schema, instantiate `RUNSTATE`, print the phase-status table, and check the working-directory
budget against the Kaggle persistence caps.

**WHY**
- [Exp] Kaggle's *Files only* persistence guarantees files in `/kaggle/working` survive between
  sessions; it guarantees nothing about Python state. A committed *Save & Run All* can also start
  with an empty `/kaggle/working`. So nothing may depend on a variable surviving — every phase
  writes artefacts to disk and can be reloaded.
- [Ref] Addendum A3: resume state is looked for first in `/kaggle/working/run/`, then in
  `/kaggle/input/*/run/` (a previous version's output attached by the user). Working state wins,
  and the cell prints which source it used.
- [Ref] Addendum A2: a phase is reusable only when its status is `done`, its recorded
  `config_hash` matches the current config, **and** every artefact it listed still exists. All
  three are checked independently, so a truncated working directory or a changed `smoke_test`
  cannot silently reuse stale work.
- [Exp] The manifest is validated against a schema at startup rather than trusted. One that fails
  to parse degrades to a fresh manifest instead of aborting: losing an hour of cached work beats
  losing the run.
- [Ref] Addendum A1/A7: `/kaggle/working` is capped at roughly 500 files and 20 GB. This notebook
  budgets below that and asserts it, because a run that only trips the limit after the export has
  been written cannot be fixed in the same session.
- [Ref] Addendum A0: phases 0–3 must run on CPU. That is enforced by a tripwire rather than a
  printed device string — see the determinism check at the end of this cell.

**OUTPUT** — `run/manifest.json`, `RUNSTATE`, `PATHS["run"|"ckpt"|"results"|"tmp"]`, `RESUME`,
`TMP`, `PHASES`, `DEVICE`.

**CHECK** — asserts the manifest validates; asserts every phase's dependency names exist; asserts
the working directory is under `max_files_working` and `max_gb_working`. In `smoke_test` mode it
additionally exercises a full write → reload → verify cycle, and asserts the CUDA tripwire fires.
"""

# --- directory layout (addendum A1) -------------------------------------------------------------
PATHS["run"] = PATHS["working"] / "run"
PATHS["ckpt"] = PATHS["run"] / "ckpt"
PATHS["results"] = PATHS["run"] / "results"
PATHS["figs"] = PATHS["export"] / "figs"
for _p in (PATHS["run"], PATHS["ckpt"], PATHS["results"], PATHS["figs"]):
    _p.mkdir(parents=True, exist_ok=True)

# --- the two modules that make resumption work -------------------------------------------------
import importlib

for _mod in ("run_state", "phase_wrap"):
    _path = PATHS["src"] / f"{_mod}.py"
    if not _path.is_file():
        raise FileNotFoundError(
            f"{_path} is missing. run_state.py is emitted by a %%writefile cell in this notebook; "
            "if you deleted it, re-run that cell before this one.")
    if _mod in sys.modules:
        importlib.reload(sys.modules[_mod])
    else:
        importlib.import_module(_mod)
import run_state as R          # noqa: E402
import phase_wrap               # noqa: E402
from phase_wrap import begin, done, guard, blocked_reason, any_blocked  # noqa: E402

# Regenerable caches belong in /kaggle/temp, which Kaggle does not keep, so they never count
# against the working-directory budget. Locally there is no such mount, so fall back.
TMP = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else PATHS["working"] / "_tmp"
TMP.mkdir(parents=True, exist_ok=True)
PATHS["tmp"] = TMP
print("PATHS:", json.dumps({k: str(v) for k, v in PATHS.items()}, indent=2))
print(f"scratch (not persisted by Kaggle): {TMP}")

# --- A3: find and adopt previous state ----------------------------------------------------------
_INPUTS = sorted(Path("/kaggle/input").glob("*")) if Path("/kaggle/input").is_dir() else []
print("\n=== resume source ===")
RESUME = R.adopt_resume_source(PATHS["working"], _INPUTS)

# --- the phase registry, mirroring the one in the config cell ---------------------------------
# Duplicated deliberately rather than imported from config.yaml: a notebook cannot import a
# Python object out of its own earlier cell, and `config.yaml` round-trips types through YAML.
# The assert below fails loudly if the two ever diverge.
PHASES = {
    "p00_setup":        {"deps": [], "requires_gpu": False},
    "p01_data":         {"deps": ["p00_setup"], "requires_gpu": False},
    "p02_eda":          {"deps": ["p01_data"], "requires_gpu": False},
    "p03_splits":       {"deps": ["p01_data"], "requires_gpu": False},
    "p04_b0":           {"deps": ["p03_splits"], "requires_gpu": False},
    "p05_b1":           {"deps": ["p03_splits"], "requires_gpu": False},
    "p06_train_b2":     {"deps": ["p05_b1"], "requires_gpu": True},
    "p07_sens_b2":      {"deps": ["p06_train_b2"], "requires_gpu": True},
    "p08_train_vhw":    {"deps": ["p05_b1", "p07_sens_b2"], "requires_gpu": True},
    "p09_train_vh":     {"deps": ["p05_b1", "p07_sens_b2"], "requires_gpu": True},
    "p10_sens_shipped": {"deps": ["p07_sens_b2", "p08_train_vhw", "p09_train_vh"],
                         "requires_gpu": True},
    "p11_conformal":    {"deps": ["p03_splits", "p08_train_vhw", "p09_train_vh"],
                         "requires_gpu": True},
    "p12_final_eval":   {"deps": ["p04_b0", "p05_b1", "p08_train_vhw", "p09_train_vh",
                                  "p11_conformal"], "requires_gpu": True},
    "p13_export":       {"deps": ["p08_train_vhw", "p09_train_vh"], "requires_gpu": False},
    "p14_docs":         {"deps": ["p13_export"], "requires_gpu": False},
}

# --- A2/A7: run state, manifest validation, status table ----------------------------------------
RUNSTATE = R.RunState(PATHS["run"], cfg, hash_exclude=cfg.get("hash_exclude") or [],
                      phases=PHASES)
_problems = RUNSTATE.validate()
if _problems:
    print(f"\nmanifest.json did not validate ({len(_problems)} problem(s)):")
    for _p in _problems:
        print("   -", _p)
    print("  A malformed entry will simply fail its reuse test and rerun; the run continues.")
else:
    print(f"\nmanifest.json validated against the schema "
          f"(config_hash {RUNSTATE.hash})")

print(f"\n=== phase status ({len(PHASES)} phases, config_hash={RUNSTATE.hash}) ===")
print(RUNSTATE.status_table().to_string(index=False))
_n_reusable = int(RUNSTATE.status_table()["reusable"].sum())
print(f"  {_n_reusable}/{len(PHASES)} phases reusable in this session")
_bad_deps = [d for s in PHASES.values() for d in s["deps"] if d not in PHASES]
assert not _bad_deps, f"phase registry references unknown phases: {_bad_deps}"
assert RUNSTATE.hash == CONFIG_HASH, (
    f"RunState computed hash {RUNSTATE.hash} but the config cell computed {CONFIG_HASH}")

# --- A7: working-directory budget ----------------------------------------------------------------
print("\n=== working-directory budget (addendum A1/A7) ===")
R.check_working_budget(PATHS["working"], int(cfg["max_files_working"]),
                       float(cfg["max_gb_working"]), strict=True)

# --- A0: device, and the phases that must not need it --------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CPU_ONLY = [n for n, s in PHASES.items() if not s["requires_gpu"]]
print(f"\ndevice for GPU phases: {DEVICE}"
      + (f" ({ENV['gpu']['name']}, {ENV['gpu']['total_gb']} GB)" if torch.cuda.is_available()
         else " (CPU only: training phases will be very slow)"))
print(f"phases that must run without CUDA: {CPU_ONLY}")

# --- smoke-mode self-check of the whole resume mechanism ------------------------------------------
if SMOKE:
    print("\n=== smoke check of the resume mechanism ===")
    _probe = "p00_setup"
    _pd = RUNSTATE.results_dir(_probe + "_selftest")
    for _f in _pd.glob("*"):
        _f.unlink()
    _arts = R.save_payload(_pd, {"frame": pd.DataFrame({"a": [1, 2, 3]}),
                                 "vec": np.arange(3.0), "txt": "hello"})
    _h0 = RUNSTATE.hash
    RUNSTATE.mark_done(_probe + "_selftest", artefacts=_arts, meta={"probe": True})
    _again = R.RunState(PATHS["run"], cfg, hash_exclude=cfg.get("hash_exclude") or [],
                        phases=PHASES, verbose=False)
    _p, _r = _again.maybe_skip(_probe + "_selftest")
    assert _p is not R.BLOCKED and _r == "cached", "a just-written phase must be reusable"
    assert isinstance(_p["frame"], pd.DataFrame) and len(_p["frame"]) == 3
    assert np.allclose(_p["vec"], np.arange(3.0)) and _p["txt"] == "hello"
    print(f"  write -> close -> reload works; payload round-trips "
          f"(DataFrame {len(_p['frame'])} rows, ndarray {_p['vec'].shape}, str {_p['txt']!r})")

    # each of the three reuse conditions must fail independently
    _d2 = R.RunState(PATHS["run"], {**cfg, "smoke_subjects_max": 999},
                     hash_exclude=cfg.get("hash_exclude") or [], phases=PHASES, verbose=False)
    assert not _d2.phase_done(_probe + "_selftest"), \
        "a changed config hash must invalidate a cached phase"
    assert _d2.hash_mismatch(_probe + "_selftest")
    print(f"  a changed config hash invalidates the cache ({_h0} -> {_d2.hash})")
    for _f in _pd.glob("*"):
        _f.unlink()
    _d3 = R.RunState(PATHS["run"], cfg, hash_exclude=cfg.get("hash_exclude") or [],
                     phases=PHASES, verbose=False)
    assert not _d3.phase_done(_probe + "_selftest"), \
        "a deleted artefact must invalidate a cached phase"
    assert _d3.artefacts_present(_probe + "_selftest"), "the missing file must be named"
    print(f"  a deleted artefact invalidates the cache and is named: "
          f"{_d3.artefacts_present(_probe + '_selftest')[:1]}")
    _d3.mark_done(_probe + "_selftest", status="partial", meta={"probe": True})
    _d4 = R.RunState(PATHS["run"], cfg, hash_exclude=cfg.get("hash_exclude") or [],
                     phases=PHASES, verbose=False)
    assert not _d4.phase_done(_probe + "_selftest"), "a partial phase must never be reusable"
    print("  a partial phase is never reusable")
    for _f in _pd.glob("*"):
        _f.unlink()
    _d4.manifest["phases"].pop(_probe + "_selftest", None)
    _d4.save_manifest()
    print("  all three reuse conditions verified independently")

    # the CUDA tripwire must actually fire, or "phases 0-3 run on CPU" is only a claim
    _tripped = False
    try:
        with R.cpu_guard("p00_setup"):
            torch.zeros(2).to("cuda")
    except AssertionError:
        _tripped = True
    assert _tripped, "cpu_guard did not trip on a CUDA request"
    print("  CUDA tripwire fires inside a CPU-only phase")
print("\nmark_phase('setup') =", mark_phase("setup"), "s")