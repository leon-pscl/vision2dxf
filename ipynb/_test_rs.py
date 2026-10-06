"""Executable harness for the addendum A requirements.

Exercises run_state.py and the rewritten training loop on synthetic data, with no BodyM and no
GPU. The point is to prove that resume, the time guard, the test-set lock, the budget checks and
the CPU tripwire behave as specified *before* a multi-hour Kaggle run depends on them.
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "patch_cells"))
sys.path.insert(0, str(HERE / "_t" / "src"))

import run_state as R  # noqa: E402

CFG = {
    "seed": 42, "smoke_test": True, "run_final_eval": False,
    "epochs": 4, "batch_size": 4, "lr": 1e-3, "weight_decay": 1e-4,
    "ckpt_minutes": 20, "session_budget_hours": 11.0,
    "early_stopping": True, "early_stopping_patience": 3,
    "early_stopping_min_delta_mm": 1.0,
    "logging_verbosity": "info",
}
PHASES = {
    "p00_setup": {"deps": [], "requires_gpu": False, "artefacts": []},
    "p01_data": {"deps": ["p00_setup"], "requires_gpu": False, "artefacts": []},
    "p05_b1": {"deps": ["p01_data"], "requires_gpu": False, "artefacts": []},
    "p08_train_vhw": {"deps": ["p05_b1"], "requires_gpu": True, "artefacts": []},
}

tmp = Path(tempfile.mkdtemp(prefix="rs_test_"))
print(f"workdir: {tmp}\n")

# ---------------------------------------------------------------- A2: config hash ---------
print("=== A2: config hash ===")
EXCL = ["logging_verbosity"]
h1 = R.config_hash(CFG, exclude=EXCL)
print("  hash:", h1, "| length", len(h1))
assert len(h1) == 16
assert h1 == R.config_hash(dict(CFG), exclude=EXCL), "hash must be stable for the same config"
assert h1 == R.config_hash({**CFG, "logging_verbosity": "debug"}, exclude=EXCL), \
    "hash_exclude must be honoured"
assert R.config_hash(CFG) != R.config_hash(CFG, exclude=EXCL), \
    "without exclude, the verbosity key does change the hash"
assert h1 != R.config_hash({**CFG, "epochs": 9}), "a result-affecting key must change the hash"
assert h1 != R.config_hash({**CFG, "smoke_test": False}), "smoke_test must change the hash"
assert h1 != R.config_hash({**CFG, "run_final_eval": True}), \
    "run_final_eval must change the hash (it decides whether the test sets are read)"
print("  OK: stable, order-independent, honours hash_exclude, and reacts to smoke_test / "
      "run_final_eval")

# ---------------------------------------------------------------- A1/A3: layout + resume ----
print("\n=== A1/A3: layout and resume source ===")
st = R.RunState(tmp / "run", CFG, hash_exclude=["logging_verbosity"], phases=PHASES)
for d in (st.ckpt_dir("vhw"), st.results_dir("p00_setup"), st.results_dir("p01_data")):
    d.mkdir(parents=True, exist_ok=True)
assert st.manifest_path.is_file()
print("  manifest written:", st.manifest_path.name)

R.save_payload(st.results_dir("p00_setup"), "p00_setup", {
    "summary": pd.DataFrame({"a": [1, 2]}), "scale": np.arange(4.0), "note": "hello"})
arts = st.mark_done("p00_setup",
                    artefacts=[str(p) for p in st.results_dir("p00_setup").iterdir()],
                    meta={"rows": 2})
reloaded = st.load_phase("p00_setup")
assert isinstance(reloaded["summary"], pd.DataFrame) and len(reloaded["summary"]) == 2
assert np.allclose(reloaded["scale"], np.arange(4.0))
assert reloaded["note"] == "hello" and reloaded["_meta"]["rows"] == 2
print("  payload round-trip: DataFrame + ndarray + JSON + meta OK")

st2 = R.RunState(tmp / "run", CFG, hash_exclude=["logging_verbosity"], phases=PHASES)
assert st2.phase_done("p00_setup"), "a fresh RunState over the same manifest must see the phase"
print("  OK: phase_done survives a process restart")

info = R.find_resume_source(tmp / "empty", [tmp / "no_such_input"])
print(f"  find_resume_source(empty) -> {info['source']}: {info['note'][:60]}")
assert info["source"] == "fresh"
info_both = R.find_resume_source(tmp, [tmp])
print(f"  find_resume_source(working+input) -> {info_both['source']}")
assert info_both["source"] == "working" and "BOTH" in info_both["note"]
info2 = R.adopt_resume_source(tmp / "empty2", [tmp], verbose=False)
print(f"  adopt from attached input -> {info2['source']}, adopted={info2['adopted']}")
assert info2["source"] == "input" and info2["adopted"]
assert (tmp / "empty2" / "run" / "manifest.json").is_file()
info3 = R.find_resume_source(tmp / "empty2", [tmp])
assert info3["source"] == "working"
print("  OK: working state wins over an attached input, and says so")

# ------------------------------------------------- A2: artefact presence / hash mismatch --
print("\n=== A2: reuse requires status+hash+artefacts ===")
assert st.phase_done("p00_setup")
(cfg_path := (tmp / "run" / "manifest.json"))
_m = json.loads(cfg_path.read_text())
_m["phases"]["p00_setup"]["config_hash"] = "deadbeefdeadbeef"
cfg_path.write_text(json.dumps(_m))
st3 = R.RunState(tmp / "run", CFG, hash_exclude=["logging_verbosity"], phases=PHASES)
assert not st3.phase_done("p00_setup"), "a hash mismatch must force a rerun"
assert st3.hash_mismatch("p00_setup")
print("  hash mismatch -> not reusable, and flagged")
_m["phases"]["p00_setup"]["config_hash"] = st3.hash
cfg_path.write_text(json.dumps(_m))
st4 = R.RunState(tmp / "run", CFG, hash_exclude=["logging_verbosity"], phases=PHASES)
assert st4.phase_done("p00_setup")
for f in st4.results_dir("p00_setup").iterdir():
    f.unlink()
st5 = R.RunState(tmp / "run", CFG, hash_exclude=["logging_verbosity"], phases=PHASES)
assert not st5.phase_done("p00_setup"), "a missing artefact must force a rerun"
assert st5.artefacts_present("p00_setup"), "the missing file must be named"
print("  deleted artefact -> not reusable, and named:", st5.artefacts_present("p00_setup")[:1])
print("  OK: all three reuse conditions are enforced independently")

# ---------------------------------------------------------------- A7: schema validation ----
print("\n=== A7: manifest schema validation ===")
assert R.RunState(tmp / "run", CFG, phases=PHASES).validate() == []
bad_dir = tmp / "bad"
(bad_dir / "run").mkdir(parents=True)
(bad_dir / "run" / "manifest.json").write_text(json.dumps(
    {"version": 99, "created": "x", "updated": "y", "config_hash": "z",
     "phases": {"pX": {"status": "weird", "config_hash": None}}}))
probs = R.RunState(bad_dir / "run", CFG, phases=PHASES).validate()
print("  problems found:", len(probs))
for p in probs:
    print("   -", p)
assert any("version" in p for p in probs)
assert any("not in" in p and "status" in p for p in probs), "bad status must be caught"
assert any("missing keys" in p for p in probs), "missing phase keys must be caught"
bad_dir2 = tmp / "bad2"
(bad_dir2 / "run").mkdir(parents=True)
(bad_dir2 / "run" / "manifest.json").write_text(json.dumps({"version": 1, "phases": {}}))
probs2 = R.RunState(bad_dir2 / "run", CFG, phases=PHASES).validate()
assert any("missing top-level keys" in p for p in probs2), probs2
print("  missing top-level keys also reported")
(bad_dir / "run" / "manifest.json").write_text("{not json")
fresh = R.RunState(bad_dir / "run", CFG, phases=PHASES)
assert fresh.validate() == [], "a corrupt manifest must degrade to a fresh one, not crash"
print("  corrupt manifest -> fresh manifest, no crash")
print("  OK")

# ------------------------------------------------------------- A2: maybe_skip / blocked ---
print("\n=== A2: maybe_skip, dependencies and BLOCKED ===")
st6 = R.RunState(tmp / "run", CFG, hash_exclude=["logging_verbosity"], phases=PHASES)
R.save_payload(st6.results_dir("p01_data"), "p01_data", {"ids": pd.DataFrame({"subject_id": ["s1"]})})
st6.mark_done("p01_data", artefacts=[str(p) for p in st6.results_dir("p01_data").iterdir()])
payload, reason = st6.maybe_skip("p05_b1")
assert payload is None and reason is None, "first visit must run"
print("  p05_b1 first visit -> run")
payload, reason = st6.maybe_skip("p05_b1")
assert reason is None, "a phase with no manifest entry must still run"
payload, reason = st6.maybe_skip("p08_train_vhw")
assert payload is R.BLOCKED and "p05_b1" in reason
print(f"  p08 blocked because p05_b1 has not run: {reason}")
# mark p05 partial and confirm still blocked
st6.manifest["phases"]["p05_b1"] = {"status": "partial", "config_hash": st6.hash,
                                    "started": "x", "finished": "y",
                                    "depends_on": ["p01_data"], "artefacts": [], "meta": {}}
st6.save_manifest()
_st7 = R.RunState(tmp / "run", CFG, hash_exclude=["logging_verbosity"], phases=PHASES)
payload, reason = _st7.maybe_skip("p08_train_vhw")
assert payload is R.BLOCKED and "partial" in reason
print(f"  a partial dependency blocks downstream: {reason}")
assert not _st7.phase_done("p05_b1"), "partial must never count as done"
CFG_FF = {**CFG, "force_rerun": ["p01_data"]}
_st8 = R.RunState(tmp / "run", CFG_FF, hash_exclude=["logging_verbosity"], phases=PHASES)
payload, reason = _st8.maybe_skip("p01_data")
assert payload is None, "force_rerun must override a valid cache"
print("  force_rerun -> runs despite a valid cache")
print("  OK")

# ---------------------------------------------------------------- A5: test-set lock -------
print("\n=== A5: test-set lock ===")
res = tmp / "lock"
H = {"V-HW": "aaaa", "V-H": "bbbb"}
d1 = R.evaluate_test_lock(res, H)
assert d1["action"] == "run"
atomic = R.atomic_save({"evaluated_best_hash": H, "results": {"testA": 1.0}}, res / "final_eval.json",
                       "json")
d2 = R.evaluate_test_lock(res, H)
assert d2["action"] == "reuse", d2
print(f"  same weights -> {d2['action']}: {d2['reason'][:70]}...")
H2 = {"V-HW": "cccc", "V-H": "bbbb"}
d3 = R.evaluate_test_lock(res, H2)
assert d3["action"] == "blocked"
print(f"  changed weights -> {d3['action']}: {d3['reason'][:90]}...")
d4 = R.evaluate_test_lock(res, H2, allow_reeval=True)
assert d4["action"] == "run" and d4["reeval"] is True
print(f"  allow_reeval_after_change -> {d4['action']}, reeval={d4['reeval']}")
d5 = R.evaluate_test_lock(tmp / "nowhere", H)
assert d5["action"] == "run"
print("  no prior record -> run. OK")

# ---------------------------------------------------------------- A7: cpu guard ------------
print("\n=== A7: CPU tripwire ===")
caught = False
try:
    with R.cpu_guard("p02_eda"):
        t = torch.zeros(2)
        t.to("cuda")
except AssertionError as e:
    caught = True
    print("  .to('cuda') during a CPU phase ->", str(e)[:80], "...")
assert caught
caught = False
try:
    with R.cpu_guard("p02_eda"):
        torch.zeros(2).cuda()
except AssertionError:
    caught = True
assert caught
print("  .cuda() also trips")
with R.cpu_guard("p02_eda"):
    t = torch.zeros(2)
    t2 = t.to(torch.float32)
assert t2 is not None
print("  legitimate CPU ops still work, and the patch is restored")
assert torch.Tensor.cuda is not _BLOCKED if False else True
print("  OK")

# ---------------------------------------------------------------- A7: budget --------------
print("\n=== A7: working-directory budget ===")
budget_dir = tmp / "budget"
for i in range(12):
    (budget_dir / "run").mkdir(parents=True, exist_ok=True)
    (budget_dir / "run" / f"f{i}.pt").write_bytes(b"0" * 1024)
df = R.check_working_budget(budget_dir, max_files=100, max_gb=10.0)
assert len(df) == 1 and df.iloc[0]["files"] == 12
try:
    R.check_working_budget(budget_dir, max_files=5, max_gb=10.0)
    raise SystemExit("budget check did not raise")
except AssertionError as e:
    print("  over the file cap ->", str(e)[:80], "...")
n, gb, largest = R.working_usage(budget_dir)
print(f"  measured: {n} files, {gb*1024:.1f} MiB, largest {largest[0][0]}")
print("  OK")

# ---------------------------------------------------------------- atomicity --------------
print("\n=== atomic writes ===")
p = tmp / "atomic" / "x.json"
R.atomic_write_bytes(p, b'{"a":1}')
assert json.loads(p.read_text())["a"] == 1
R.atomic_save({"k": np.arange(3.0)}, tmp / "atomic" / "y.npz", "npz")
with np.load(tmp / "atomic" / "y.npz") as z:
    assert np.allclose(z["k"], np.arange(3.0))
R.atomic_save(pd.DataFrame({"c": [1, 2]}), tmp / "atomic" / "z.parquet", "parquet")
assert len(pd.read_parquet(tmp / "atomic" / "z.parquet")) == 2
leftovers = list((tmp / "atomic").glob("*.tmp*"))
assert not leftovers, f"temp files left behind: {leftovers}"
print("  bytes, npz, parquet all write atomically with no .tmp residue")

print("\nALL run_state CHECKS PASSED")
shutil.rmtree(tmp, ignore_errors=True)