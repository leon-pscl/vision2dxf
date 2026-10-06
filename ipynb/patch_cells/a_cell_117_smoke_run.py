require("p08_train_vhw", "p09_train_vh")

if RUN_FINAL_EVAL:
    _probe_sid = TEST_IDS["testA"][0]
    _probe_meta = IDX["testA"].subjects.set_index("subject_id").loc[_probe_sid]
    _probe_photo = IDX["testA"].photos[IDX["testA"].photos.subject_id == _probe_sid].iloc[0]
    _probe_split = "testA"
else:
    # run_final_eval is false: using a Test-A subject here would consume the single-use
    # evaluation budget on a smoke test, which brief section 1 forbids. Fall back to a
    # validation subject, which is already fully used.
    _probe_sid = ids_val[0]
    _probe_meta = IDX["train"].subjects.set_index("subject_id").loc[_probe_sid]
    _probe_photo = IDX["train"].photos[IDX["train"].photos.subject_id == _probe_sid].iloc[0]
    _probe_split = "validation"
    print("run_final_eval: false -> the infer.py smoke test uses a VALIDATION subject so the "
          "single-use Test-A/Test-B budget is not consumed by a smoke test.")

(EXPORT_DIR / "_smoke_input.json").write_text(json.dumps({
    "front_mask": str(_probe_photo.front_mask), "side_mask": str(_probe_photo.side_mask),
    "height_cm": float(_probe_meta.height_cm), "weight_kg": float(_probe_meta.weight_kg),
    "subject_id": str(_probe_sid), "split": _probe_split, "targets": TARGETS}),
    encoding="utf-8")

print(f"end-to-end test on real {_probe_split} subject {_str(_probe_sid)[:16]} "
      f"(gender {_probe_meta.gender}, {_probe_meta.height_cm:.1f} cm, "
      f"{_probe_meta.weight_kg:.1f} kg, BMI {_probe_meta.bmi:.1f})")

_r = subprocess.run([sys.executable, str(EXPORT_DIR / "_smoke_test.py")],
                    capture_output=True, text=True, cwd=str(EXPORT_DIR))
if _r.returncode != 0:
    print("STDERR:\n", _r.stderr[-3000:])
    raise RuntimeError(f"infer.py smoke test failed (exit {_r.returncode})")
_smoke = json.loads(_r.stdout.strip().splitlines()[-1])

res_hw, res_h = _smoke["V-HW"], _smoke["V-H"]
assert _smoke["select"] == ["V-HW", "V-H"], f"variant auto-selection wrong: {_smoke['select']}"
assert sorted(res_hw) == sorted(TARGETS), "V-HW did not return every measurement"
assert sorted(res_h) == sorted(TARGETS), "V-H did not return every measurement"
print("  variant auto-selection:", _smoke["select"], "OK")
assert all("clamped" in res_hw[m] for m in TARGETS), \
    "every measurement must report whether its interval was clamped (fix B12)"

truth = _probe_meta[TARGETS].to_numpy(float)
print(f"\nmeasurement          V-HW    truth   err(mm)  lo90(mm)  hi90(mm)  in90  "
      f"clamp     V-H")
n_in90 = 0
for j, m in enumerate(TARGETS):
    inside = res_hw[m]["lo90"] <= truth[j] <= res_hw[m]["hi90"]
    n_in90 += int(inside)
    print(f"  {m:18s} {res_hw[m]['value_cm']:7.2f} {truth[j]:8.2f} "
          f"{(res_hw[m]['value_cm']-truth[j])*10:8.2f} {res_hw[m]['lo90']*10:9.2f} "
          f"{res_hw[m]['hi90']*10:9.2f}   {'y' if inside else 'N'}   "
          f"{str(res_hw[m]['clamped']):5s}  {res_h[m]['value_cm']:7.2f}")
print(f"\n90% interval contains truth for {n_in90}/{len(TARGETS)} measurements on this subject")

for _r_, _nm in ((res_hw, "V-HW"), (res_h, "V-H")):
    for m in TARGETS:
        d = _r_[m]
        assert d["lo90"] <= d["lo80"] <= d["value_cm"] <= d["hi80"] <= d["hi90"], \
            f"{_nm} interval nesting broken for {m}: {d}"
print("interval nesting (lo90 <= lo80 <= value <= hi80 <= hi90) holds for every measurement,")
print("including after clamping - a clamp never removes the point estimate from its own "
      "interval (fix B12).")

(EXPORT_DIR / f"result_{_probe_split}_subject.json").write_text(json.dumps(
    {"subject_id": str(_probe_sid), "split": _probe_split,
     "height_cm": float(_probe_meta.height_cm),
     "weight_kg": float(_probe_meta.weight_kg), "targets": TARGETS,
     "ground_truth_cm": {m: float(truth[j]) for j, m in enumerate(TARGETS)},
     "V-HW": res_hw, "V-H": res_h}, indent=2), encoding="utf-8")
(EXPORT_DIR / "_smoke_test.py").unlink()
(EXPORT_DIR / "_smoke_input.json").unlink()
print("wrote example result:", EXPORT_DIR / f"result_{_probe_split}_subject.json")
print("mark_phase('infer smoke test') =", mark_phase("infer smoke test"), "s")