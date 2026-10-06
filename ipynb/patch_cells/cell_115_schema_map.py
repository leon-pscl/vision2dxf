# TODO(user) pattern-engine key names. Left as placeholders on purpose (brief section 14).
PATTERN_KEY_TODO = {
    "chest": "TODO(user): e.g. bust_circumference",
    "waist": "TODO(user): e.g. waist_circumference",
    "hip": "TODO(user): e.g. hip_circumference",
    "ankle": "TODO(user)",
    "arm-length": "TODO(user)",
    "bicep": "TODO(user)",
    "calf": "TODO(user)",
    "forearm": "TODO(user)",
    "height": "TODO(user)",
    "leg-length": "TODO(user)",
    "shoulder-breadth": "TODO(user)",
    "shoulder-to-crotch": "TODO(user)",
    "thigh": "TODO(user)",
    "wrist": "TODO(user)",
}
assert set(PATTERN_KEY_TODO) == set(TARGETS), (
    f"pattern-key placeholders must cover every target; missing "
    f"{sorted(set(TARGETS) - set(PATTERN_KEY_TODO))}")

NOT_PROVIDED = [
    "bust_point (bust apex position)", "underbust_circumference", "back_length",
    "armscye_depth", "crotch_depth", "nape_to_wrist", "across_back",
    "bust_span", "side_seam_length", "dart positions", "ease allowance",
]


def _test_mae(split: str, m: str) -> Optional[float]:
    """Test-split MAE for one measurement, or None when the final evaluation was not run.

    Parameters
    ----------
    split : {'testA', 'testB'}
        Test split name.
    m : str
        Measurement name.

    Returns
    -------
    float or None
        MAE in mm, or None when ``run_final_eval`` was false.
    """
    if FINAL_COMPARISON is None:
        return None
    sel = FINAL_COMPARISON[(FINAL_COMPARISON.split == split) & (FINAL_COMPARISON.model == "V-HW")
                          & (FINAL_COMPARISON.measurement == m)]
    return float(sel["mae_mm"].iloc[0]) if len(sel) else None


schema_map = {
    "schema_version": 1,
    "units": "cm",
    "source_dataset": "BodyM (Ruiz et al., 2022)",
    "ground_truth_definition":
        "SMPL-registered mesh vertex-path length; NOT an ISO 20685 tape measurement. "
        "No ISO 20685 compliance is claimed or implied.",
    "ground_truth_definition_note":
        "Ruiz et al. (2022) do not publish the vertex path behind each column name, so each "
        "entry below states only what is known.",
    "evaluation_unit": "photo_pair (one front/side silhouette pair), matching infer.py",
    "final_evaluation_run": bool(RUN_FINAL_EVAL),
    "measurements": {
        m: {"body_m_name": m,
            "pattern_engine_key": PATTERN_KEY_TODO[m],
            "units": "cm",
            "definition": DEFINITIONS[m],       # fix D4: one honest, uniform statement
            "is_ground_truth_input": m == "height",
            "validation_mae_mm": float(mae_tbl.set_index("measurement").loc[m, "mae_VHW_mm"]),
            "validation_beats_B0": bool(mae_tbl.set_index("measurement").loc[m, "VHW_beats_B0"]),
            "testA_mae_mm": _test_mae("testA", m),
            "testB_mae_mm": _test_mae("testB", m)}
        for m in TARGETS},
    "pattern_keys_not_provided": NOT_PROVIDED,
    "tolerance_thresholds_mm": cfg["tolerance_mm"],
    "tolerance_owner": "TODO(user): the pattern engine must state which threshold it accepts per "
                       "measurement; these are reporting thresholds only.",
}
(EXPORT_DIR / "schema_map.yaml").write_text(yaml.safe_dump(schema_map, sort_keys=False),
                                           encoding="utf-8")
print("written:", EXPORT_DIR / "schema_map.yaml")
if not RUN_FINAL_EVAL:
    _sm = yaml.safe_load((EXPORT_DIR / "schema_map.yaml").read_text(encoding="utf-8"))
    _null_mae = [k for k, v in _sm["measurements"].items() if v["testA_mae_mm"] is None]
    print(f"  final_evaluation_run: false -> testA/testB MAE are null for all "
          f"{len(_null_mae)} measurements (not fabricated, not omitted)")
print("pattern keys NOT provided (step 8 must not invent these):")
for k in NOT_PROVIDED:
    print("  -", k)
print("mark_phase('schema map') =", mark_phase("schema map"), "s")