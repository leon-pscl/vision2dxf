if not RUN_FINAL_EVAL:
    print("=" * 78)
    print("SECTION 8 SKIPPED: config.yaml run_final_eval = false")
    print("Test-A and Test-B are reserved for a single final evaluation (brief section 1).")
    print("Nothing below this line consumes them. Everything that depends on these results")
    print("(FINAL_COMPARISON, COVERAGE_SUMMARY, the model card's test tables, schema_map's")
    print("test MAEs, the summary figure) will report 'final evaluation not run'.")
    print("Set run_final_eval: true and Save & Run All once, after reviewing sections 1-7.")
    print("=" * 78)

    RESULTS: Dict[str, Dict[str, object]] = {}
    FINAL_COMPARISON: Optional[pd.DataFrame] = None
    COVERAGE_SUMMARY: Optional[pd.DataFrame] = None
    BMI_ERRS: Optional[pd.DataFrame] = None
    TEST_IDS: Dict[str, List[str]] = {}

else:
    TEST_IDS = {"testA": sorted(IDX["testA"].subjects.subject_id),
                "testB": sorted(IDX["testB"].subjects.subject_id)}

    print("=== single-use protocol audit ===")
    train_all = set(ids_fit) | set(ids_val) | set(ids_calib)
    for split, ids in TEST_IDS.items():
        overlap = train_all & set(ids)
        print(f"  {split}: {len(ids)} subjects, overlap with any train split = {len(overlap)}")
        assert not overlap, f"{split} subjects leaked into training"
    # also audit the *photo* ids, which is the stricter check
    train_photos = set(IDX["train"].photos.photo_id)
    for split in ("testA", "testB"):
        ov = train_photos & set(IDX[split].photos.photo_id)
        assert not ov, f"{split} photo ids overlap train ({len(ov)})"
        print(f"  {split}: photo-id overlap with train = 0")
    print("  Test-A/Test-B used exactly once, below this line.")

    FROZEN = {
        "V-HW": {"state_dict": {k: v.cpu() for k, v in net_vhw.state_dict().items()},
                 "conformal": CONFORMAL_QUANTILES["V-HW"], "use_weight": True},
        "V-H": {"state_dict": {k: v.cpu() for k, v in net_vh.state_dict().items()},
                "conformal": CONFORMAL_QUANTILES["V-H"], "use_weight": False},
    }
    print("\nfrozen artefacts:", {k: {"tensors": len(v["state_dict"]),
                                      "measurements": len(v["conformal"])} for k, v in FROZEN.items()})
    print("mark_phase('protocol audit') =", mark_phase("protocol audit"), "s")