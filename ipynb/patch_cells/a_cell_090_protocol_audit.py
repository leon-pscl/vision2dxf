# --- phase p12_final_eval: the single-use test evaluation, guarded ---------------------------
if not RUN_FINAL_EVAL:
    print("=" * 78)
    print("SECTION 8 SKIPPED: config.yaml run_final_eval = false")
    print("Test-A and Test-B are reserved for a single final evaluation (brief section 1).")
    print("Nothing below this line consumes them. Everything that depends on these results")
    print("(FINAL_COMPARISON, COVERAGE_SUMMARY, the model card's test tables, schema_map's")
    print("test MAEs, the summary figure) will report 'final evaluation not run'.")
    print("=" * 78)

    RESULTS: Dict[str, Dict[str, object]] = {}
    FINAL_COMPARISON: Optional[pd.DataFrame] = None
    COVERAGE_SUMMARY: Optional[pd.DataFrame] = None
    BMI_ERRS: Optional[pd.DataFrame] = None
    TEST_IDS: Dict[str, List[str]] = {}
    TEST_LOCK: Dict[str, Any] = {"action": "skipped", "reason": "run_final_eval: false"}

elif net_vhw is None or net_vh is None:
    # A training phase was blocked, so there is no model to evaluate. Refuse to read the test
    # sets rather than evaluating half a model.
    print("=" * 78)
    print("SECTION 8 SKIPPED: V-HW or V-H is unavailable")
    print(f"  p08_train_vhw blocked: {blocked_reason('p08_train_vhw') or 'no weights'}")
    print(f"  p09_train_vh  blocked: {blocked_reason('p09_train_vh') or 'no weights'}")
    print("Test-A and Test-B are NOT read. Finish the training phases first.")
    print("=" * 78)
    RESULTS = {}
    FINAL_COMPARISON = None
    COVERAGE_SUMMARY = None
    BMI_ERRS = None
    TEST_IDS = {}
    TEST_LOCK = {"action": "skipped", "reason": "a production variant is unavailable"}

else:
    # --- A5: the single-use lock ---------------------------------------------------------------
    # The test sets may be read once, from one specific set of weights. A second evaluation of
    # the *same* weights is pointless and spends the budget, so it is refused too.
    def _best_hash(variant: str) -> str:
        """sha256 of a variant's best.pt.

        Parameters
        ----------
        variant : str
            Variant tag, e.g. ``'vhw'``.

        Returns
        -------
        str
        """
        p = RUNSTATE.ckpt_dir(variant) / "best.pt"
        if not p.is_file():
            return f"missing:{p.name}"
        return hashlib.sha256(p.read_bytes()).hexdigest()[:16]

    _WEIGHT_HASHES = {"V-HW": _best_hash("vhw"), "V-H": _best_hash("vh")}
    _FE_DIR = RUNSTATE.results_dir("p12_final_eval")
    TEST_LOCK = R.evaluate_test_lock(_FE_DIR, _WEIGHT_HASHES,
                                     allow_reeval=bool(cfg.get("allow_reeval_after_change")))
    print("=== single-use test-set lock (addendum A5) ===")
    print(f"  current weights: {_WEIGHT_HASHES}")
    print(f"  action: {TEST_LOCK['action']}")
    print(f"  {TEST_LOCK['reason']}")
    REEVAL_USED = bool(TEST_LOCK.get("reeval"))

    if TEST_LOCK["action"] == "reuse":
        print("\nReusing the cached final evaluation. Test-A/Test-B are not read again.")
        _prior = TEST_LOCK["prior"]
        RESULTS = _prior.get("results_payload", {})
        FINAL_COMPARISON = R.load_payload(_FE_DIR).get("final_comparison")
        FINAL_COMPARISON = (pd.DataFrame(FINAL_COMPARISON)
                            if isinstance(FINAL_COMPARISON, list) else FINAL_COMPARISON)
        COVERAGE_SUMMARY = R.load_payload(_FE_DIR).get("coverage_summary")
        COVERAGE_SUMMARY = (pd.DataFrame(COVERAGE_SUMMARY)
                            if isinstance(COVERAGE_SUMMARY, list) else COVERAGE_SUMMARY)
        BMI_ERRS = R.load_payload(_FE_DIR).get("bmi_errs")
        BMI_ERRS = pd.DataFrame(BMI_ERRS) if isinstance(BMI_ERRS, list) else BMI_ERRS
        TEST_IDS = {s: list(v) for s, v in (_prior.get("subjects") or {}).items()}
        REEVAL_USED = bool(_prior.get("reeval_after_change"))
        print(f"  loaded FINAL_COMPARISON with {len(FINAL_COMPARISON) if FINAL_COMPARISON is not None else 0} rows")

    elif TEST_LOCK["action"] == "blocked":
        # Per the chosen behaviour: block the evaluation but keep going, so the run still
        # produces a bundle. A hard raise here would take sections 9-11 down with it.
        print("\n" + "=" * 78)
        print("TEST-SET LOCK: Test-A/Test-B were already used, from different weights.")
        print(TEST_LOCK["reason"])
        print("Evaluation skipped. Sections 9-11 still run and will ship a bundle whose model card")
        print("records that the test metrics are absent and why.")
        print("=" * 78)
        RUNSTATE.mark_done("p12_final_eval", status="blocked",
                           meta={"reason": TEST_LOCK["reason"][:500],
                                 "weight_hashes": _WEIGHT_HASHES})
        RESULTS = {}
        FINAL_COMPARISON = None
        COVERAGE_SUMMARY = None
        BMI_ERRS = None
        TEST_IDS = {}
        REEVAL_USED = False

    else:
        TEST_IDS = {"testA": sorted(IDX["testA"].subjects.subject_id),
                    "testB": sorted(IDX["testB"].subjects.subject_id)}

        print("\n=== single-use protocol audit ===")
        train_all = set(ids_fit) | set(ids_val) | set(ids_calib)
        for split, ids in TEST_IDS.items():
            overlap = train_all & set(ids)
            print(f"  {split}: {len(ids)} subjects, overlap with any train split = {len(overlap)}")
            assert not overlap, f"{split} subjects leaked into training"
        train_photos = set(IDX["train"].photos.photo_id)
        for split in ("testA", "testB"):
            ov = train_photos & set(IDX[split].photos.photo_id)
            assert not ov, f"{split} photo ids overlap train ({len(ov)})"
            print(f"  {split}: photo-id overlap with train = 0")
        print("  Test-A/Test-B used exactly once, below this line.")

        FROZEN = {
            "V-HW": {"state_dict": {k: v.cpu() for k, v in net_vhw.state_dict().items()},
                     "conformal": CONFORMAL_QUANTILES.get("V-HW"), "use_weight": True,
                     "best_hash": _WEIGHT_HASHES["V-HW"]},
            "V-H": {"state_dict": {k: v.cpu() for k, v in net_vh.state_dict().items()},
                    "conformal": CONFORMAL_QUANTILES.get("V-H"), "use_weight": False,
                    "best_hash": _WEIGHT_HASHES["V-H"]},
        }
        print("\nfrozen artefacts:", {k: {"tensors": len(v["state_dict"]),
                                          "measurements": len(v["conformal"] or {}),
                                          "best_hash": v["best_hash"]}
                                      for k, v in FROZEN.items()})
        print("mark_phase('protocol audit') =", mark_phase("protocol audit"), "s")