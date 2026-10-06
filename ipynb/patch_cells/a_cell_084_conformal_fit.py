require("p08_train_vhw", "p09_train_vh")

# --- fix B5 / U2: calibrate on ONE randomly chosen photo pair per calibration subject -------
# Deployment predicts from one front/side pair. The previous code averaged a calibration
# subject's photos first, so the conformal scores were computed on lower-variance estimates
# than deployment ever sees, and the intervals under-covered for exactly that reason. The
# `unit` and `pairs_per_subject` fields written into conformal.json record this.
CALIB_RNG = np.random.default_rng(cfg["seed"] + 977)
_calib_photos = (IDX["train"].photos[IDX["train"].photos.subject_id.isin(set(ids_calib))]
                 .sort_values(["subject_id", "photo_id"]))
_calib_choice = (_calib_photos.groupby("subject_id", as_index=False)
                 .sample(n=1, random_state=int(cfg["seed"]) + 977)
                 .sort_values("subject_id"))
assert set(_calib_choice.subject_id) == set(ids_calib), (
    "one calibration pair per subject required; got "
    f"{_calib_choice.subject_id.nunique()} for {len(ids_calib)} subjects")
_calib_choice.to_csv(ART / "calibration_photo_pairs.csv", index=False)
print(f"conformal calibration unit: 1 photo pair per subject "
      f"({len(_calib_choice)} pairs from {len(ids_calib)} subjects)")
print("  chosen photo_id per subject, first 5:",
      list(_calib_choice.photo_id.head(5).astype(str)))
print("  photos per calibration subject available:",
      dict(_calib_photos.groupby("subject_id").size().value_counts().sort_index()))
assert len(_calib_choice) == len(ids_calib), "calibration unit must be one pair per subject"

CONFORMAL_QUANTILES: Dict[str, Dict[str, Dict[str, float]]] = {}
CONFORMAL_META: Dict[str, Dict] = {}

for name, model in (("V-HW", net_vhw), ("V-H", net_vh)):
    # restrict the index to the one chosen pair per calibration subject
    _calib_index = BodyMIndex(
        IDX["train"].split,
        IDX["train"].subjects[IDX["train"].subjects.subject_id.isin(set(ids_calib))].copy(),
        _calib_choice.copy(),
        IDX["train"].root)
    P_c, Y_c, s_c, p_c = predict_index(model, _calib_index, list(ids_calib), TARGETS, cfg, device,
                                       batch_size=cfg["batch_size"])
    assert sorted(set(map(str, s_c))) == sorted(ids_calib), "calibration subject set mismatch"
    assert len(P_c) == len(ids_calib), (
        f"{name}: expected exactly one prediction per calibration subject, got {len(P_c)}")

    q = C.fit_conformal(Y_c, P_c, TARGETS, cfg["coverages"])
    CONFORMAL_QUANTILES[name] = q

    # Empirical coverage ON the calibration split: a sanity floor, not a guarantee. The real
    # coverage estimate is computed on Test-A/Test-B in section 8.
    emp = {}
    iv = C.apply_intervals(P_c, q, TARGETS)
    m_cal = M.per_measurement_metrics(Y_c, P_c, TARGETS, intervals=iv)
    for cov in cfg["coverages"]:
        emp[f"coverage_{int(cov*100)}"] = m_cal[f"coverage_{int(cov*100)}"].mean()

    CONFORMAL_META[name] = {
        "unit": "photo_pair",
        "pairs_per_subject": 1,
        "n_calibration_subjects": len(ids_calib),
        "n_calibration_predictions": len(P_c),
        "calib_pair_mae_mm": float(np.abs(P_c - Y_c).mean() * 10.0),
        "empirical_coverage_on_calibration_pct": emp,
    }
    print(f"\n=== {name}: conformal quantiles (cm half-widths) from n={len(P_c)} "
          f"photo pairs, one per subject ===")
    qt = pd.DataFrame([{"measurement": m, **{f"q{int(k)}": v for k, v in d.items()}}
                       for m, d in q.items()])
    # fix B12: q is a half-width, so mm is q*10, not q*20
    qt["halfwidth80_mm"] = qt["q80"] * 10.0
    qt["halfwidth90_mm"] = qt["q90"] * 10.0
    qt["width90_mm"] = 2.0 * qt["q90"] * 10.0
    print(M.format_metric_table(qt))
    print("calibration-pair diagnostics:", json.dumps(CONFORMAL_META[name], indent=2))

    for meas in TARGETS:
        assert 0 <= q[meas]["80"] <= q[meas]["90"], f"non-monotone quantiles for {meas}"
        assert q[meas]["90"] >= q[meas]["80"] > 0, f"degenerate quantile for {meas}"
    assert emp["coverage_80"] >= 79.0, (
        f"{name}: empirical 80% coverage on calibration is {emp['coverage_80']:.1f}%, below "
        "nominal. A finite-sample-corrected quantile must cover on the data it was fitted to; "
        "if it does not, the quantile rule has been changed.")
    print(f"  {name}: calibration 80% coverage {emp['coverage_80']:.1f}%, "
          f"90% coverage {emp['coverage_90']:.1f}%")