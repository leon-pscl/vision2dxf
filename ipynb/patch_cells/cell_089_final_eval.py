# --- B0 and B1 on the test splits (fitted on fit-only, never refitted) ----------------------
def evaluate_b0_b1(index: BodyMIndex, subject_ids: Sequence[str]) -> Dict[str, np.ndarray]:
    """Predict with the frozen B0 and B1 models on one split.

    Parameters
    ----------
    index : BodyMIndex
        Assembled index for the split.
    subject_ids : sequence of str
        Subjects to predict, one photo each.

    Returns
    -------
    dict
        ``{'y_true': (n, m), 'b0': (n, m), 'b1': (n, m), 'subjects': list}``.

    Notes
    -----
    ``y_true`` is indexed by ``subjects`` explicitly rather than by the order
    ``subject_table`` happens to return. The previous version built ``y_true`` from
    ``subject_table`` (whose row order follows ``index.subjects``) while ``b1_matrix``
    returns sorted ids, and then asserted the two agreed — an assertion that only passed
    because ``subjects`` happened to be pre-sorted. Both now use ``subjects``.
    """
    order = sorted(subject_ids)
    meta = index.subjects.set_index("subject_id")
    Y = meta.loc[order, list(TARGETS)].to_numpy(float)

    tbl = subject_table(index, order).set_index("subject_id")
    assert list(tbl.index) == order, "subject_table must be reindexed to the requested order"
    P0 = b0.predict(b0_features(tbl.reset_index()), which="gbm")
    Ft, _, ids = b1_matrix(index, order, "B1 eval")
    assert ids == order, "b1_matrix returned a different subject order"
    P1 = b1.predict(Ft)
    return {"y_true": Y, "b0": P0, "b1": P1, "subjects": order}


RESULTS: Dict[str, Dict[str, object]] = {}

if RUN_FINAL_EVAL:
    for split in ("testA", "testB"):
        ids = TEST_IDS[split]
        idx = IDX[split]
        print(f"\n===== evaluating {split} ({len(ids)} subjects) =====")

        ev = evaluate_b0_b1(idx, ids)
        Y_t = ev["y_true"]
        assert np.allclose(Y_t, idx.subjects.set_index("subject_id")
                           .loc[ev["subjects"], TARGETS].to_numpy(float), atol=1e-6), \
            "B0/B1 ground truth order mismatch"

        preds = {"B0": ev["b0"], "B1": ev["b1"]}
        for vname, model in (("V-HW", net_vhw), ("V-H", net_vh)):
            # fix B5: evaluate per photo pair, the way deployment will predict
            P, Yv, sv, pv = predict_index(model, idx, ids, TARGETS, cfg, device,
                                          batch_size=cfg["batch_size"],
                                          max_photos_per_subject=_SENS_PAIRS)
            assert sorted(set(map(str, sv))) == ev["subjects"], f"{split}/{vname} subject set mismatch"
            # predict_index already returns this subject's target row per pair; use it directly
            # rather than re-deriving it, so there is one source of truth for the alignment
            assert Yv.shape == P.shape, (
                f"{split}/{vname}: prediction rows {P.shape} vs target rows {Yv.shape}")
            preds[vname] = P
            preds[vname + "_y"] = Yv
            preds[vname + "_subjects"] = sv
            preds[vname + "_photo_ids"] = pv
            iv = C.apply_intervals(P, CONFORMAL_QUANTILES[vname], TARGETS)
            iv = C.clamp_intervals(iv, PLAUSIBLE_CM)
            preds[vname + "_intervals"] = iv

        print("  B0 / B1 (one photo per subject):",
              photos_per_subject_note(idx, ev["subjects"]))
        print("  V-HW / V-H (per photo pair):     ",
              photos_per_subject_note(idx, sorted(set(map(str, preds["V-HW_subjects"])))))

        metrics = {}
        metrics["B0"] = M.per_measurement_metrics(Y_t, preds["B0"], TARGETS)
        metrics["B1"] = M.per_measurement_metrics(Y_t, preds["b1"], TARGETS)
        for vname in ("V-HW", "V-H"):
            metrics[vname] = M.per_measurement_metrics(preds[vname + "_y"], preds[vname],
                                                      TARGETS,
                                                      intervals=preds[vname + "_intervals"])

        # secondary subject-averaged tables, clearly labelled
        subject_avg = {}
        for vname in ("V-HW", "V-H"):
            ap, at, asub = T.aggregate_by_subject(preds[vname], preds[vname + "_y"],
                                                  preds[vname + "_subjects"])
            subject_avg[vname] = M.per_measurement_metrics(at, ap, TARGETS)
        print("\n  SECONDARY (subject-averaged, not the deployment unit):")
        for vname in ("V-HW", "V-H"):
            print(f"    {vname}: MAE "
                  + ", ".join(f"{c}={subject_avg[vname].set_index('measurement').loc[c,'mae_mm']:.1f}"
                              for c in cfg["key_measurements"]) + " mm")

        meta = idx.subjects.set_index("subject_id").loc[ev["subjects"]]
        strata_sex = meta["sex"].map({0: "female", 1: "male"}).to_numpy()
        strata_bmi = bmi_band(meta["bmi"], cfg["bmi_bands"]).astype(int).map(
            dict(enumerate(cfg["bmi_labels"]))).to_numpy()

        stratified = {}
        for name in ("B0", "V-HW", "V-H"):
            if name == "B0":
                stratified[name] = M.stratified_metrics(Y_t, preds["B0"], TARGETS, strata_sex,
                                                        cfg["min_stratum_n"])
                stratified[name + "_bmi"] = M.stratified_metrics(Y_t, preds["B0"], TARGETS,
                                                                 strata_bmi, cfg["min_stratum_n"])
            else:
                # strata per vision row, aligned with preds[name + "_subjects"]
                m_sub = meta["sex"].map({0: "female", 1: "male"}).to_dict()
                b_sub = bmi_band(meta["bmi"], cfg["bmi_bands"]).astype(int).map(
                    dict(enumerate(cfg["bmi_labels"]))).to_dict()
                row_sex = np.asarray([m_sub[s] for s in preds[name + "_subjects"]], dtype=object)
                row_bmi = np.asarray([b_sub[s] for s in preds[name + "_subjects"]], dtype=object)
                iv = preds[name + "_intervals"]
                stratified[name] = M.stratified_metrics(preds[name + "_y"], preds[name],
                                                        TARGETS, row_sex, cfg["min_stratum_n"],
                                                        intervals=iv)
                stratified[name + "_bmi"] = M.stratified_metrics(preds[name + "_y"], preds[name],
                                                                 TARGETS, row_bmi,
                                                                 cfg["min_stratum_n"],
                                                                 intervals=iv)

        RESULTS[split] = {"subjects": ev["subjects"], "y_true": Y_t, "preds": preds,
                          "metrics": metrics, "subject_avg": subject_avg,
                          "stratified": stratified, "meta": meta,
                          "strata_sex": strata_sex, "strata_bmi": strata_bmi}

        print()
        print(M.format_metric_table(metrics["V-HW"]))
        assert len(metrics) == 4 and set(metrics) == {"B0", "B1", "V-HW", "V-H"}
        for name in ("V-HW", "V-H"):
            assert "coverage_80" in metrics[name] and "coverage_90" in metrics[name], \
                f"{name}: conformal coverage missing on {split}"
    print("\nmark_phase('final eval') =", mark_phase("final eval"), "s")