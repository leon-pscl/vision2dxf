if RUN_FINAL_EVAL:
    for split in ("testA", "testB"):
        # fix A5: df.groupby("stratum").n() raises
        # "TypeError: 'SeriesGroupBy' object is not callable" on every pandas that has an `n`
        # column - and this table always has one. Use bracket indexing.
        for vname in ("V-HW", "V-H"):
            for stratum_key, label in (("", "sex"), ("_bmi", "bmi")):
                tbl = RESULTS[split]["stratified"][vname + stratum_key]
                n_subj = M.stratum_sizes(tbl["stratum"])
                n_per_row = tbl.groupby("stratum")["n"].first()
                print(f"\n--- {split} / {vname} / stratified by {label} "
                      f"(rows per stratum: {n_subj}) ---")
                print("    n per stratum:", n_per_row.to_dict())
                print(M.format_metric_table(tbl[tbl.measurement.isin(cfg["key_measurements"])]))
                unrel = tbl[~tbl.reliable]
                if len(unrel):
                    unrel_n = (unrel.groupby("stratum")["n"].first())
                    print(f"  !! n < {cfg['min_stratum_n']}: "
                          f"{ {str(k): int(v) for k, v in unrel_n.items()} } "
                          f"-> NOT RELIABLE, reporting only")

    # per-subject BMI error growth: the high-BMI failure mode, quantified
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, split in zip(axes, ("testA", "testB")):
        meta = RESULTS[split]["meta"]
        preds = RESULTS[split]["preds"]
        Yt = preds["V-HW_y"]
        bmi_rows = meta.loc[preds["V-HW_subjects"], "bmi"].to_numpy()
        for vname, ls in (("V-HW", "-"), ("V-H", "--")):
            P = preds[vname]
            for col, c in (("chest", "tab:blue"), ("waist", "tab:orange"),
                           ("hip", "tab:green")):
                j = TARGETS.index(col)
                ax.scatter(bmi_rows, np.abs(P[:, j] - Yt[:, j]) * 10, s=9, alpha=0.45,
                           ls=ls, color=c, label=f"{col}/{vname}")
        for _band, _lab in enumerate(cfg["bmi_labels"]):
            ax.axvline([18.5, 25, 30, 40][_band] if _band < 4 else 45, ls=":", lw=0.6,
                       c="gray")
        ax.set_xlabel("BMI (kg/m$^2$)"); ax.set_ylabel("absolute error (mm)")
        ax.set_title(f"{split}: error vs BMI (per photo pair)")
        ax.legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(ART / "fig_bmi_failure_mode.png", bbox_inches="tight")
    plt.show(); plt.close(fig)

    # quantify it
    bmi_rows = []
    for split in ("testA", "testB"):
        meta = RESULTS[split]["meta"]
        preds = RESULTS[split]["preds"]
        for vname in ("V-HW", "V-H"):
            P = preds[vname]
            Yt = preds[vname + "_y"]
            subj = preds[vname + "_subjects"]
            bands = bmi_band(meta.loc[subj, "bmi"], cfg["bmi_bands"]).astype(int)
            for lab_i, lab in enumerate(cfg["bmi_labels"]):
                sel = (bands == lab_i).to_numpy()
                if sel.sum() == 0:
                    continue
                for col in cfg["key_measurements"]:
                    j = TARGETS.index(col)
                    bmi_rows.append({"split": split, "variant": vname, "bmi_band": lab,
                                     "measurement": col, "n": int(sel.sum()),
                                     "n_subjects": int(len(set(map(str, subj[sel])))),
                                     "mae_mm": float(np.abs(P[sel, j] - Yt[sel, j]).mean() * 10),
                                     "reliable": bool(sel.sum() >= cfg["min_stratum_n"])})
    BMI_ERRS = pd.DataFrame(bmi_rows)
    BMI_ERRS.to_csv(ART / "bmi_stratum_errors.csv", index=False)
    print("\n=== key-measurement MAE by BMI band (V-HW, per photo pair) ===")
    print(M.format_metric_table(
        BMI_ERRS[(BMI_ERRS.variant == "V-HW")].pivot_table(
            index="bmi_band", columns=["split", "measurement"],
            values="mae_mm").reset_index().rename(columns={"index": "bmi_band"})))
    print("\nmarked unreliable (n < %d):" % cfg["min_stratum_n"],
          {str(k): int(v) for k, v in
           BMI_ERRS[~BMI_ERRS.reliable].groupby(["split", "bmi_band"])["n"].first().items()})
    assert "n" in BMI_ERRS.columns and "reliable" in BMI_ERRS.columns, \
        "bmi_stratum_errors.csv must carry n and reliable"
else:
    print("=== stratified report: SKIPPED (run_final_eval: false) ===")
    print("No Test-A/Test-B data was read.")
print("mark_phase('stratified report') =", mark_phase("stratified report"), "s")