require("p04_b0", "p05_b1", "p08_train_vhw", "p09_train_vh")

# --- the one comparison table brief section 10 asks for ------------------------------------
if not RUN_FINAL_EVAL:
    print("=== MAE (mm) — all models, both test splits ===")
    print("FINAL EVALUATION NOT RUN (config.yaml run_final_eval: false).")
    print("Run once with run_final_eval: true to populate final_comparison.csv.")
    FINAL_COMPARISON = None
    COVERAGE_SUMMARY = None
    fig = None

else:
    rows = []
    for split in ("testA", "testB"):
        for model in ("B0", "B1", "V-HW", "V-H"):
            m = RESULTS[split]["metrics"][model].set_index("measurement")
            for meas in TARGETS:
                row = {"split": split, "model": model, "measurement": meas,
                       "unit": "subject" if model in ("B0", "B1") else "photo_pair",
                       "n": int(m.loc[meas, "n"]),
                       "mae_mm": m.loc[meas, "mae_mm"], "bias_mm": m.loc[meas, "bias_mm"],
                       "tp50_mm": m.loc[meas, "tp50_mm"], "tp90_mm": m.loc[meas, "tp90_mm"]}
                for tol in cfg["tolerance_mm"]:
                    row[f"pct_within_{int(tol)}mm"] = m.loc[meas, f"pct_within_{int(tol)}mm"]
                if "coverage_80" in m.columns and pd.notna(m.loc[meas, "coverage_80"]):
                    row["coverage_80"] = m.loc[meas, "coverage_80"]
                    row["coverage_90"] = m.loc[meas, "coverage_90"]
                    row["width_90_mm"] = m.loc[meas, "width_90_mm"]
                rows.append(row)

    FINAL_COMPARISON = pd.DataFrame(rows)
    FINAL_COMPARISON.to_csv(ART / "final_comparison.csv", index=False)
    assert {"unit", "coverage_80", "coverage_90"}.issubset(FINAL_COMPARISON.columns), (
        "FINAL_COMPARISON must record the evaluation unit per row and the coverage columns")
    print("evaluation unit per row:",
          FINAL_COMPARISON.groupby(["model", "unit"]).n.sum().to_dict())

    print("\n=== MAE (mm) — all models, both test splits ===")
    mae_pivot = FINAL_COMPARISON.pivot_table(index=["measurement"], columns=["split", "model"],
                                             values="mae_mm")
    print(M.format_metric_table(mae_pivot.reset_index()))

    print("\n=== % within 25 mm ===")
    print(M.format_metric_table(FINAL_COMPARISON.pivot_table(index="measurement",
                                                            columns=["split", "model"],
                                                            values="pct_within_25mm").reset_index()))

    # secondary subject-averaged table, clearly labelled
    _sa_rows = []
    for split in ("testA", "testB"):
        for vname, tbl in RESULTS[split]["subject_avg"].items():
            t = tbl.set_index("measurement")
            for meas in TARGETS:
                _sa_rows.append({"split": split, "variant": vname, "measurement": meas,
                                 "n": int(t.loc[meas, "n"]),
                                 "mae_mm": t.loc[meas, "mae_mm"],
                                 "unit": "subject_averaged"})
    SUBJECT_AVG_COMPARISON = pd.DataFrame(_sa_rows)
    SUBJECT_AVG_COMPARISON.to_csv(ART / "subject_averaged_comparison.csv", index=False)
    print("\n=== SECONDARY: subject-averaged MAE (mm) — NOT the deployment unit ===")
    print(SUBJECT_AVG_COMPARISON.pivot_table(index="measurement", columns=["split", "variant"],
                                             values="mae_mm").round(2).to_string())

    cov_rows = []
    for split in ("testA", "testB"):
        for vname in ("V-HW", "V-H"):
            m = RESULTS[split]["metrics"][vname]
            cov_rows.append({"split": split, "variant": vname, "unit": "photo_pair",
                             "n": int(m["n"].mean()),
                             "nominal_80": 80.0, "empirical_80": m["coverage_80"].mean(),
                             "nominal_90": 90.0, "empirical_90": m["coverage_90"].mean(),
                             "mean_width_80_mm": m["width_80_mm"].mean(),
                             "mean_width_90_mm": m["width_90_mm"].mean()})
    COVERAGE_SUMMARY = pd.DataFrame(cov_rows)
    COVERAGE_SUMMARY.to_csv(ART / "coverage_summary.csv", index=False)
    print("\n=== conformal coverage on truly held-out splits (per photo pair) ===")
    print(M.format_metric_table(COVERAGE_SUMMARY))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    ax = mae_pivot[("testA", "V-HW")].plot(kind="barh", ax=axes[0], width=0.8, label="Test-A")
    _ = mae_pivot[("testB", "V-HW")].plot(kind="barh", ax=axes[0], width=0.4, label="Test-B")
    axes[0].set_title("V-HW MAE (mm): Test-A vs Test-B")
    axes[0].set_xlabel("MAE (mm), per photo pair"); axes[0].legend(); axes[0].grid(axis="x", alpha=0.25)

    for split, ls in (("testA", "-"), ("testB", "--")):
        d = FINAL_COMPARISON[(FINAL_COMPARISON.split == split) & (FINAL_COMPARISON.model == "V-HW")]
        axes[1].errorbar(d.tp50_mm, np.arange(len(d)), xerr=[d.tp90_mm - d.tp50_mm], fmt="o",
                         ls=ls, capsize=3, label=split)
    axes[1].set_yticks(np.arange(len(d)))
    axes[1].set_yticklabels(d.measurement)
    axes[1].set_xlabel("absolute error (mm): marker = TP50, bar to TP90")
    axes[1].set_title("Error distribution, V-HW")
    axes[1].axvline(15, ls=":", c="tab:gray", lw=0.8); axes[1].legend(); axes[1].grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(ART / "fig_testA_vs_testB.png", bbox_inches="tight")
    plt.show(); plt.close(fig)

print("mark_phase('comparison table') =", mark_phase("comparison table"), "s")