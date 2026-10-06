require("p08_train_vhw", "p09_train_vh")

# --- optional: one summary figure for the bundle -------------------------------------------
if not RUN_FINAL_EVAL:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    _w = mae_tbl.set_index("measurement")[["mae_B0_mm", "mae_B1_mm", "mae_B2_mm",
                                           "mae_VHW_mm", "mae_VH_mm"]]
    _w.plot(kind="barh", ax=axes[0], width=0.8)
    axes[0].set_title("Validation MAE, per photo pair (mm)")
    axes[0].set_xlabel("MAE (mm)"); axes[0].legend(fontsize=7)
    _b = MAX_TOLERABLE_BOUNDARY[(MAX_TOLERABLE_BOUNDARY.model == "V-HW")
                                & (MAX_TOLERABLE_BOUNDARY.family == "morph")
                                & (MAX_TOLERABLE_BOUNDARY.criterion == cfg["max_tolerable"])
                                & (MAX_TOLERABLE_BOUNDARY.tolerance_mm == cfg["tolerance_mm"][-1])]
    if len(_b):
        _p = _b.pivot_table(index="measurement", columns="side", values="max_tolerable_px")
        _p.plot(kind="barh", ax=axes[1], width=0.8)
        axes[1].set_title(f"V-HW max tolerable erosion/dilation at "
                          f"{cfg['tolerance_mm'][-1]:.0f} mm (px)")
        axes[1].set_xlabel("tolerable boundary error (px)")
    axes[1].legend(fontsize=7)
    for ax in axes:
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("BodyM measurement model — validation summary "
                 "(final evaluation not run: run_final_eval=false)")
else:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))
    _summary = FINAL_COMPARISON.pivot_table(index="measurement", columns=["split", "model"],
                                            values="mae_mm")
    _summary[("testA", "V-HW")].plot(kind="barh", ax=axes[0, 0], width=0.8, legend=False)
    axes[0, 0].set_title("V-HW MAE, Test-A (mm), per photo pair")
    axes[0, 0].axvline(25, ls="--", c="k", lw=0.8)
    axes[0, 0].set_xlabel("MAE (mm)")

    _summary[("testB", "V-HW")].plot(kind="barh", ax=axes[0, 1], width=0.8, legend=False)
    axes[0, 1].set_title("V-HW MAE, Test-B (mm), per photo pair")
    axes[0, 1].axvline(25, ls="--", c="k", lw=0.8)
    axes[0, 1].set_xlabel("MAE (mm)")

    axes[1, 0].plot(COVERAGE_SUMMARY.split + "/" + COVERAGE_SUMMARY.variant,
                    COVERAGE_SUMMARY.empirical_90, "o", label="empirical 90%")
    axes[1, 0].axhline(90, ls="--", c="k", lw=0.9, label="nominal 90%")
    axes[1, 0].set_ylim(60, 105); axes[1, 0].set_ylabel("coverage (%)")
    axes[1, 0].set_title("Conformal coverage on held-out splits"); axes[1, 0].legend(fontsize=7)

    _w = mae_tbl.set_index("measurement")[["mae_B0_mm", "mae_B1_mm", "mae_B2_mm",
                                           "mae_VHW_mm", "mae_VH_mm"]]
    _w.plot(kind="barh", ax=axes[1, 1], width=0.8)
    axes[1, 1].set_title(f"Validation MAE, per photo pair "
                         f"(PROVISIONAL at {TRAIN_STEPS['V-HW']} steps)")
    axes[1, 1].set_xlabel("MAE (mm)"); axes[1, 1].legend(fontsize=7)
    for ax in axes.ravel():
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("BodyM measurement model — run summary")

fig.tight_layout()
fig.savefig(EXPORT_DIR / "fig_summary.png", bbox_inches="tight")
plt.show(); plt.close(fig)
print("summary figure written:", EXPORT_DIR / "fig_summary.png")

if cfg["zip_export"]:
    zip_path = shutil.make_archive(str(PATHS["working"] / "export"), "zip", root_dir=EXPORT_DIR)
    print(f"re-zipped with summary figure: {zip_path} "
          f"({Path(zip_path).stat().st_size/1e6:.1f} MB)")
print(f"\nDONE. The zip is at {PATHS['working'] / 'export.zip'} "
      f"(/kaggle/working/export.zip). Kaggle's /kaggle/outputs directory is only populated on "
      f"Save Version; download it from the notebook's output pane.")
print("If run_final_eval was false, set it to true and Save & Run All once to populate "
      "final_comparison.csv, the held-out coverage table and the model card's test metrics.")
print("See DECISIONS.md for the decision log and the open items for the user.")