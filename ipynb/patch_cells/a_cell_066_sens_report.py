require("p06_train_b2")

fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.2), sharey=True)
for ax, model in zip(axes, ["B1", "B2"]):
    sub = SENSITIVITY[(SENSITIVITY.model == model) & (SENSITIVITY.family == "morph")]
    for meas in TARGETS:
        s = sub[sub.measurement == meas].sort_values("magnitude")
        if len(s) == len(range(-cfg["erosion_px"], cfg["erosion_px"] + 1)):
            ax.plot(s.magnitude, s.delta_mae_mm, marker=".", lw=1.1, ms=3.5, label=meas)
    ax.axhline(0, c="k", lw=0.7)
    for tol in cfg["tolerance_mm"]:
        ax.axhline(tol, ls="--", lw=0.7, c="tab:gray")
        ax.text(-cfg["erosion_px"] + 0.3, tol, f"{tol} mm (delta)", fontsize=6.5,
                va="bottom", c="tab:gray")
    ax.set_title(f"{model}: erosion/dilation sensitivity")
    ax.set_xlabel("boundary offset k (px), negative = eroded")
    ax.legend(fontsize=6, ncol=2)
axes[0].set_ylabel("ΔMAE vs unperturbed (mm)")
fig.suptitle("ΔMAE = perturbation-induced error. A tolerable-k budget must also respect the "
             "model's own baseline MAE — see the total/delta table below.", fontsize=8)
fig.tight_layout(); fig.savefig(ART / "fig_sensitivity_morph.png", bbox_inches="tight")
plt.show(); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.0))
for ax, model in zip(axes, ["B1", "B2"]):
    sub = SENSITIVITY[(SENSITIVITY.model == model) & (SENSITIVITY.family != "morph")
                      & (SENSITIVITY.family != "none")]
    for meas in TARGETS:
        for fam, ls in zip(["jitter", "downsample"], ["-", "--"]):
            s = sub[(sub.measurement == meas) & (sub.family == fam)].sort_values("magnitude")
            if len(s):
                ax.plot(s.magnitude, s.delta_mae_mm, ls, marker="o", lw=1.1, ms=3.5,
                        label=f"{meas}/{fam}")
    ax.axhline(0, c="k", lw=0.7)
    ax.set_title(f"{model}: jitter and resolution sensitivity")
    ax.set_xlabel("jitter sigma (px) / downsample factor")
axes[0].set_ylabel("ΔMAE vs unperturbed (mm)")
axes[1].legend(fontsize=5.5, ncol=2)
fig.tight_layout(); fig.savefig(ART / "fig_sensitivity_other.png", bbox_inches="tight")
plt.show(); plt.close(fig)

MM_PER_PX_NOMINAL = float(np.median(EDA["mm_per_px"]))
print(f"nominal scale used for px->mm: {MM_PER_PX_NOMINAL:.3f} mm/px "
      f"(IQR {np.percentile(EDA['mm_per_px'],25):.3f}-{np.percentile(EDA['mm_per_px'],75):.3f})")

# --- fix B8: total-error criterion, contiguity, erosion/dilation split, factor not mm -------
MAX_TOLERABLE_BOUNDARY = S.max_tolerable_boundary_error(
    SENSITIVITY, cfg["key_measurements"], cfg["tolerance_mm"], MM_PER_PX_NOMINAL,
    mode=cfg.get("max_tolerable", "total"))
MAX_TOLERABLE_BOUNDARY.to_csv(ART / "max_tolerable_boundary_error.csv", index=False)
assert not MAX_TOLERABLE_BOUNDARY.empty, "max_tolerable_boundary_error produced no rows"

for _model in ("B2", "B1"):
    _sub = MAX_TOLERABLE_BOUNDARY[(MAX_TOLERABLE_BOUNDARY.model == _model)
                                  & (MAX_TOLERABLE_BOUNDARY.criterion == cfg["max_tolerable"])]
    print(f"\n=== maximum tolerable boundary error — {_model}, "
          f"{cfg['max_tolerable']}-error criterion ===")
    print(M.format_metric_table(_sub.pivot_table(
        index=["measurement", "family", "side"], columns="tolerance_mm",
        values=["max_tolerable_px", "max_tolerable_mm"], dropna=False).reset_index()))

_tot = MAX_TOLERABLE_BOUNDARY[MAX_TOLERABLE_BOUNDARY.criterion == "total"]
_del = MAX_TOLERABLE_BOUNDARY[MAX_TOLERABLE_BOUNDARY.criterion == "delta"]
_none_rows = _tot[_tot.max_tolerable_px.astype(str) == "none tolerable"]
print(f"\nrows where NO magnitude passes at the loosest tolerance "
      f"({cfg['tolerance_mm'][-1]} mm): {len(_none_rows)} "
      f"-> reported as 'none tolerable', not 0.0")
_ds_units = sorted(set(MAX_TOLERABLE_BOUNDARY[
    MAX_TOLERABLE_BOUNDARY.family == "downsample"].unit))
print(f"downsample rows carry unit={_ds_units} and their px/mm columns are NaN "
      "(a factor is not a distance, so it is never converted)")
assert (MAX_TOLERABLE_BOUNDARY[MAX_TOLERABLE_BOUNDARY.family == "downsample"]
        ["max_tolerable_px"].isna()).all(), "downsample rows must not report px"
assert set(MAX_TOLERABLE_BOUNDARY.side.unique()) >= {"erosion", "dilation"}, \
    "erosion and dilation must be reported separately (fix B8)"

# --- fix B10: augmentation magnitudes require ALL measurements (except `height`) to pass -----
AUG_MAG = S.augmentation_magnitudes(SENSITIVITY, TARGETS, cfg, model="B2")
AUG_MAG["mm_per_px"] = MM_PER_PX_NOMINAL
print("\nAUG_MAG derived from Phase 4 (B2 sweep, total-error criterion):")
print(yaml.safe_dump(AUG_MAG))
print(f"  erosion ±{AUG_MAG['erosion_px']} px | dilation +{AUG_MAG['dilation_px']} px "
      f"| jitter sigma {AUG_MAG['jitter_sigma_px']} px "
      f"| downsample x{AUG_MAG['downsample_factor']}")
print(f"  rule: every measurement except {AUG_MAG['excluded_targets']} must stay within "
      f"{AUG_MAG['tolerance_mm_used']} mm (fix B10)")
print(f"  binding (first-failing) measurement: {AUG_MAG['binding_measurement']}")
assert AUG_MAG["binding_measurement"] is None or isinstance(AUG_MAG["binding_measurement"], str)
print("  note: the augmentation magnitudes come from the B2 sweep, which must precede "
      "training. The V-HW/V-H sweep in the next cell is the one the model card reports.")
(ART / "augmentation_magnitudes.yaml").write_text(yaml.safe_dump(AUG_MAG))
print("mark_phase('sensitivity report') =", mark_phase("sensitivity report"), "s")