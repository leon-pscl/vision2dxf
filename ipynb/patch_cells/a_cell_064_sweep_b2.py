require("p06_train_b2")

import bodym_sensitivity as S
from bodym_perturb import (WorkerRNG, apply_perturbation, boundary_jitter, downsample_upsample,
                           jitter_area_stats, morph, training_augmenter)

grid = S.build_grid(cfg)
print(f"sweep grid: {len(grid)} conditions ->", grid)
assert grid[0] == ("none", 0.0), "the identity condition must be first in the grid"

# --- fix B7: an independent per-pair reference MAE for the sensitivity subjects -------------
# The old self-check asserted dMAE at ('none', 0) was ~0, which is true by construction: dMAE is
# the difference from that very row. It could not detect a broken sweep. The replacement computes
# the unperturbed MAE twice by independent code paths and compares them.
N_SENS = int(cfg["smoke_sensitivity_subjects"]) if SMOKE else int(cfg["n_sensitivity_subjects"])

rng = np.random.default_rng(cfg["seed"])
n_sub = min(N_SENS, len(ids_val))
sens_ids = sorted(rng.choice(ids_val, size=n_sub, replace=False).tolist())
# B1's calibration is refitted under each perturbation on this disjoint subset of the FIT split,
# never on the subjects being evaluated.
sens_fit_ids = sorted(rng.choice(ids_fit, size=min(n_sub, len(ids_fit)), replace=False).tolist())
assert not (set(sens_ids) & set(sens_fit_ids)), "sensitivity fit/eval subjects overlap"
_pps = IDX["train"].photos[IDX["train"].photos.subject_id.isin(sens_ids)]
print(f"sensitivity: {n_sub} validation subjects ({len(_pps)} photo pairs) + "
      f"{len(sens_fit_ids)} fit subjects for the B1 refit")
print(f"  photos per subject in the sensitivity set: "
      f"{dict(_pps.groupby('subject_id').size().value_counts().sort_index())}")

# --- independent reference MAE for B2 on these subjects, per photo pair ---------------------
_ref_loader = DataLoader(
    SilhouetteDataset(IDX["train"], sens_ids, TARGETS,
                      img_height=cfg["img_height"], img_width=cfg["img_width"],
                      threshold=cfg["mask_threshold"], use_weight=True,
                      max_photos_per_subject=SMOKE_PHOTOS),
    batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])
_ref_pred, _ref_true, _ref_subs = T.predict_split(net_b2, _ref_loader, device)
REF_MAE_B2_MM: Dict[str, float] = {
    t: float(v) for t, v in zip(TARGETS, np.abs(_ref_pred - _ref_true).mean(axis=0) * 10.0)}
print("independent per-pair B2 reference MAE (mm), first 5:",
      {k: round(v, 2) for k, v in list(REF_MAE_B2_MM.items())[:5]})

sens_b1 = S.sweep_b1(IDX["train"], sens_ids, TARGETS, cfg, grid, seed=cfg["seed"],
                     fit_subject_ids=sens_fit_ids)
print("B1 sweep done:", sens_b1.shape)
# B1's identity row must equal a direct unperturbed B1 evaluation, recomputed here
_b1_F_fit, _b1_Y_fit, _ = b1_matrix(IDX["train"], sens_fit_ids, "B1 refit")
_b1_F_val, _b1_Y_val, _ = b1_matrix(IDX["train"], sens_ids, "B1 ref eval")
_b1_direct = B1Geometric(threshold=cfg["mask_threshold"]).fit(_b1_F_fit, _b1_Y_fit, TARGETS)
_b1_direct_mae = {t: float(v) for t, v in
                  zip(TARGETS, np.abs(_b1_direct.predict(_b1_F_val) - _b1_Y_val).mean(axis=0) * 10.0)}
_b1_none = sens_b1[sens_b1.family == "none"].set_index("measurement")["mae_mm"].to_dict()
print("  B1 identity sweep MAE vs direct recomputation, max |diff| (mm):",
      f"{max(abs(_b1_none[t] - _b1_direct_mae[t]) for t in TARGETS):.3e}")
assert max(abs(_b1_none[t] - _b1_direct_mae[t]) for t in TARGETS) < 1e-6, (
    "B1 identity-condition MAE disagrees with an independent recomputation - the sweep harness "
    "is inconsistent (fix B7)")

sens_b2 = S.sweep_b2(net_b2, IDX["train"], sens_ids, TARGETS, cfg, grid, device, seed=cfg["seed"])
print("B2 sweep done:", sens_b2.shape)

# --- fix B7: real cross-checks on the B2 sweep ---------------------------------------------
_b2_none = sens_b2[sens_b2.family == "none"].set_index("measurement")["mae_mm"].to_dict()
_b2_none_gap = max(abs(_b2_none[t] - REF_MAE_B2_MM[t]) for t in TARGETS)
print("  B2 identity sweep MAE vs independent per-pair MAE, max |diff| (mm):", f"{_b2_none_gap:.3e}")
assert _b2_none_gap < 1e-6, (
    f"B2 identity-condition MAE disagrees with the independently computed per-pair MAE by "
    f"{_b2_none_gap:.3e} mm - the sweep harness is inconsistent (fix B7)")

_k0 = sens_b2[(sens_b2.family == "morph") & (sens_b2.magnitude == 0.0)].set_index("measurement")["mae_mm"]
_k0_gap = max(abs(_k0[t] - _b2_none[t]) for t in TARGETS)
print("  B2 morph k=0 MAE vs identity MAE, max |diff| (mm):", f"{_k0_gap:.3e}")
assert _k0_gap < 1e-6, "morph k=0 must be exactly the identity condition (fix B7)"

SENSITIVITY = pd.concat([sens_b1, sens_b2], ignore_index=True)
SENSITIVITY_CSV = ART / "sensitivity.csv"
SENSITIVITY.to_csv(SENSITIVITY_CSV, index=False)
print("written:", SENSITIVITY_CSV)

combos = SENSITIVITY.groupby(["model", "family", "magnitude"]).measurement.nunique()
assert (combos == len(TARGETS)).all(), "sweep does not cover every measurement"
assert set(SENSITIVITY.model) == {"B1", "B2"} and len(SENSITIVITY.family.unique()) == 4
print("sweep covers", len(combos), "(model, family, magnitude) combinations x",
      len(TARGETS), "measurements")
print("mark_phase('sensitivity sweep') =", mark_phase("sensitivity sweep"), "s")