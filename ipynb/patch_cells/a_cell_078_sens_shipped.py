require("p06_train_b2", "p08_train_vhw", "p09_train_vh")

# --- fix B9: re-run the sweep on the SHIPPED models ------------------------------------------
# The earlier sweep ran on B2 only, so the model card's V-HW sensitivity filter was empty.
_SENS_MAX_PAIRS = SMOKE_PHOTOS if SMOKE else 2
_sens_shipped = []
for _vname, _model in (("V-HW", net_vhw), ("V-H", net_vh)):
    _t0 = time.time()
    _tbl = S.sweep_model(_model, IDX["train"], sens_ids, TARGETS, cfg, grid, device,
                         seed=cfg["seed"], model_name=_vname,
                         max_pairs_per_subject=_SENS_MAX_PAIRS)
    _sens_shipped.append(_tbl)
    print(f"{_vname} sweep done: {_tbl.shape} in {time.time()-_t0:.0f}s")

sens_vhw, sens_vh = _sens_shipped
SENSITIVITY = pd.concat([sens_b1, sens_b2, sens_vhw, sens_vh], ignore_index=True)
SENSITIVITY.to_csv(SENSITIVITY_CSV, index=False)
print("SENSITIVITY models:", sorted(SENSITIVITY.model.unique()))
print("written:", SENSITIVITY_CSV)

# Same identity-condition cross-check as the B2 sweep (fix B7 pattern), for both variants.
for _vname, _model, _ds in (("V-HW", net_vhw, val_ds_vhw), ("V-H", net_vh, val_ds_vh)):
    _dl = DataLoader(SilhouetteDataset(IDX["train"], sens_ids, TARGETS,
                                       img_height=cfg["img_height"], img_width=cfg["img_width"],
                                       threshold=cfg["mask_threshold"],
                                       use_weight=_model.use_weight,
                                       max_photos_per_subject=_SENS_MAX_PAIRS),
                     batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])
    _p, _t, _ = T.predict_split(_model, _dl, device)
    _ref = {c: float(v) for c, v in zip(TARGETS, np.abs(_p - _t).mean(axis=0) * 10.0)}
    _none = SENSITIVITY[(SENSITIVITY.model == _vname) & (SENSITIVITY.family == "none")
                        ].set_index("measurement")["mae_mm"].to_dict()
    _gap = max(abs(_none[c] - _ref[c]) for c in TARGETS)
    print(f"  {_vname} identity sweep MAE vs independent per-pair MAE: max |diff| = {_gap:.3e} mm")
    assert _gap < 1e-6, f"{_vname} sweep identity condition is inconsistent (fix B7)"

MAX_TOLERABLE_BOUNDARY = S.max_tolerable_boundary_error(
    SENSITIVITY, cfg["key_measurements"], cfg["tolerance_mm"], MM_PER_PX_NOMINAL,
    mode=cfg["max_tolerable"])
MAX_TOLERABLE_BOUNDARY.to_csv(ART / "max_tolerable_boundary_error.csv", index=False)

_vhw_budget = MAX_TOLERABLE_BOUNDARY[(MAX_TOLERABLE_BOUNDARY.model == "V-HW")
                                     & (MAX_TOLERABLE_BOUNDARY.criterion == cfg["max_tolerable"])]
print(f"\n=== the deliverable: maximum tolerable boundary error for V-HW "
      f"({cfg['max_tolerable']}-error criterion) ===")
print(M.format_metric_table(_vhw_budget.pivot_table(
    index=["measurement", "family", "side"], columns="tolerance_mm",
    values=["max_tolerable_px", "max_tolerable_mm"], dropna=False).reset_index()))
assert not _vhw_budget.empty, (
    "the V-HW sensitivity table is empty - the model card's step-3 budget would be blank "
    "(fix B9)")
assert set(_vhw_budget.model) == {"V-HW"}, "budget rows must be labelled V-HW"

# Is the augmented production model actually less boundary-sensitive than the plain baseline?
# This is the falsifiable prediction behind section 6's augmentation rationale.
_cmp = (MAX_TOLERABLE_BOUNDARY[(MAX_TOLERABLE_BOUNDARY.family == "morph")
                               & (MAX_TOLERABLE_BOUNDARY.criterion == cfg["max_tolerable"])
                               & (MAX_TOLERABLE_BOUNDARY.tolerance_mm == cfg["tolerance_mm"][-1])
                               & (MAX_TOLERABLE_BOUNDARY.side == "erosion")]
        .pivot_table(index="measurement", columns="model", values="max_tolerable_px",
                     dropna=False))
print(f"\nmax tolerable erosion at {cfg['tolerance_mm'][-1]} mm (px), B2 vs the augmented models:")
print(_cmp.to_string())
print("(V-HW >= B2 is the predicted effect of the Phase-4-derived augmentation.)")

print("mark_phase('sensitivity on shipped models') =", mark_phase("sensitivity on shipped models"), "s")