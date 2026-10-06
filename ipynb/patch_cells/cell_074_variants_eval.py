@torch.no_grad()
def predict_index(model, index: BodyMIndex, subject_ids: Sequence[str], targets: Sequence[str],
                  cfg: Dict, device: torch.device, batch_size: int = 8,
                  max_photos_per_subject: Optional[int] = None):
    """Predict on any split, **one row per photo pair** (fix B5).

    The previous version averaged all of a subject's photos and returned one row per subject.
    Deployment does not do that: ``infer.py`` receives exactly one front mask and one side
    mask. Subject averaging therefore (a) understated the error, by the variance reduction from
    pooling several noisy views of the same body, and (b) understated it *unevenly*, because
    Test-A has far more photos per subject than Test-B. The Test-A vs Test-B gap was in part
    just a difference in averaging count.

    Masks are decoded **inside the batch loop**. Pre-stacking every silhouette would need
    ``n_photos x C x H x 2W x 4`` bytes, which at the corrected input size is several GB for
    Test-A alone, so this streams instead.

    Parameters
    ----------
    model : nn.Module
        Trained network.
    index : BodyMIndex
        Assembled index for the split.
    subject_ids : sequence of str
        Subjects to predict.
    targets : sequence of str
        Target names.
    cfg : dict
        Notebook configuration (image size, threshold).
    device : torch.device
        Compute device.
    batch_size : int, default 8
        Number of silhouettes decoded per forward pass.
    max_photos_per_subject : int, optional
        Cap on photo pairs per subject, for cost control.

    Returns
    -------
    (pred, true, subjects, photo_ids) : tuple of ndarray
        Predictions and ground truth in cm of shape ``(n_photo_pairs, n_targets)``, and the
        per-row subject and photo ids. Row *i* is one front/side pair.
    """
    from bodym_cnn import to_tensor

    keep = set(subject_ids)
    photos = index.photos[index.photos.subject_id.isin(keep)]
    if max_photos_per_subject:
        photos = (photos.sort_values("photo_id")
                  .groupby("subject_id", head=0, as_index=False)
                  .head(max_photos_per_subject))
    photos = photos.reset_index(drop=True)
    meta = index.subjects.set_index("subject_id")
    height, weight = meta["height_cm"].to_dict(), meta["weight_kg"].to_dict()
    Y = {s: meta.loc[s, list(targets)].to_numpy(float) for s in meta.index}
    use_w = getattr(model, "use_weight", True)

    model.eval()
    preds, trues, subs, pids = [], [], [], []
    for i in tqdm(range(0, len(photos), batch_size), desc="predict", leave=False):
        rows = photos.iloc[i:i + batch_size]
        xs = [to_tensor(np.array(Image.open(r.front_mask)),
                        np.array(Image.open(r.side_mask)),
                        height[r.subject_id], weight[r.subject_id],
                        cfg["img_height"], cfg["img_width"], cfg["mask_threshold"], use_w)
              for r in rows.itertuples()]
        xb = torch.from_numpy(np.stack(xs)).to(device)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            p = model(xb).float().cpu().numpy()
        preds.append(p)
        trues.append(np.stack([Y[r.subject_id] for r in rows.itertuples()]))
        subs.extend([r.subject_id for r in rows.itertuples()])
        pids.extend([r.photo_id for r in rows.itertuples()])
    if not preds:
        return (np.zeros((0, len(targets))), np.zeros((0, len(targets))),
                np.asarray([], dtype=object), np.asarray([], dtype=object))
    return (np.concatenate(preds, axis=0), np.concatenate(trues, axis=0),
            np.asarray(subs, dtype=object), np.asarray(pids, dtype=object))


def photos_per_subject_note(index: BodyMIndex, subject_ids: Sequence[str]) -> str:
    """One-line summary of how many photo pairs each metric row represents (fix B5 CHECK).

    Printed next to every metric table so a reader can see whether the table is per-pair or
    per-subject, and how unevenly the pairs are distributed.

    Parameters
    ----------
    index : BodyMIndex
        Assembled index.
    subject_ids : sequence of str
        Subjects in the table.

    Returns
    -------
    str
        ``"<n> pairs from <k> subjects; photos/subject min/median/max a/b/c"``.
    """
    p = index.photos[index.photos.subject_id.isin(set(subject_ids))]
    counts = p.groupby("subject_id").size()
    return (f"{len(p)} photo pairs from {counts.size} subjects; photos/subject "
            f"min/median/max {counts.min()}/{int(counts.median())}/{counts.max()}")


_SENS_PAIRS = SMOKE_PHOTOS if SMOKE else 6      # cap for the per-subject evaluation tables

P_VHW_val, Y_VHW_val, subs_vhw, pids_vhw = predict_index(
    net_vhw, IDX["train"], ids_val, TARGETS, cfg, device, batch_size=cfg["batch_size"],
    max_photos_per_subject=_SENS_PAIRS)
P_VH_val, Y_VH_val, subs_vh, pids_vh = predict_index(
    net_vh, IDX["train"], ids_val, TARGETS, cfg, device, batch_size=cfg["batch_size"],
    max_photos_per_subject=_SENS_PAIRS)
assert list(subs_vhw) == list(subs_vh), "validation prediction order mismatch"
assert np.allclose(Y_VHW_val, Y_VH_val, atol=1e-6)
assert Y_VHW_val.shape[1] == len(TARGETS)

print("unit of evaluation: PHOTO PAIR (fix B5)")
print("  V-HW:", photos_per_subject_note(IDX["train"], ids_val), f"-> {len(P_VHW_val)} rows")
metrics_VHW = M.per_measurement_metrics(Y_VHW_val, P_VHW_val, TARGETS)
metrics_VH = M.per_measurement_metrics(Y_VH_val, P_VH_val, TARGETS)
np.save(ART / "P_VHW_val.npy", P_VHW_val)
np.save(ART / "P_VH_val.npy", P_VH_val)

# --- fix B1: prove the trained model actually USES the scalars ------------------------------
# The old zero-init design meant the scalars never received gradient, so V-HW and V-H were
# numerically the same model. A +10% height must now move the girth predictions.
print("\nscalar sensitivity of the trained models (+10% height_cm):")
SCALAR_DELTA = {}
for vname, model, ds in (("V-HW", net_vhw, val_ds_vhw), ("V-H", net_vh, val_ds_vh)):
    _rows = ds.photos.iloc[:min(12, len(ds.photos))]
    _meta = IDX["train"].subjects.set_index("subject_id")
    _x1, _x2 = [], []
    for r in _rows.itertuples():
        f = np.array(Image.open(r.front_mask))
        s = np.array(Image.open(r.side_mask))
        h = float(_meta.loc[r.subject_id, "height_cm"])
        w = float(_meta.loc[r.subject_id, "weight_kg"])
        a = to_tensor(f, s, h, w if model.use_weight else None, cfg["img_height"],
                      cfg["img_width"], cfg["mask_threshold"], model.use_weight)
        b = to_tensor(f, s, h * 1.10, w if model.use_weight else None, cfg["img_height"],
                      cfg["img_width"], cfg["mask_threshold"], model.use_weight)
        _x1.append(a); _x2.append(b)
    with torch.no_grad():
        p1 = model(torch.from_numpy(np.stack(_x1)).to(device)).cpu().numpy()
        p2 = model(torch.from_numpy(np.stack(_x2)).to(device)).cpu().numpy()
    d = np.abs(p2 - p1).mean(axis=0) * 10.0            # mean |Δ| in mm, per measurement
    SCALAR_DELTA[vname] = dict(zip(TARGETS, d.tolist()))
    _girths = [c for c in cfg["key_measurements"]]
    print(f"  {vname}: " + ", ".join(f"{c} {d[TARGETS.index(c)]:.2f} mm" for c in _girths))
    for c in _girths:
        print(f"      |Δ height +10%| for {c:8s} = {d[TARGETS.index(c)]:6.3f} mm")
    assert all(d[TARGETS.index(c)] > 0 for c in _girths), (
        f"{vname}: a 10% height change does not move ANY girth prediction. The scalar path is "
        "dead - the head's scalar-input weights must be learning something (fix B1).")

# ---- flag table: required by brief section 7 ---------------------------------------------
mae_tbl = pd.DataFrame({
    "measurement": TARGETS,
    "mae_B0_mm": metrics_B0_gbm.set_index("measurement").loc[TARGETS, "mae_mm"].values,
    "mae_B1_mm": metrics_B1.set_index("measurement").loc[TARGETS, "mae_mm"].values,
    "mae_B2_mm": metrics_B2.set_index("measurement").loc[TARGETS, "mae_mm"].values,
    "mae_VHW_mm": metrics_VHW.set_index("measurement").loc[TARGETS, "mae_mm"].values,
    "mae_VH_mm": metrics_VH.set_index("measurement").loc[TARGETS, "mae_mm"].values,
})
GIRTH = {"chest", "waist", "hip", "bicep", "thigh", "calf", "ankle", "wrist", "forearm"}
mae_tbl["kind"] = ["girth" if m in GIRTH else "length/input" for m in mae_tbl.measurement]
for v in ("VHW", "VH"):
    mae_tbl[f"{v}_beats_B0"] = mae_tbl[f"mae_{v}_mm"] < mae_tbl["mae_B0_mm"]
mae_tbl["weight_gain_mm"] = mae_tbl["mae_VH_mm"] - mae_tbl["mae_VHW_mm"]

FLAGS = mae_tbl[~mae_tbl.VHW_beats_B0 | ~mae_tbl.VH_beats_B0][
    ["measurement", "kind", "mae_B0_mm", "mae_VHW_mm", "mae_VH_mm",
     "VHW_beats_B0", "VH_beats_B0"]].copy()
print("\n=== FLAG TABLE: measurements where a variant does not beat B0 on validation ===")
print(M.format_metric_table(FLAGS) if len(FLAGS) else "  (none - every variant beats B0 everywhere)")

# fix B12: the old assertion was `... or True`, which asserts nothing. Every flagged row must
# genuinely fail against B0 for at least one variant.
_expected_flag_cols = {"measurement", "kind", "mae_B0_mm", "mae_VHW_mm", "mae_VH_mm",
                       "VHW_beats_B0", "VH_beats_B0"}
assert _expected_flag_cols.issubset(set(FLAGS.columns)), (
    f"FLAGS is missing columns: {sorted(_expected_flag_cols - set(FLAGS.columns))}")
for _r in FLAGS.itertuples():
    _fails = (not _r.VHW_beats_B0) or (not _r.VH_beats_B0)
    _really_fails = (_r.mae_VHW_mm >= _r.mae_B0_mm) or (_r.mae_VH_mm >= _r.mae_B0_mm)
    assert _fails and _really_fails, (
        f"FLAGS row {_r.measurement} is flagged but both variants actually beat B0 "
        f"(B0 {_r.mae_B0_mm:.3f}, V-HW {_r.mae_VHW_mm:.3f}, V-H {_r.mae_VH_mm:.3f})")
_unflagged = mae_tbl[mae_tbl.VHW_beats_B0 & mae_tbl.VH_beats_B0]
for _r in _unflagged.itertuples():
    assert _r.mae_VHW_mm < _r.mae_B0_mm and _r.mae_VH_mm < _r.mae_B0_mm, (
        f"{_r.measurement} is not flagged but does not beat B0")
print(f"flag table verified: {len(FLAGS)} flagged, {len(_unflagged)} clear, "
      f"both directions checked")
print("PROVISIONAL (fix C1): these comparisons are made at "
      f"{TRAIN_STEPS['V-HW']} optimiser steps for V-HW / {TRAIN_STEPS['V-H']} for V-H. "
      "A flag here means 'did not beat B0 at this budget', not 'cannot beat B0'.")

mae_tbl.to_csv(ART / "validation_mae_comparison.csv", index=False)
FLAGS.to_csv(ART / "flagged_measurements.csv", index=False)

print("\n=== Validation MAE (mm) across all models — per PHOTO PAIR ===")
print(M.format_metric_table(mae_tbl.drop(columns=["VHW_beats_B0", "VH_beats_B0"])))
print("\nweight-channel gain (V-H minus V-HW MAE, mm; positive = weight helps):")
print(mae_tbl.set_index("measurement")["weight_gain_mm"].round(2).to_string())

# --- secondary, clearly-labelled subject-averaged table ------------------------------------
_a_p, _a_t, _a_s = T.aggregate_by_subject(P_VHW_val, Y_VHW_val, subs_vhw)
_b_p, _b_t, _b_s = T.aggregate_by_subject(P_VH_val, Y_VH_val, subs_vh)
SUBJECT_AVG_VHW = M.per_measurement_metrics(_a_t, _a_p, TARGETS)
SUBJECT_AVG_VH = M.per_measurement_metrics(_b_t, _b_p, TARGETS)
SUBJECT_AVG_VHW.to_csv(ART / "subject_averaged_metrics.csv", index=False)
_cmp = mae_tbl.set_index("measurement")[["mae_VHW_mm"]].join(
    SUBJECT_AVG_VHW.set_index("measurement")[["mae_mm"]].rename(
        columns={"mae_mm": "mae_VHW_subject_avg_mm"}))
_cmp["saving_from_averaging_mm"] = _cmp["mae_VHW_subject_avg_mm"] - _cmp["mae_VHW_mm"]
print("\n=== SECONDARY: subject-averaged MAE (not the deployment unit) ===")
print(_cmp.round(2).to_string())
print("  The difference is the optimism that per-pair evaluation removes. Quoting the "
      "averaged figure would understate the error a single capture actually sees.")

fig, ax = plt.subplots(figsize=(11, 4.6))
mae_tbl.set_index("measurement")[["mae_B0_mm", "mae_B1_mm", "mae_VHW_mm", "mae_VH_mm"]].plot(
    kind="barh", ax=ax, width=0.8)
ax.axvline(25, ls="--", c="k", lw=0.8)
ax.text(25.4, -0.4, "25 mm", fontsize=7)
ax.set_xlabel("Validation MAE (mm), per photo pair")
ax.set_title(f"B0 vs B1 vs V-HW vs V-H (PROVISIONAL at {TRAIN_STEPS['V-HW']} steps)")
ax.legend(fontsize=8); ax.grid(axis="x", alpha=0.25)
fig.tight_layout(); fig.savefig(ART / "fig_variants_validation.png", bbox_inches="tight")
plt.show(); plt.close(fig)
print("mark_phase('variant comparison') =", mark_phase("variant comparison"), "s")