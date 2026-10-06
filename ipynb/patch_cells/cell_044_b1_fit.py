# --- B1 ----------------------------------------------------------------------------------
# Both fixes below change the numbers materially, so both are checked in code rather than
# asserted in prose:
#   B3  slice rows were vertically mirrored (f=0 mapped to the top image row)
#   A1  the calibration dropped every Ridge intercept
B1_MODEL = B1Geometric(threshold=cfg["mask_threshold"], ridge_alpha=1.0)


def first_photo_per_subject(index: BodyMIndex, max_photos: Optional[int] = None) -> pd.DataFrame:
    """Select one silhouette per subject, deterministically.

    B0 and B1 predict *subject-level* ground truth. Fixing one photo per subject keeps the
    train-time and inference-time semantics identical and avoids averaging features across a
    variable number of photos, which would make the feature count depend on the subject.

    Parameters
    ----------
    index : BodyMIndex
        Assembled index for one split.
    max_photos : int, optional
        Cap on photos per subject, used by ``smoke_test``.

    Returns
    -------
    DataFrame
        ``index.photos`` restricted to the lexicographically first ``photo_id`` per subject.
    """
    photos = index.photos.sort_values("photo_id")
    if max_photos:
        photos = photos.groupby("subject_id", head=0, as_index=False).head(max_photos)
    return photos.groupby("subject_id", as_index=False).first()


def b1_matrix(index: BodyMIndex, subject_ids: Sequence[str], desc: str
              ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract the B1 feature and target matrices for given subjects.

    Parameters
    ----------
    index : BodyMIndex
        Assembled index.
    subject_ids : sequence of str
        Subjects, in the order the caller wants them.
    desc : str
        Progress-bar label.

    Returns
    -------
    (F, Y, ids) : tuple
        ``F`` of shape (n, d) in mm/cm units, ``Y`` of shape (n, m) in cm, and the ordered ids.
    """
    photos = first_photo_per_subject(index, max_photos=SMOKE_PHOTOS if SMOKE else None
                                     ).set_index("subject_id")
    hgt = index.subjects.set_index("subject_id")["height_cm"]
    tgt = index.subjects.set_index("subject_id")[list(TARGETS)]

    feats, targs, ids = [], [], []
    for sid in tqdm(list(subject_ids), desc=desc):
        row = photos.loc[sid]
        front = np.array(Image.open(row.front_mask))
        side = np.array(Image.open(row.side_mask))
        feats.append(B1_MODEL.features(front, side, float(hgt.loc[sid])))
        targs.append(tgt.loc[sid].to_numpy(float))
        ids.append(sid)
    return np.asarray(feats), np.asarray(targs), ids


# --- fix B3: prove the slice rows are in the right vertical order before trusting any girth ----
_probe_row0 = IDX["train"].photos.iloc[0]
_probe_front = np.array(Image.open(_probe_row0.front_mask))
_probe_side = np.array(Image.open(_probe_row0.side_mask))
_probe_h = float(IDX["train"].subjects.set_index("subject_id")
                 .loc[_probe_row0.subject_id, "height_cm"])
_rows_probe, _runs_probe = B1_MODEL.measure_with_runs(_probe_front, _probe_side, _probe_h)
_row_of = {n.split("_")[0]: int(v) for n, v in _rows_probe.items() if n.endswith("_row")}
print("B1 slice rows (image row index; larger = lower in the image):")
for n in ("ankle", "calf", "crotch", "hip", "waist", "chest", "shoulder"):
    if n in _row_of:
        print(f"    {n:10s} f={B1_MODEL.slice_heights()[n]:.3f}  row={_row_of[n]}")
assert _row_of["ankle"] > _row_of["chest"], (
    f"slice rows are inverted: ankle row {_row_of['ankle']} must be BELOW chest row "
    f"{_row_of['chest']} in image coordinates (fix B3)")
print(f"  ankle row {_row_of['ankle']} > chest row {_row_of['chest']}: slice order is correct")

F_fit, YF_fit, ids_fit_b1 = b1_matrix(IDX["train"], ids_fit, "B1 fit")
F_val, YF_val, ids_val_b1 = b1_matrix(IDX["train"], ids_val, "B1 val")
assert ids_fit_b1 == ids_fit and ids_val_b1 == ids_val, "B1 subject order must match B0 order"

derived, unsupported = set(B1_MODEL.derived(TARGETS)), set(B1_MODEL.unsupported())
assert derived | unsupported == set(TARGETS), "B1 derived + unsupported must partition targets"
assert not (derived & unsupported), "B1 derived and unsupported overlap"
print(f"B1 derives {len(derived)} targets, unsupported = {sorted(unsupported)}")

b1 = B1Geometric(threshold=cfg["mask_threshold"], ridge_alpha=1.0)
b1.fit(F_fit, YF_fit, TARGETS)
P1_val = b1.predict(F_val)
assert P1_val.shape == Y_val.shape and np.isfinite(P1_val).all(), "B1 prediction invalid"

# fix A1: the fitted Ridge objects are stored, so each prediction now includes its intercept.
# Print them: a per-measurement intercept that is large relative to the coefficient-weighted
# feature range means the calibration is mostly a constant offset, which is worth seeing.
_fn = B1_MODEL.feature_names()
print("\nB1 calibration intercepts (cm):")
for c in TARGETS:
    if c in b1.unsupported():
        continue
    print(f"    {c:20s} intercept {b1.intercept(c):+8.3f}")
assert all(np.isfinite(b1.intercept(c)) for c in TARGETS if c not in b1.unsupported()), \
    "B1 intercepts must be finite (fix A1)"

# A geometric model that cannot beat a constant train-mean predictor has a calibration bug, not
# a weak heuristic. This is the check that would have caught fitting a ridge on unstandardised
# millimetre-scale features, and it is also the check A1 was failing.
_baseline = np.abs(YF_fit.mean(axis=0) - Y_val).mean(axis=0)
_worse = [c for j, c in enumerate(TARGETS)
          if c != "height" and np.abs(P1_val[:, j] - Y_val[:, j]).mean() > _baseline[j] + 1e-6]
print("\nB1 measurements worse than a constant train-mean predictor:", _worse or "none")
print("  train-mean baseline MAE (mm):",
      {c: round(float(_baseline[j] * 10), 1) for j, c in enumerate(TARGETS)})
print("  B1 MAE (mm):",
      {c: round(float(np.abs(P1_val[:, j] - Y_val[:, j]).mean() * 10), 1)
       for j, c in enumerate(TARGETS)})
assert not _worse, f"B1 calibration is broken for: {_worse}"

metrics_B1 = M.per_measurement_metrics(Y_val, P1_val, TARGETS)
print("\n=== B1 geometric per-slice ellipse — validation ===")
print(M.format_metric_table(metrics_B1))

# Coefficient probe: which geometry does each key prediction actually lean on?
# Read from the FITTED model (b1), not the feature extractor (B1_MODEL). Because each
# measurement now uses only its own features (fix B4), the probe shows a genuine dependency
# rather than the largest of 27 correlated coefficients.
print("\nB1 per-measurement calibration (feature: coefficient; standardised units)")
for col in cfg["key_measurements"]:
    c = b1.coefficients(col)
    own = B1_MODEL.features_for(col)
    top = sorted(own, key=lambda i: -abs(c[i]))[:4]
    print(f"  {col:10s} uses {[_fn[i] for i in own]}")
    print(f"  {'':10s} top: " + ", ".join(f"{_fn[i]}={c[i]:+.4g}" for i in top)
          + f"  intercept={b1.intercept(col):+.3f}")
_scale_sd = b1.scaler_scale_[_fn.index("waist_perim_mm")]
print(f"  (waist_perim_mm feature sd in the fit split: {_scale_sd:.1f} mm)")

# --- fix B4 visual check: draw every slice row and the runs actually measured ---------------
_slice_overlay_rows = [(n, _row_of[n]) for n in B1_MODEL.slice_heights() if n in _row_of]
_overlay_runs, _overlay_extent = slice_overlay(
    _probe_front, _slice_overlay_rows, ART / "fig_b1_slices.png",
    threshold=cfg["mask_threshold"],
    title="B1 slice rows and the foreground runs measured\n"
          "(fix B3: rows in the correct vertical order; fix B4: midline run, not full row)")
_multi = {n: len(v) for n, v in _overlay_runs.items() if len(v) > 1}
print(f"fig_b1_slices.png written; slices whose row has >1 foreground run: "
      f"{len(_multi)}/{len(_overlay_runs)}")
print("  e.g.", dict(list(_multi.items())[:6]))
assert len(_multi) > 0, (
    "no slice row has more than one foreground run - if the front mask is a single blob "
    "everywhere, the midline-run correction (fix B4) cannot be demonstrated and the figure "
    "needs inspecting")
for n in ("thigh", "calf"):
    assert len(_overlay_runs[n]) >= 1, f"{n} slice row has no foreground run"

np.save(PATHS["artifacts"] / "P1_val.npy", P1_val)
np.save(PATHS["artifacts"] / "P0_val_gbm.npy", P0_val_gbm)
np.save(PATHS["artifacts"] / "Y_val.npy", Y_val)
(PATHS["artifacts"] / "val_subjects.json").write_text(json.dumps(ids_val))
print("mark_phase('B0+B1') =", mark_phase("B0+B1"), "s")