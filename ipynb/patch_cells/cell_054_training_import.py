import bodym_training as T

print("src/bodym_training.py written:", (PATHS["src"] / "bodym_training.py").exists())

# --- smoke_test subsampling (fix brief section 0) -------------------------------------------
# ONE helper, applied at every dataset construction site, so no cell can forget it.
SMOKE_SUBJECTS_MAX = int(cfg["smoke_subjects_max"]) if SMOKE else None
SMOKE_PHOTOS = int(cfg["smoke_photos_per_subject"]) if SMOKE else None
EPOCHS_B2 = 1 if SMOKE else int(cfg["epochs_b2"])
EPOCHS_PROD = 1 if SMOKE else int(cfg["epochs_prod"])
if SMOKE:
    _rng_smoke = np.random.default_rng(cfg["seed"])

    def _subsample(ids: Sequence[str]) -> List[str]:
        """Cap a split's subject list for the smoke run.

        Parameters
        ----------
        ids : sequence of str
            Full subject list.

        Returns
        -------
        list of str
            Deterministically subsampled, at most ``smoke_subjects_max`` subjects.
        """
        ids = list(ids)
        if len(ids) <= SMOKE_SUBJECTS_MAX:
            return ids
        return sorted(_rng_smoke.choice(ids, size=SMOKE_SUBJECTS_MAX, replace=False).tolist())

    ids_fit = _subsample(ids_fit)
    ids_val = _subsample(ids_val)
    ids_calib = _subsample(ids_calib)
    print(f"SMOKE: fit {len(ids_fit)} | val {len(ids_val)} | calib {len(ids_calib)} subjects, "
          f"<= {SMOKE_PHOTOS} photos each, epochs_b2={EPOCHS_B2}, epochs_prod={EPOCHS_PROD}")

ds_probe = SilhouetteDataset(IDX["train"], ids_val[:4], TARGETS,
                             img_height=cfg["img_height"], img_width=cfg["img_width"],
                             threshold=cfg["mask_threshold"], use_weight=True,
                             max_photos_per_subject=SMOKE_PHOTOS)
x0, y0, sid0, pid0 = ds_probe[0]
print(f"sample tensor {tuple(x0.shape)} dtype={x0.dtype} "
      f"| mask channel unique values={sorted(np.unique(x0[0].numpy()).tolist())}")
print("channel 1 (height) =", float(x0[1].unique()[0]),
      "expected height_cm/200 =", float(ds_probe.height[sid0]) / HEIGHT_SCALE)
print("channel 2 (weight) =", float(x0[2].unique()[0]),
      "expected weight_kg/100 =", float(ds_probe.weight[sid0]) / WEIGHT_SCALE)
print("channel count:", x0.shape[0], "expected:", N_CHANNELS)
assert x0.shape == (N_CHANNELS, cfg["img_height"], 2 * cfg["img_width"])
assert set(np.unique(x0[0].numpy())) <= {0.0, 1.0}, "mask channel must be binarised to {0,1}"
assert abs(float(x0[1].unique()[0]) - float(ds_probe.height[sid0]) / HEIGHT_SCALE) < 1e-6
assert abs(float(x0[2].unique()[0]) - float(ds_probe.weight[sid0]) / WEIGHT_SCALE) < 1e-6
assert y0.shape == (len(TARGETS),)
print("scalar channels are constant across the image:",
      {c: float(np.ptp(x0[c].numpy())) for c in (1, 2)})

xw = to_tensor(np.array(Image.open(ds_probe.photos.iloc[0].front_mask)),
               np.array(Image.open(ds_probe.photos.iloc[0].side_mask)),
               float(ds_probe.height[sid0]), float(ds_probe.weight[sid0]),
               cfg["img_height"], cfg["img_width"], cfg["mask_threshold"], use_weight=False)
print("V-H tensor channels (weight removed, not zero-filled):", xw.shape[0])
assert xw.shape[0] == N_CHANNELS_VH, "V-H must drop the channel entirely"
assert not (xw.shape[0] > 2 and np.allclose(xw[2], 0)), \
    "V-H must NOT zero-fill a weight channel"

# The same check from the network's side: MeasurementNet must reject the wrong channel count,
# and must report the count it expects.
_net_probe = MeasurementNet(backbone=cfg["backbone"], pretrained=False, n_outputs=len(TARGETS),
                            hidden=cfg["hidden"], dropout=cfg["dropout"], use_weight=False)
assert _net_probe.n_channels == N_CHANNELS_VH
del _net_probe
print("preprocessing self-check passed")
print("mark_phase('training module') =", mark_phase("training module"), "s")