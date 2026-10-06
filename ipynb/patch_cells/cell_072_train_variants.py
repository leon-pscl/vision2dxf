print("\n=== V-HW: height + weight channels ===")
train_ds_vhw = make_production_dataset(ids_fit, use_weight=True, augment=AUGMENTER)
val_ds_vhw = make_production_dataset(ids_val, use_weight=True)
net_vhw = MeasurementNet(backbone=cfg["backbone"], pretrained=cfg["pretrained"],
                         n_outputs=len(TARGETS), hidden=cfg["hidden"], dropout=cfg["dropout"],
                         use_weight=True, offline_weights=cfg["offline_weights"]).to(device)
net_vhw, hist_vhw = T.train_model(net_vhw, train_ds_vhw, val_ds_vhw, TARGETS, cfg, device,
                                 ckpt_dir=PATHS["working"] / "checkpoints", tag="vhw",
                                 epochs=EPOCHS_PROD)
hist_vhw.to_csv(ART / "history_vhw.csv", index=False)

print("\n=== V-H: height channel only (weight channel removed) ===")
train_ds_vh = make_production_dataset(ids_fit, use_weight=False, augment=AUGMENTER)
val_ds_vh = make_production_dataset(ids_val, use_weight=False)
net_vh = MeasurementNet(backbone=cfg["backbone"], pretrained=cfg["pretrained"],
                        n_outputs=len(TARGETS), hidden=cfg["hidden"], dropout=cfg["dropout"],
                        use_weight=False, offline_weights=cfg["offline_weights"]).to(device)
net_vh, hist_vh = T.train_model(net_vh, train_ds_vh, val_ds_vh, TARGETS, cfg, device,
                                ckpt_dir=PATHS["working"] / "checkpoints", tag="vh",
                                epochs=EPOCHS_PROD)
hist_vh.to_csv(ART / "history_vh.csv", index=False)

# Channel-count proof that V-H removed rather than zero-filled the weight channel.
_x_hw = train_ds_vhw[0][0]
_x_h = train_ds_vh[0][0]
print(f"\nchannel counts -> V-HW {_x_hw.shape[0]}, V-H {_x_h.shape[0]}")
assert _x_hw.shape[0] == N_CHANNELS, f"V-HW must have {N_CHANNELS} channels, got {_x_hw.shape[0]}"
assert _x_h.shape[0] == N_CHANNELS_VH, f"V-H must have {N_CHANNELS_VH} channels, got {_x_h.shape[0]}"
assert len(hist_vhw) == EPOCHS_PROD and len(hist_vh) == EPOCHS_PROD

# fix C1: record the optimiser-step budget beside the results that depend on it.
def _steps_for(dataset, batch_size: int, epochs: int) -> int:
    """Total optimiser steps for one training run.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Training dataset.
    batch_size : int
        Configured batch size.
    epochs : int
        Configured epochs.

    Returns
    -------
    int
        ``epochs * floor(n_samples / batch_size)`` (``drop_last=True``).
    """
    per_epoch = len(dataset) // max(batch_size, 1)
    return int(epochs * max(per_epoch, 0))


TRAIN_STEPS = {
    "B2": _steps_for(train_ds_b2, cfg["batch_size"], EPOCHS_B2),
    "V-HW": _steps_for(train_ds_vhw, cfg["batch_size"], EPOCHS_PROD),
    "V-H": _steps_for(train_ds_vh, cfg["batch_size"], EPOCHS_PROD),
}
print("optimiser steps:", json.dumps(TRAIN_STEPS))
for _k, _v in TRAIN_STEPS.items():
    print(f"    {_k:6s} {_v:7d} steps")
print("reference: Ruiz et al. (2022) train for 150k iterations at batch 22.")
print(f"    this run is {min(TRAIN_STEPS.values())/150000:.3%} of that budget at minimum. "
      "Every B0-beats-vision comparison below is PROVISIONAL at this budget (fix C1).")
print("mark_phase('production training') =", mark_phase("production training"), "s")