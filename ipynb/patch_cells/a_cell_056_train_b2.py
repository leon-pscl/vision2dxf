# --- phase p06_train_b2: train the B2 baseline, resumably ------------------------------------
_res, _why = begin("p06_train_b2", deps=["p05_b1"])
if _res is R.BLOCKED:
    net_b2 = hist_b2 = metrics_B2 = P2_val = None
    TRAIN_META_B2: Dict[str, Any] = {}
    print(f"p06_train_b2 SKIPPED (blocked): {_why}")
else:
    train_ds_b2 = SilhouetteDataset(IDX["train"], ids_fit, TARGETS,
                                    img_height=cfg["img_height"], img_width=cfg["img_width"],
                                    threshold=cfg["mask_threshold"], use_weight=True,
                                    max_photos_per_subject=SMOKE_PHOTOS)
    val_ds_b2 = SilhouetteDataset(IDX["train"], ids_val, TARGETS,
                                  img_height=cfg["img_height"], img_width=cfg["img_width"],
                                  threshold=cfg["mask_threshold"], use_weight=True,
                                  max_photos_per_subject=SMOKE_PHOTOS)
    print(f"B2 train samples (photos): {len(train_ds_b2)} | val samples: {len(val_ds_b2)}")

    if _res is not None and (_res.get("net_b2") is not None):
        # rebuild the network from the saved state dict rather than re-training
        net_b2 = MeasurementNet(backbone=cfg["backbone"], pretrained=False,
                                n_outputs=len(TARGETS), hidden=cfg["hidden"],
                                dropout=cfg["dropout"], use_weight=True).to(DEVICE)
        net_b2.load_state_dict(torch.load(_res["net_b2"], map_location=DEVICE,
                                          weights_only=False)["model"])
        hist_b2 = pd.read_parquet(_res["hist_b2"])
        metrics_B2 = pd.read_parquet(_res["metrics_B2"])
        P2_val = np.load(_res["P2_val"])
        TRAIN_META_B2 = _res.get("_meta", {})
        print(f"  B2 reloaded from cache: best val MAE "
              f"{TRAIN_META_B2.get('best_val_mae_mm')} mm over "
              f"{TRAIN_META_B2.get('epochs_completed')} epochs")
    else:
        with guard("p06_train_b2") as _dev:
            net_b2 = net_b2.to(_dev)

            def _on_interrupt(status, meta):
                """Record an interrupted B2 run as partial so it is never reused as done.

                Parameters
                ----------
                status : str
                    'partial' or 'done'.
                meta : dict
                    Loop metadata.
                """
                done("p06_train_b2", {}, status=status, meta={**meta, "stage": "interrupted"})

            net_b2, hist_b2, TRAIN_META_B2 = T.train_model(
                net_b2, train_ds_b2, val_ds_b2, TARGETS, cfg, _dev,
                ckpt_dir=RUNSTATE.ckpt_dir("b2"), tag="b2", epochs=EPOCHS_B2,
                on_interrupt=_on_interrupt)
        val_dl_b2 = DataLoader(val_ds_b2, batch_size=cfg["batch_size"], shuffle=False,
                               num_workers=cfg["num_workers"], pin_memory=True)
        P2_val, Y2_val, subs2 = T.predict_split(net_b2, val_dl_b2, DEVICE)
        assert list(subs2) and P2_val.shape[1] == len(TARGETS)
        assert np.isfinite(P2_val).all()
        assert len(hist_b2) >= 1, "no epoch completed"
        metrics_B2 = M.per_measurement_metrics(Y2_val, P2_val, TARGETS)
        # best.pt is the weights of record; a copy next to the phase payload lets a cached phase
        # be reloaded without reconstructing the model from the variant registry.
        _b2_state = R.atomic_save({k: v.cpu() for k, v in net_b2.state_dict().items()},
                                 RUNSTATE.results_dir("p06_train_b2") / "b2_state.pt", "torch")
        done("p06_train_b2", {"hist_b2": hist_b2, "metrics_B2": metrics_B2},
            status=TRAIN_META_B2.get("status", "done"),
            extra_artefacts=[RUNSTATE.ckpt_dir("b2") / "best.pt", _b2_state],
            meta=TRAIN_META_B2)
    np.save(ART / "P2_val.npy", P2_val)

if _res is not R.BLOCKED and metrics_B2 is not None:
    print(f"\n=== B2 validation ({TRAIN_META_B2.get('epochs_completed')}/"
          f"{TRAIN_META_B2.get('epochs_configured')} epochs, "
          f"stop={TRAIN_META_B2.get('stop_reason')}, best val MAE "
          f"{TRAIN_META_B2.get('best_val_mae_mm')} mm) ===")
    if isinstance(hist_b2, pd.DataFrame) and len(hist_b2):
        fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
        ax[0].plot(hist_b2["epoch"], hist_b2["train_loss_cm"], "o-", label="train L1 (cm)")
        ax[0].plot(hist_b2["epoch"], hist_b2["val_mae_mm"], "s-", label="val MAE (mm) / 10")
        ax[0].set_xlabel("epoch"); ax[0].set_title("B2 training"); ax[0].legend()
        comp = pd.DataFrame({
            "B0_gbm": metrics_B0_gbm.set_index("measurement")["mae_mm"],
            "B1": metrics_B1.set_index("measurement")["mae_mm"],
            "B2": metrics_B2.set_index("measurement")["mae_mm"]})
        comp.plot(kind="barh", ax=ax[1], width=0.8)
        ax[1].set_xlabel("MAE (mm)")
        ax[1].set_title(f"Validation MAE by model (B2 at {TRAIN_META_B2.get('total_steps_run')} steps)")
        ax[1].axvline(25, ls="--", c="k", lw=0.8)
        ax[1].grid(axis="x", alpha=0.25)
        fig.tight_layout(); fig.savefig(PATHS["figs"] / "fig_b2_training.png", bbox_inches="tight")
        plt.show(); plt.close(fig)
        print(comp.round(2).to_string())
        b2_beats_b0 = comp["B2"] < comp["B0_gbm"]
        print(f"\nB2 beats B0 on: {int(b2_beats_b0.sum())}/{len(comp)} measurements")
        print("B2 does NOT beat B0 on:", sorted(comp.index[~b2_beats_b0]))
print("mark_phase('B2 train') =", mark_phase("B2 train"), "s")