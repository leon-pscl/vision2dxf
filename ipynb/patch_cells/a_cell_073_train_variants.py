# --- phases p08_train_vhw and p09_train_vh: the two production variants ----------------------
# Each variant is its own phase, so one can finish while the other is still partial (addendum A4).
TRAIN_META: Dict[str, Dict[str, Any]] = {}
PRED_VAL: Dict[str, Any] = {}

for _phase, _tag, _use_w in (("p08_train_vhw", "vhw", True), ("p09_train_vh", "vh", False)):
    _res, _why = begin(_phase, deps=["p05_b1", "p07_sens_b2"])
    if _res is R.BLOCKED:
        print(f"{_phase} SKIPPED (blocked): {_why}")
        if _tag == "vhw":
            net_vhw = hist_vhw = None
        else:
            net_vh = hist_vh = None
        TRAIN_META[_tag] = {}
        continue

    _ds_tr = make_production_dataset(ids_fit, use_weight=_use_w, augment=AUGMENTER)
    _ds_va = make_production_dataset(ids_val, use_weight=_use_w)
    print(f"\n=== {_tag.upper()}: {'height + weight' if _use_w else 'height only'} "
          f"({len(_ds_tr)} train photos, {len(_ds_va)} val photos) ===")

    if _res is not None and _res.get(f"state_{_tag}") is not None:
        _net = MeasurementNet(backbone=cfg["backbone"], pretrained=False, n_outputs=len(TARGETS),
                              hidden=cfg["hidden"], dropout=cfg["dropout"],
                              use_weight=_use_w).to(DEVICE)
        _net.load_state_dict(torch.load(_res[f"state_{_tag}"], map_location=DEVICE,
                                        weights_only=False)["model"])
        _hist = pd.read_parquet(_res[f"hist_{_tag}"])
        _meta = _res.get("_meta", {})
        print(f"  {_tag} reloaded from cache: best val MAE {_meta.get('best_val_mae_mm')} mm "
              f"over {_meta.get('epochs_completed')} epochs (stop={_meta.get('stop_reason')})")
    else:
        with guard(_phase) as _dev:
            _net = MeasurementNet(backbone=cfg["backbone"], pretrained=cfg["pretrained"],
                                  n_outputs=len(TARGETS), hidden=cfg["hidden"],
                                  dropout=cfg["dropout"], use_weight=_use_w,
                                  offline_weights=cfg["offline_weights"]).to(_dev)

            def _mk_interrupt(phase_name, tag):
                """Build an on_interrupt callback that records the phase as partial.

                Parameters
                ----------
                phase_name : str
                    Registry phase name.
                tag : str
                    Variant tag.

                Returns
                -------
                callable
                """
                def _cb(status, meta):
                    done(phase_name, {}, status=status, meta={**meta, "stage": "interrupted"})
                return _cb

            _net, _hist, _meta = T.train_model(
                _net, _ds_tr, _ds_va, TARGETS, cfg, _dev,
                ckpt_dir=RUNSTATE.ckpt_dir(_tag), tag=_tag, epochs=EPOCHS_PROD,
                worker_rng=AUG_RNG, on_interrupt=_mk_interrupt(_phase, _tag))
        _state = R.atomic_save({k: v.cpu() for k, v in _net.state_dict().items()},
                               RUNSTATE.results_dir(_phase) / f"{_tag}_state.pt", "torch")
        done(_phase, {f"hist_{_tag}": _hist, f"state_{_tag}": _state},
            status=_meta.get("status", "done"),
            extra_artefacts=[RUNSTATE.ckpt_dir(_tag) / "best.pt"],
            meta=_meta)

    if _tag == "vhw":
        net_vhw, hist_vhw = _net, _hist
    else:
        net_vh, hist_vh = _net, _hist
    TRAIN_META[_tag] = _meta

# --- the channel contract, proved on the real training datasets -------------------------------
_ok_ds = {"V-HW": train_ds_vhw, "V-H": train_ds_vh}
for _k, _ds in _ok_ds.items():
    if _ds is None:
        print(f"{_k}: dataset unavailable (phase blocked), channel check skipped")
        continue
    _x = _ds[0][0]
    print(f"{_k} tensor channels: {_x.shape[0]}")
    assert _x.shape[0] == (N_CHANNELS if _k == "V-HW" else N_CHANNELS_VH), (
        f"{_k} must carry {N_CHANNELS if _k == 'V-HW' else N_CHANNELS_VH} channels; the V-H "
        "weight channel must be removed, not zero-filled")
print("channel contract: V-H removed the weight channel rather than zero-filling it")

TRAIN_STEPS = {t: int(TRAIN_META[t].get("total_steps_run") or 0) for t in ("b2", "vhw", "vh")}
print("\noptimiser steps run:", json.dumps(TRAIN_STEPS))
print("reference: Ruiz et al. (2022) train for 150k iterations at batch 22.")
_lo = min(v for v in TRAIN_STEPS.values() if v) if any(TRAIN_STEPS.values()) else 0
print(f"    this run is {_lo/150000:.2%} of that budget at minimum. Every B0-beats-vision "
      "comparison below is PROVISIONAL (fix C1, DECISIONS.md D19).")
print(f"early stopping: {cfg['early_stopping']}, patience "
      f"{cfg['early_stopping_patience']}, min_delta {cfg['early_stopping_min_delta_mm']} mm")
for _t in ("vhw", "vh"):
    _m = TRAIN_META.get(_t) or {}
    if _m:
        print(f"    {_t:5s} epochs {_m.get('epochs_completed')}/{_m.get('epochs_configured')}, "
              f"stop={_m.get('stop_reason')}, early_stopped={_m.get('early_stopped')}")
print("mark_phase('production training') =", mark_phase("production training"), "s")