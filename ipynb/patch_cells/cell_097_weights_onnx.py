# fix A6: EXPORT_DIR was used from §9 onward but never defined anywhere.
EXPORT_DIR = PATHS["export"]
print("EXPORT_DIR =", EXPORT_DIR)

# --- fix A8: save weights and ONNX, and verify ONNX == PyTorch on REAL tensors ---------------
CHECKPOINT_META = {
    "backbone": cfg["backbone"],
    "hidden": cfg["hidden"],
    "dropout": cfg["dropout"],
    "targets": list(TARGETS),
    "img_height": cfg["img_height"],
    "img_width": cfg["img_width"],
    "mask_threshold": cfg["mask_threshold"],
    "seed": cfg["seed"],
    "train_steps": None,          # filled per variant below
    "provisional": bool(SMOKE),
}


def save_variant(net: nn.Module, tag: str, use_weight: bool, steps: int) -> Dict[str, Path]:
    """Write one variant's PyTorch checkpoint and ONNX graph.

    Parameters
    ----------
    net : nn.Module
        Trained :class:`src.bodym_cnn.MeasurementNet` with best-val weights loaded.
    tag : str
        ``'vhw'`` or ``'vh'``; becomes the filename stem.
    use_weight : bool
        Variant's channel contract, recorded in the checkpoint.
    steps : int
        Optimiser steps used to train it, recorded so a reader can judge the budget.

    Returns
    -------
    dict
        ``{'pt': Path, 'onnx': Path}``.
    """
    meta = dict(CHECKPOINT_META)
    meta["use_weight"] = use_weight
    meta["train_steps"] = steps
    pt_path = EXPORT_DIR / f"model_{tag}.pt"
    torch.save({"model": {k: v.cpu() for k, v in net.state_dict().items()}, **meta}, pt_path)

    onnx_path = EXPORT_DIR / f"model_{tag}.onnx"
    dummy = torch.zeros(1, N_CHANNELS if use_weight else N_CHANNELS_VH,
                        cfg["img_height"], 2 * cfg["img_width"])
    net.eval()
    torch.onnx.export(
        net, dummy, str(onnx_path),
        input_names=["input"], output_names=["measurements_cm"],
        dynamic_axes={"input": {0: "batch"}, "measurements_cm": {0: "batch"}},
        opset_version=17, do_constant_folding=True)
    return {"pt": pt_path, "onnx": onnx_path}


EXPORT_PATHS = {
    "V-HW": save_variant(net_vhw, "vhw", True, TRAIN_STEPS["V-HW"]),
    "V-H": save_variant(net_vh, "vh", False, TRAIN_STEPS["V-H"]),
}
for _k, _v in EXPORT_PATHS.items():
    print(f"saved {_k}: {_v['pt'].name} ({_v['pt'].stat().st_size/1e6:.1f} MB), "
          f"{_v['onnx'].name} ({_v['onnx'].stat().st_size/1e6:.1f} MB)")

# --- verify ONNX == PyTorch on REAL validation tensors --------------------------------------
import onnxruntime as ort

_verify_rows = []
_n_check = min(int(cfg["n_onnx_check"]), len(val_ds_vhw))
_real_tensors = np.stack([val_ds_vhw[i][0].numpy() for i in range(_n_check)])
_real_photo_ids = [str(val_ds_vhw[i][3]) for i in range(_n_check)]
print(f"\nONNX verification on {_n_check} REAL validation tensors "
      f"(photo ids: {_real_photo_ids[:3]}...)")
print(f"tensor shape {_real_tensors.shape}, per-view size "
      f"{cfg['img_height']}x{cfg['img_width']}, "
      f"channels {N_CHANNELS} (V-HW)")

for _vname, _net in (("V-HW", net_vhw), ("V-H", net_vh)):
    _ds = val_ds_vhw if _vname == "V-HW" else val_ds_vh
    _n = min(_n_check, len(_ds))
    _xb = torch.from_numpy(np.stack([_ds[i][0].numpy() for i in range(_n)])).to(device)
    _net.eval()
    with torch.no_grad():
        _torch_out = _net(_xb).float().cpu().numpy()
    _sess = ort.InferenceSession(str(EXPORT_PATHS[_vname]["onnx"]),
                                 providers=["CPUExecutionProvider"])
    _onnx_out = _sess.run(None, {"input": _xb.cpu().numpy()})[0]
    _maxdiff = float(np.abs(_torch_out - _onnx_out).max())
    _verify_rows.append({"variant": _vname, "n_samples": _n,
                         "max_abs_diff_cm": _maxdiff,
                         "atol_cm": cfg["onnx_atol"] / 10.0,
                         "onnx_atol_config": cfg["onnx_atol"],
                         "pass": bool(_maxdiff < cfg["onnx_atol"] / 10.0)})
    print(f"  {_vname}: max |onnx - torch| = {_maxdiff*10:.3e} cm "
          f"({_maxdiff:.3e} cm) over {_n} samples")
    assert _maxdiff < cfg["onnx_atol"] / 10.0, (
        f"{_vname}: ONNX output differs from PyTorch by {_maxdiff:.3e} cm, above the "
        f"{cfg['onnx_atol']/10.0:.1e} cm tolerance. cfg['onnx_atol'] is expressed in mm; "
        "compare in the same unit.")

EXPORT_VERIFY = pd.DataFrame(_verify_rows)
EXPORT_VERIFY.to_csv(ART / "onnx_verification.csv", index=False)
assert bool(EXPORT_VERIFY["pass"].all()), "ONNX verification failed for at least one variant"
print(M.format_metric_table(EXPORT_VERIFY))
print("ONNX <-> PyTorch agreement verified on real tensors for both variants.")

# --- the checkpoint must be self-describing enough to rebuild the net -------------------------
for _vname, _net in (("V-HW", net_vhw), ("V-H", net_vh)):
    _ck = torch.load(EXPORT_PATHS[_vname]["pt"], map_location="cpu", weights_only=False)
    for _k in ("model", "backbone", "hidden", "dropout", "targets", "use_weight",
               "img_height", "img_width", "mask_threshold"):
        assert _k in _ck, f"{_vname} checkpoint is missing '{_k}'; infer.py needs it"
    _fresh = MeasurementNet(backbone=_ck["backbone"], pretrained=False,
                            n_outputs=len(_ck["targets"]), hidden=_ck["hidden"],
                            dropout=_ck["dropout"], use_weight=_ck["use_weight"])
    _fresh.load_state_dict(_ck["model"])
    _fresh.eval()
    with torch.no_grad():
        _a = _fresh(torch.from_numpy(_real_tensors[:2])).numpy()
        _b = _net(torch.from_numpy(_real_tensors[:2]).to(device)).float().cpu().numpy()
    print(f"  {_vname} checkpoint round-trip max |diff| = {float(np.abs(_a-_b).max()):.3e} cm")
    assert np.allclose(_a, _b, atol=1e-6), f"{_vname} checkpoint does not round-trip"

# --- reference tensors AND reference inputs, for the preprocessing self-test (fix D1) ---------
_ref_n = min(4, len(val_ds_vhw))
ref_idx = rng.choice(len(val_ds_vhw), size=_ref_n, replace=False)
REF_TENSORS = {"V-HW": [val_ds_vhw[int(i)][0].numpy() for i in ref_idx],
               "V-H": [val_ds_vh[int(i)][0].numpy() for i in ref_idx]}
# Save the INPUTS as well as the outputs. Without them the exported self-test can only build a
# synthetic mask and compare shapes, which cannot detect a changed resize filter or an
# inverted mask channel (fix D1).
REF_INPUTS: Dict[str, np.ndarray] = {}
REF_META: Dict[str, dict] = {}
_meta_t = IDX["train"].subjects.set_index("subject_id")
for _v, _ds in (("V-HW", val_ds_vhw), ("V-H", val_ds_vh)):
    for _k, _i in enumerate(ref_idx):
        _i = int(_i)
        _row = _ds.photos.iloc[_i]
        REF_INPUTS[f"front_{_v}_{_k}"] = np.array(Image.open(_row.front_mask))
        REF_INPUTS[f"side_{_v}_{_k}"] = np.array(Image.open(_row.side_mask))
        REF_INPUTS[f"height_cm_{_v}_{_k}"] = np.float32(float(_meta_t.loc[_row.subject_id, "height_cm"]))
        REF_INPUTS[f"weight_kg_{_v}_{_k}"] = np.float32(float(_meta_t.loc[_row.subject_id, "weight_kg"]))
    REF_META[_v] = {"photo_id": [str(_ds.photos.iloc[int(i)].photo_id) for i in ref_idx],
                    "targets": list(TARGETS)}
np.savez_compressed(EXPORT_DIR / "reference_tensors.npz",
                    **{f"{v}_{i}": t for v, lst in REF_TENSORS.items() for i, t in enumerate(lst)})
np.savez_compressed(EXPORT_DIR / "reference_inputs.npz", **REF_INPUTS)
(EXPORT_DIR / "reference_meta.json").write_text(json.dumps(REF_META, indent=2))
print("\nsaved reference tensors:", {v: [tuple(t.shape) for t in lst] for v, lst in REF_TENSORS.items()})
print("saved reference inputs:", len(REF_INPUTS), "arrays "
      "(front/side masks + height/weight scalars per variant per sample)")
assert len(REF_INPUTS) == 4 * _ref_n * 2, "reference inputs incomplete"

# The export modules read their constants from config.yaml / conformal.json rather than baking
# them in, so the bundle has exactly one source of truth for sizes and bounds.
if not (EXPORT_DIR / "config.yaml").is_file():
    shutil.copy2(CFG_PATH, EXPORT_DIR / "config.yaml")
_cfg_export = yaml.safe_load((EXPORT_DIR / "config.yaml").read_text(encoding="utf-8"))
_cfg_export["preprocess_atol"] = 1e-6
(EXPORT_DIR / "config.yaml").write_text(yaml.safe_dump(_cfg_export, sort_keys=True),
                                        encoding="utf-8")
print("config.yaml present next to the export modules (with preprocess_atol)")
print("mark_phase('weights + onnx') =", mark_phase("weights + onnx"), "s")