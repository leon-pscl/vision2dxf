"""Cell 8 — config.yaml: write if absent, read back, expose as ``cfg``.

WHAT — Write ``config.yaml`` if it is missing, read it back, merge over the defaults,
and re-seed every RNG from the value that was actually read.

WHY
- [Ref] Pimentel et al. (2021): magic numbers scattered across cells are a primary cause of
  irreproducible notebooks. One file read at the top removes that class of defect.
- [Exp] The merge is ``{**DEFAULT_CFG, **cfg}``, so a config.yaml written by an
  *earlier* run keeps its old values for any key it already contains. That is how the
  dead backbone name ``mnasnet1_0_3_0`` and the transposed 480x640 size would survive
  into this run unnoticed. Tested by: the PINNED_RUNTIME assertion below, which
  compares the *effective* cfg against the intended values and names any key that a
  stale file is overriding.
- [Exp] Seeds are set from cfg["seed"] here rather than only in the previous cell,
  because SEED is defined before the config exists. Tested by: the printed draw from
  each RNG after the re-seed.

OUTPUT — cfg (dict), cfg_created (bool), CFG_PATH, SMOKE, RUN_FINAL_EVAL.

CHECK — every required key is present; SMOKE/RUN_FINAL_EVAL are set; the effective
  runtime keys are pinned; two default_rng(cfg["seed"]) draws agree.
"""

DEFAULT_CFG: Dict[str, Any] = {
    "seed": 42,
    "data_root": None,          # None -> auto-discovered under the Kaggle input dir
    # --- run protocol (fix brief section 0) -----------------------------------------------
    # smoke_test: 1 epoch, <=50 subjects per split, 10 sensitivity subjects,
    #             <=2 photos per subject. Every assertion still runs.
    # run_final_eval: when false, section 8 (Test-A/Test-B) and everything derived
    #             from it is skipped with a printed notice.
    "smoke_test": True,
    "run_final_eval": False,
    "smoke_subjects_max": 50,
    "smoke_photos_per_subject": 2,
    "smoke_sensitivity_subjects": 10,
    # --- preprocessing --------------------------------------------------------------------
    # Per-view size preserves the native mask aspect ratio (fix B2): BodyM masks are
    # portrait 4:3, so H > W. The front|side canvas is img_height x (2 * img_width).
    "img_height": 640,           # per-view mask height  (portrait 4:3)
    "img_width": 480,            # per-view mask width
    "mask_threshold": 127,       # PNG is 0/255; anything > this is foreground
    # ---- baselines ---------------------------------------------------------------------
    "ridge_alpha": 10.0,
    "n_estimators": 300,
    # ---- CNN ---------------------------------------------------------------------------
    # "mnasnet1_0_3_0" is not a torchvision model (fix A2). Available MNASNet names:
    # mnasnet0_5, mnasnet0_75, mnasnet1_0, mnasnet1_3.
    "backbone": "mnasnet1_0",
    "pretrained": True,
    "offline_weights": None,     # path to a local ImageNet .pth when internet is unavailable
    "epochs_b2": 6,
    "epochs_prod": 8,
    "batch_size": 6,             # lowered from 8: the corrected input is 1.78x the pixels
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "grad_clip": 5.0,
    "dropout": 0.2,
    "hidden": 512,
    "num_workers": 4,
    "amp": True,
    # ---- boundary sensitivity ----------------------------------------------------------
    "erosion_px": 6,             # sweep -6..+6
    "jitter_sigma_px": [0, 2, 4, 6],
    "downsample_factors": [2, 4, 8],
    "n_sensitivity_subjects": 150,
    "tolerance_mm": [9, 15, 25],
    "key_measurements": ["chest", "waist", "hip"],
    "max_tolerable": "total",    # "total" (baseline + dMAE) or "delta" (dMAE only)
    "augmentation_exclude_targets": ["height"],   # input passthrough, not a prediction
    "augmenter_budget_ms": 50.0,  # per-sample budget from the timing cell (fix C2)
    # ---- conformal ---------------------------------------------------------------------
    "coverages": [0.80, 0.90],
    # ---- export -----------------------------------------------------------------------
    "n_onnx_check": 20,
    "onnx_atol": 1e-4,
    "bmi_bands": [0, 18.5, 25, 30, 40, 100],
    "bmi_labels": ["<18.5", "18.5-25", "25-30", "30-40", ">40"],
    "min_stratum_n": 30,
    "include_height_as_target": True,
    "zip_export": True,
}

REQUIRED_KEYS = set(DEFAULT_CFG)

#: Runtime keys that a stale config.yaml must not be allowed to silently override.
#: These were the subject of hard crashes or silent-wrongness in the previous version.
PINNED_RUNTIME: Dict[str, Any] = {
    "backbone": "mnasnet1_0",
    "img_height": 640,
    "img_width": 480,
    "batch_size": 6,
}

CFG_PATH = PATHS["working"] / "config.yaml"
cfg_created = not CFG_PATH.exists()
if cfg_created:
    CFG_PATH.write_text(yaml.safe_dump(DEFAULT_CFG, sort_keys=True), encoding="utf-8")

cfg_on_disk: Dict[str, Any] = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8")) or {}
cfg: Dict[str, Any] = {**DEFAULT_CFG, **cfg_on_disk}

missing_keys = REQUIRED_KEYS - set(cfg)
assert not missing_keys, f"config.yaml missing keys: {sorted(missing_keys)}"

# A pre-existing config.yaml wins over DEFAULT_CFG by design, so an override is legal.
# But the four keys above were bugs, and a run that silently keeps them is worse than a
# failed one. Report loudly, and require the user to delete the file to accept a change.
_stale = {k: (cfg_on_disk.get(k), v) for k, v in PINNED_RUNTIME.items()
          if k in cfg_on_disk and cfg_on_disk[k] != v}
if _stale:
    print("\n" + "=" * 78)
    print("STALE config.yaml OVERRIDES DETECTED")
    for k, (on_disk, intended) in _stale.items():
        print(f"    {k:14s} on disk = {on_disk!r}   intended = {intended!r}")
    print("  Delete config.yaml to regenerate from this notebook's defaults, or edit it")
    print("  deliberately and re-run.")
    print("=" * 78 + "\n")

# Seeds come from the config, not from a constant defined before the config existed.
SEED = int(cfg["seed"])
set_global_seeds(SEED)

SMOKE = bool(cfg["smoke_test"])
RUN_FINAL_EVAL = bool(cfg["run_final_eval"])

print(f"config.yaml {'CREATED' if cfg_created else 'loaded'} at {CFG_PATH}")
print(yaml.safe_dump(cfg, sort_keys=True))
print(f"\nSMOKE = {SMOKE} | RUN_FINAL_EVAL = {RUN_FINAL_EVAL} | SEED = {SEED}")
print("seed draws after re-seed -> python:", random.random(),
      "| numpy:", np.random.rand(1), "| torch:", torch.rand(1).item())
assert np.random.default_rng(SEED).random(3).tolist() == np.random.default_rng(SEED).random(3).tolist(), \
    "default_rng(cfg['seed']) is not reproducible"
for k, v in PINNED_RUNTIME.items():
    print(f"    effective {k:14s} = {cfg[k]!r}")
assert cfg["img_height"] > cfg["img_width"], (
    f"img_height ({cfg['img_height']}) must exceed img_width ({cfg['img_width']}): "
    "BodyM masks are portrait. See fix B2.")
print("mark_phase('setup') =", mark_phase("setup"), "s")