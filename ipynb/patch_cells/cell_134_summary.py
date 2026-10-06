# =========================================================================================
# REPRODUCIBILITY SUMMARY
# =========================================================================================
PHASE_TIMES.pop("_last", None)
total = time.time() - t_notebook_start

print("=" * 78)
print("REPRODUCIBILITY SUMMARY — BodyM measurement model")
print("=" * 78)

print(f"\nRUN PROTOCOL")
print(f"    {'smoke_test':28s} {SMOKE}")
print(f"    {'run_final_eval':28s} {RUN_FINAL_EVAL}")
if SMOKE:
    print("    ** SMOKE RUN: 1 epoch, subsampled splits. Metrics here are not results. **")

print(f"\nwall clock total        : {total/60:.1f} min ({total:.0f} s)")
print("per-phase seconds (measured runtimes; the §0.5 table was removed as ungrounded):")
for k, v in PHASE_TIMES.items():
    if k != "_last":
        print(f"    {k:28s} {v:9.1f} s   ({100*v/max(total,1):5.1f} %)")

print("\nENVIRONMENT")
for k in ("python", "platform", "numpy", "pandas", "scikit_learn", "scipy", "torch",
          "torchvision", "matplotlib", "pillow"):
    print(f"    {k:28s} {ENV[k]}")
print(f"    {'GPU':28s} {ENV['gpu']['name'] if ENV['gpu']['available'] else 'none (CPU only)'}"
      f"  capability {ENV['gpu']['capability']}, {ENV['gpu']['total_gb']} GB")
print(f"    {'internet':28s} {ENV['internet']}")
print(f"    {'kaggle session':28s} {ENV['kaggle_env']}")
print(f"    {'backbone weights':28s} {net_vhw.weights_source}")

print("\nSEEDS")
print(f"    master seed                {cfg['seed']} (python, numpy, torch, cuda, dataloader workers)")
print(f"    cudnn.deterministic        {torch.backends.cudnn.deterministic}")
print(f"    {'re-seeded after config':28s} yes (SEED = cfg['seed'])")

print("\nDATA")
print(f"    root                       {DATA_ROOT}")
for s in SPLITS:
    idx = INTEGRITY_INDEX[s]
    print(f"    {s:26s} {len(idx.subjects):5d} subjects  {len(idx.photos):5d} silhouettes  "
          f"masks front+side = {2*len(idx.photos)} files")
print(f"    integrity checks           "
      f"{int((integrity_report.status=='ok').sum())} ok, "
      f"{int((integrity_report.status=='warn').sum())} warn, "
      f"{int((integrity_report.status=='fail').sum())} fail")
print("\n    photos per subject (min / median / max):")
for _s, _idx in INTEGRITY_INDEX.items():
    _p = _idx.photos.groupby("subject_id").size()
    print(f"        {_s:22s} {_p.min():4d} / {_p.median():6.1f} / {_p.max():4d}")
print("    NOTE: evaluation is per photo pair, so a subject with more photos contributes")
print("          more rows. This is deliberate and matches single-capture deployment.")

print("\nSPLITS (train only, by subject)")
print(f"    fit {len(ids_fit)} | val {len(ids_val)} | conformal calibration {len(ids_calib)}")
print(f"    validation unit: {photos_per_subject_note(IDX['train'], ids_val)}")
if RUN_FINAL_EVAL:
    print(f"    Test-A {len(TEST_IDS['testA'])} and Test-B {len(TEST_IDS['testB'])} "
          "evaluated exactly once, in section 8")
else:
    print("    Test-A / Test-B: NOT EVALUATED (run_final_eval: false)")

print("\nMODELS  (optimiser steps; see DECISIONS.md D19)")
for name, hist, steps in (("B2 baseline", hist_b2, TRAIN_STEPS["B2"]),
                          ("V-HW", hist_vhw, TRAIN_STEPS["V-HW"]),
                          ("V-H", hist_vh, TRAIN_STEPS["V-H"])):
    print(f"    {name:28s} epochs {len(hist):2d}  steps {steps:7d}  "
          f"best val MAE {hist.val_mae_mm.min():7.2f} mm "
          f"(epoch {int(hist.loc[hist.val_mae_mm.idxmin(),'epoch'])})")
for name, pred in (("B0 ridge", P0_val_ridge), ("B0 gbm", P0_val_gbm)):
    print(f"    {name:28s} val MAE {np.abs(pred - Y_val).mean()*10:7.2f} mm "
          "(per subject, one photo)")
print(f"    {'B1 geometric':28s} val MAE {metrics_B1.mae_mm.mean():7.2f} mm (per subject)")
print("    reference: Ruiz et al. (2022) train for 150,000 iterations at batch 22.")

print("\nSCALAR PATH (fix B1)")
for _v, _d in SCALAR_DELTA.items():
    print(f"    {_v:28s} mean |Δ| for a +10% height_cm, per girth (mm): "
          + ", ".join(f"{c}={_d[c]:.2f}" for c in cfg["key_measurements"]))
print("    (Previously the scalars never received gradient: two zero-initialised convolutions")
print("     in series, so only a bias could learn and it could not see height or weight.)")

print("\nAUGMENTER (fix C2)")
print(f"    {'ms per sample':28s} {AUG_MS_PER_SAMPLE:.1f} "
      f"(budget {cfg['augmenter_budget_ms']:.0f})")
print(f"    {'erosion / dilation':28s} ±{AUG_MAG['erosion_px']} / +{AUG_MAG['dilation_px']} px")
print(f"    {'jitter sigma':28s} {AUG_MAG['jitter_sigma_px']} px")
print(f"    {'downsample factor':28s} x{AUG_MAG['downsample_factor']}")
print(f"    {'binding measurement':28s} {AUG_MAG['binding_measurement']}")
print(f"    {'rule':28s} all targets except {AUG_MAG['excluded_targets']} within "
      f"{AUG_MAG['tolerance_mm_used']} mm")

print("\nBOUNDARY SENSITIVITY")
print(f"    {'nominal scale':28s} {MM_PER_PX_NOMINAL:.3f} mm/px")
print(f"    {'swept models':28s} {sorted(SENSITIVITY.model.unique())}")
_k = MAX_TOLERABLE_BOUNDARY[(MAX_TOLERABLE_BOUNDARY.model == "V-HW")
                            & (MAX_TOLERABLE_BOUNDARY.measurement.isin(cfg["key_measurements"]))
                            & (MAX_TOLERABLE_BOUNDARY.family == "morph")
                            & (MAX_TOLERABLE_BOUNDARY.side == "erosion")
                            & (MAX_TOLERABLE_BOUNDARY.criterion == cfg["max_tolerable"])
                            & (MAX_TOLERABLE_BOUNDARY.tolerance_mm == cfg["tolerance_mm"][-1])]
if len(_k):
    print(f"    {'V-HW max tolerable erosion':28s} at {cfg['tolerance_mm'][-1]:.0f} mm: "
          + ", ".join(f"{r.measurement} {r.max_tolerable_px:.0f} px"
                      if isinstance(r.max_tolerable_px, float) else f"{r.measurement} {r.max_tolerable_px}"
                      for r in _k.itertuples()))

print("\nCONFORMAL PREDICTION")
for v in ("V-HW", "V-H"):
    _half90 = float(np.mean([CONFORMAL_QUANTILES[v][m]["90"] for m in TARGETS]) * 10.0)
    _line = (f"    {v:28s} n_calib {len(ids_calib):4d} pairs (1/subject)  "
             f"mean 90% half-width {_half90:6.1f} mm")
    if COVERAGE_SUMMARY is not None:
        for _sp in ("testA", "testB"):
            _r = COVERAGE_SUMMARY[(COVERAGE_SUMMARY.split == _sp)
                                 & (COVERAGE_SUMMARY.variant == v)]
            _line += (f"  {_sp} {float(_r.empirical_90.iloc[0]):.1f}%" if len(_r) else "")
    else:
        _line += "  (no held-out coverage: run_final_eval false)"
    print(_line)
print("    q is a half-width: mm = q x 10, not q x 20.")

print(f"\nFLAGGED (variant does not beat B0 on validation): {len(FLAGS)} measurement(s)")
print("  PROVISIONAL — decided at the training budget above (DECISIONS.md D19).")
if len(FLAGS):
    for r in FLAGS.itertuples():
        print(f"    {r.measurement:22s} B0 {r.mae_B0_mm:6.2f} mm | V-HW {r.mae_VHW_mm:6.2f} | "
              f"V-H {r.mae_VH_mm:6.2f}")

print("\nEXPORTED FILES")
for p in sorted(EXPORT_DIR.iterdir()):
    if p.is_file():
        print(f"    {p.name:42s} {p.stat().st_size/1024:10.1f} KiB")
zip_p = PATHS["working"] / "export.zip"
if zip_p.is_file():
    print(f"    {'export.zip':42s} {zip_p.stat().st_size/1e6:10.1f} MB")
print(f"\n    export dir : {EXPORT_DIR}")
print(f"    figures    : {ART}")
print(f"    checkpoints: {PATHS['working'] / 'checkpoints'}")
print(f"    config     : {CFG_PATH}")

print("\nOPEN DECISIONS FOR THE USER (not resolved by this notebook)")
print("    1. does the capture protocol collect weight? (determines the default variant)")
print("    2. pattern-engine key names for schema_map.yaml  [TODO(user)]")
print("    3. per-measurement tolerance thresholds the pattern engine accepts  [TODO(user)]")
print("    4. plausibility envelope for interval clamping  [TODO(user)]")
print("    5. commercial licensing clearance (CC BY-NC vs registry CC BY)")
print("    6. which segmenter step 3 uses, and whether it fits the boundary budget")
print("    7. whether to keep mesh-derived height as a 14th target")
print("    8. whether B1 belongs in the shipped bundle")
print("    9. whether the training budget is sufficient for the B0 flag table to be final")

print("\nCLAIMS EXPLICITLY NOT MADE")
print("    - ISO 20685 compliance (targets are SMPL mesh vertex-path lengths, not tape measurements)")
print("    - per-subject or per-capture conditional coverage (conformal coverage is marginal)")
print("    - performance outside the BodyM height / BMI / sex distribution")
print("    - commercial use of the training data or the derived weights")
print("    - that any measurement which fails to beat B0 is unrecoverable (the budget is small)")
print("=" * 78)