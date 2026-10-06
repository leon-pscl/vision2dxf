# --- freeze the conformal files -----------------------------------------------------------
PLAUSIBLE_CM: Dict[str, Tuple[float, float]] = {
    # Wide enough not to truncate a real body, tight enough that an interval reaching the bound is
    # visibly wrong. TODO(user): replace with the pattern engine's own plausibility envelope.
    "chest": (55.0, 180.0), "waist": (45.0, 180.0), "hip": (60.0, 190.0),
    "shoulder-breadth": (22.0, 60.0), "shoulder-to-crotch": (40.0, 100.0),
    "arm-length": (30.0, 70.0), "leg-length": (50.0, 110.0),
    "thigh": (30.0, 90.0), "calf": (22.0, 60.0), "ankle": (14.0, 40.0),
    "bicep": (15.0, 55.0), "forearm": (14.0, 45.0), "wrist": (10.0, 30.0),
    "height": (130.0, 210.0),
}
assert set(PLAUSIBLE_CM) == set(TARGETS), (
    f"plausibility envelope must cover every target; missing "
    f"{sorted(set(TARGETS) - set(PLAUSIBLE_CM))}")

CONFORMAL_PATH = C.save_conformal(
    CONFORMAL_QUANTILES, PATHS["export"] / "conformal.json", extra={
        "generated_by": "model_training_measurement.ipynb",
        "seeds": {"master": cfg["seed"]},
        # fix B5 / U2: the calibration unit is one photo pair per subject
        "unit": "photo_pair",
        "pairs_per_subject": 1,
        "unit_rationale": (
            "Deployment predicts from ONE front/side pair (infer.py signature). Calibration "
            "scores are therefore computed from one seeded random photo pair per calibration "
            "subject, not from a subject-averaged prediction. Averaging first would make the "
            "scores exchangeable with a quantity the model never produces at inference."),
        "n_calibration_subjects": len(ids_calib),
        "n_calibration_pairs_per_variant": {k: v["n_calibration_predictions"]
                                            for k, v in CONFORMAL_META.items()},
        "method": "split conformal, per measurement, |residual| order statistic at "
                  "level = ceil((n+1)*coverage)/n with method='higher'",
        "quantile_units": "centimetres; each value is a HALF-WIDTH, intervals are [pred-q, pred+q]",
        "coverage_guarantee": "marginal over calibration and test photo pairs jointly, "
                              "conditional on exchangeability; NOT conditional per-subject",
        "assumptions_and_breaks": [
            "Exchangeability of calibration and deployment photo pairs.",
            "Breaks under segmenter shift: quantiles are fitted on BodyM DeepLabv3+-derived masks.",
            "Breaks under capture/device shift: a different camera changes the nominal mm/px scale.",
            "Breaks under population shift: the high-BMI tail is thin in BodyM.",
            "Breaks under pose shift: one pair per subject is one draw from that subject's "
            "capture conditions, not the average over them.",
            "MUST be recalibrated on tape-measured ground truth from pipeline step 10.",
        ],
        "per_variant_diagnostics": CONFORMAL_META,
        "plausibility_bounds_cm": {k: list(v) for k, v in PLAUSIBLE_CM.items()},
    })
print("written:", CONFORMAL_PATH)

# fix B5: assert the unit field the whole calibration design rests on actually shipped.
_shipped = json.loads(CONFORMAL_PATH.read_text(encoding="utf-8"))
assert _shipped["unit"] == "photo_pair", "conformal.json must record unit='photo_pair'"
assert _shipped["pairs_per_subject"] == 1, "conformal.json must record pairs_per_subject=1"
assert set(_shipped["conformal_quantiles_cm"]) == {"V-HW", "V-H"}
print(f"conformal.json records unit={_shipped['unit']!r}, "
      f"pairs_per_subject={_shipped['pairs_per_subject']}, "
      f"{len(_shipped['conformal_quantiles_cm'])} variants")

# Demonstrate the plausibility gate step 7 will run, on a REAL V-HW validation prediction.
# The previous demo fed in the mean of a B0 feature vector repeated across all 14 targets, which
# produced 14 identical intervals and demonstrated nothing about the model.
_demo_subject = subs_vhw[0] if len(subs_vhw) else None
_demo_pred_row = P_VHW_val[0]
_demo_iv = C.apply_intervals(_demo_pred_row[None, :], CONFORMAL_QUANTILES["V-HW"], TARGETS)
_demo_clamped = C.clamp_intervals(_demo_iv, PLAUSIBLE_CM)
_demo_truth = Y_VHW_val[0]
print(f"\nplausibility-gate demo on a real V-HW validation prediction "
      f"(subject {str(_demo_subject)[:20]}):")
print("measurement          value_cm    truth_cm   lo90_mm   hi90_mm   in90  clamped")
for _j, _c in enumerate(TARGETS):
    _lo, _hi = _demo_clamped[0.9][_c]
    _inside = float(_lo) <= _demo_truth[_j] <= float(_hi)
    _clamped = (abs(float(_lo) - float(_iv[0.9][_c][0])) > 1e-12
                or abs(float(_hi) - float(_iv[0.9][_c][1])) > 1e-12)
    print(f"  {_c:18s} {_demo_pred_row[_j]:8.2f} {_demo_truth[_j]:11.2f} "
          f"{float(_lo)*10:9.2f} {float(_hi)*10:9.2f}   {'y' if _inside else 'N'}    "
          f"{'yes' if _clamped else 'no'}")
    assert float(_lo) <= _demo_pred_row[_j] <= float(_hi), (
        f"{_c}: clamping excluded the point estimate from its own interval (fix B12)")
print("every interval still contains its own point estimate after clamping")
print("mark_phase('conformal') =", mark_phase("conformal"), "s")