# --- fix C2: time the augmenter before committing to a multi-hour run -------------------------
# The production variants train with this callable on every sample of every epoch. If it is
# slower than the GPU, the run is augmenter-bound and the fix is to cache the distance
# transform (boundary_jitter recomputes two EDT passes per view per sample).
_n_time = 20
_times = []
for _i in range(_n_time):
    _t0 = time.perf_counter()
    AUGMENTER(_probe_f.copy(), _probe_s.copy())
    _times.append((time.perf_counter() - _t0) * 1000.0)
_ms_per_sample = float(np.median(_times))
print(f"augmenter: median {np.median(_times):.1f} ms/sample over {_n_time} calls "
      f"(p10 {np.percentile(_times,10):.1f}, p90 {np.percentile(_times,90):.1f})")
print(f"budget from config.yaml: {cfg['augmenter_budget_ms']:.0f} ms/sample")
AUG_MS_PER_SAMPLE = _ms_per_sample
if _ms_per_sample > cfg["augmenter_budget_ms"]:
    print("!! OVER BUDGET. With num_workers="
          f"{cfg['num_workers']} the DataLoader must sustain "
          f"{1000.0/_ms_per_sample*cfg['num_workers']:.0f} samples/s just to keep up. Options, "
          "in order of value:")
    print("   1. cache the signed-distance transform per sample and jitter only the boundary")
    print("   2. skip the downsample step when downsample_factor is 1")
    print("   3. reduce rotation/scale/translate to PIL affine (cheaper than ndi.rotate+shift)")
    print("   Logged in DECISIONS.md as an open performance item; the run continues.")
else:
    print("within budget")
print("mark_phase('augmenter timing') =", mark_phase("augmenter timing"), "s")