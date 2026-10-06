import bodym_conformal as C

# --- unit test of the quantile rule against a hand-checked case --------------------------
_r = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
_q80 = C.conformal_quantile(_r, 0.80)
print(f"n=5, coverage=0.80 -> level = ceil(6*0.8)/5 = {np.ceil(6*0.8)/5:.2f} -> q = {_q80}")
assert _q80 == 5.0, "conformal quantile must use the finite-sample correction and 'higher' method"
try:
    C.conformal_quantile(_r, 1.5)
    raise AssertionError("expected ValueError for coverage > 1")
except ValueError:
    pass
print("conformal quantile self-check passed")

# fix B12: q is a HALF-WIDTH, so the interval is [pred - q, pred + q] and the full width is
# 2q. The previous table and summary printed q*20 as "halfwidth_mm", double-counting.
_half = C.conformal_quantile(_r, 0.90)
print(f"half-width check: pred=100 cm, q90={_half} cm -> interval "
      f"[{100-_half}, {100+_half}] cm, width {2*_half} cm = {2*_half*10:.0f} mm")
print(f"  half-width in mm = q*10 = {_half*10:.0f} mm  (the old code printed q*20 = {_half*20:.0f} mm)")
assert abs(_half * 10 - 10.0) < 1e-9, "half-width conversion is q*10, not q*20"

# fix B12: clamping must never exclude the point estimate from its own interval.
_sc = C.clamp_scalar_intervals(100.0, {"80": 30.0, "90": 45.0}, (95.0, 105.0))
print("clamped against a too-tight envelope (95, 105):", json.dumps(_sc, indent=2))
assert _sc["lo80"] <= 100.0 <= _sc["hi80"], "clamp excluded the point estimate (fix B12)"
assert _sc["lo90"] <= _sc["lo80"] and _sc["hi80"] <= _sc["hi90"], "clamp broke interval nesting"
assert _sc["clamped"] is True, "clamp must be reported"
print("conformal self-checks passed")