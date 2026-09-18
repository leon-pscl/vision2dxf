from __future__ import annotations
import math
from ...core.models import MeasurementProfile


def select_template(
    index: dict, m: MeasurementProfile
) -> tuple[dict, float]:
    measurements = [m.chest, m.waist, m.hip]

    best_tpl = None
    best_dist = float("inf")

    for tpl in index["templates"]:
        ref = tpl["reference_measurements"]
        ref_vals = [ref["chest"], ref["waist"], ref["hip"]]
        max_vals = [max(r, v) for r, v in zip(ref_vals, measurements)]
        norm = [abs(r - v) / mx if mx > 0 else 0 for r, v, mx in zip(ref_vals, measurements, max_vals)]
        dist = math.sqrt(sum(n**2 for n in norm))
        if dist < best_dist:
            best_dist = dist
            best_tpl = tpl

    return best_tpl, best_dist
