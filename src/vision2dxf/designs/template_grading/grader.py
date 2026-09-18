from __future__ import annotations
from ...core.models import MeasurementProfile


def apply_grading(
    base_points: dict[str, list],
    grading_limits: dict,
    ref: dict[str, float],
    m: MeasurementProfile,
    ease: float,
) -> tuple[dict[str, tuple[float, float]], list[dict]]:
    chest_ratio = (m.chest + ease) / (ref["chest"] + ease)
    waist_ratio = (m.waist + ease) / (ref["waist"] + ease)
    torso_ratio = m.torso_length / ref["torso_length"]
    shoulder_ratio = m.shoulder / ref["shoulder"]
    armhole_ratio = m.armhole_depth / ref["armhole_depth"]

    adjusted: dict[str, tuple[float, float]] = {}
    grading_info: list[dict] = []

    for name, coords in base_points.items():
        x, y = float(coords[0]), float(coords[1])
        orig_x, orig_y = x, y

        if name in ("center_front_neck", "center_front_waist"):
            pass
        elif name in ("side_neck", "shoulder_tip"):
            x *= shoulder_ratio
        else:
            x *= chest_ratio

        if name in ("center_front_neck", "side_neck"):
            pass
        elif name in ("armhole_bottom",):
            y *= armhole_ratio
        elif name in ("side_waist", "center_front_waist", "dart_leg_inner", "dart_leg_outer"):
            y *= torso_ratio
        elif name == "shoulder_tip":
            y *= shoulder_ratio
        elif name == "dart_tip":
            y *= torso_ratio

        limited = False
        if name in grading_limits:
            lim = grading_limits[name]
            dx = x - orig_x
            dy = y - orig_y
            cx = max(lim["x"][0], min(lim["x"][1], dx))
            cy = max(lim["y"][0], min(lim["y"][1], dy))
            if cx != dx or cy != dy:
                limited = True
            x = orig_x + cx
            y = orig_y + cy

        adjusted[name] = (round(x, 4), round(y, 4))
        grading_info.append({
            "landmark": name,
            "requested_dx": round(x - orig_x, 4),
            "requested_dy": round(y - orig_y, 4),
            "applied_dx": round(x - orig_x, 4),
            "applied_dy": round(y - orig_y, 4),
            "limit_reached": limited,
        })

    return adjusted, grading_info
