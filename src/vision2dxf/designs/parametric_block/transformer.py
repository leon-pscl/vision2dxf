from __future__ import annotations
from ...core.models import MeasurementProfile


def transform_block(
    base_points: dict[str, list[float]],
    ref: dict[str, float],
    m: MeasurementProfile,
    ease: float,
) -> dict[str, tuple[float, float]]:
    chest_ratio = (m.chest + ease) / (ref["chest"] + ease)
    waist_ratio = (m.waist + ease) / (ref["waist"] + ease)
    torso_ratio = m.torso_length / ref["torso_length"]
    shoulder_ratio = m.shoulder / ref["shoulder"]
    armhole_ratio = m.armhole_depth / ref["armhole_depth"]

    points: dict[str, tuple[float, float]] = {}
    for name, coords in base_points.items():
        x, y = float(coords[0]), float(coords[1])

        if name in ("center_front_neck", "center_front_waist"):
            x = x  # center front stays at x=0
        elif name in ("side_neck", "shoulder_tip"):
            x *= shoulder_ratio
        else:
            x *= chest_ratio

        if name in ("center_front_neck", "side_neck"):
            y = y  # top stays
        elif name in ("armhole_bottom",):
            y *= armhole_ratio
        elif name in ("side_waist", "center_front_waist", "dart_leg_inner", "dart_leg_outer"):
            y *= torso_ratio
        elif name == "shoulder_tip":
            y *= shoulder_ratio
        elif name == "dart_tip":
            y = y * torso_ratio

        points[name] = (round(x, 4), round(y, 4))

    return points
