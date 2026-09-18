from __future__ import annotations


def get_formula_rules() -> dict:
    return {
        "required_landmarks": [
            "center_front_neck", "side_neck", "shoulder_tip",
            "armhole_bottom", "side_waist", "center_front_waist",
            "dart_leg_inner", "dart_leg_outer", "dart_tip",
        ],
        "min_width": 10.0,
        "min_height": 20.0,
        "max_width": 40.0,
        "max_height": 60.0,
    }
