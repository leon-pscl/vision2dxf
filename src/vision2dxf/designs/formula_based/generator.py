from __future__ import annotations
from ...core.models import MeasurementProfile, PatternResult


class FormulaBasedGenerator:
    design_id = "design_a_formula"

    def generate(self, m: MeasurementProfile) -> PatternResult:
        ease = m.ease
        chest_ease = m.chest + ease
        waist_ease = m.waist + ease

        nw = 0.08 * m.chest
        nd = 0.075 * m.chest
        sh_len = 0.30 * m.shoulder
        armhole_d = 0.50 * m.armhole_depth
        bust_half = 0.25 * chest_ease
        waist_half = 0.25 * waist_ease
        dart_w = 0.04 * waist_ease
        dart_l = 0.10 * m.torso_length

        points: dict[str, tuple[float, float]] = {
            "center_front_neck": (0.0, 0.0),
            "side_neck": (nw, 0.0),
            "shoulder_tip": (nw + sh_len, 0.05 * m.shoulder),
            "armhole_bottom": (bust_half, armhole_d),
            "side_waist": (bust_half, m.torso_length),
            "center_front_waist": (0.0, m.torso_length),
            "dart_leg_inner": (waist_half - dart_w / 2, m.torso_length),
            "dart_leg_outer": (waist_half + dart_w / 2, m.torso_length),
            "dart_tip": (waist_half, m.torso_length - dart_l),
        }

        segments = [
            {"type": "line", "from": "center_front_neck", "to": "side_neck"},
            {"type": "line", "from": "side_neck", "to": "shoulder_tip"},
            {"type": "curve", "from": "shoulder_tip", "to": "armhole_bottom",
             "control_points": [(bust_half + 1, armhole_d * 0.45), (bust_half + 2, armhole_d * 0.8)]},
            {"type": "line", "from": "armhole_bottom", "to": "side_waist"},
            {"type": "line", "from": "side_waist", "to": "center_front_waist"},
            {"type": "line", "from": "dart_leg_inner", "to": "dart_tip"},
            {"type": "line", "from": "dart_tip", "to": "dart_leg_outer"},
        ]

        return PatternResult(
            design_id=self.design_id,
            points=points,
            segments=segments,
            valid=False,
            validation_errors=[],
            metadata={
                "formula_version": "1.0",
                "ease_applied": ease,
                "units": "cm",
                "status": "prototype_only",
            },
        )
