"""Derived drafting dimensions and the assembled rule registry.

Only girth-derived widths, angle handling and the shoulder/neckline runs live here;
panel geometry is built from the landmark values in drafting/panels.py.
"""
from __future__ import annotations

from math import cos, radians, sin

from ..config import drafting_config
from .landmarks import build_landmark_rules
from .measurements import BodyMeasurements, MeasurementWarning
from .rules import DraftingRule, RuleRegistry

# Inputs supplied to the registry that are not themselves produced by a rule:
# the confirmed millimetre measurements, the resolved ease and the explicit overrides.
EXTERNAL_INPUTS = (
    "bust", "waist", "back_width", "waist_back_width", "neck_w", "shoulder_w",
    "armscye_depth", "bust_points", "vert_bust_line",
    "bust_level", "waist_level",
    "ease_bust", "ease_waist",
    "shoulder_slope_deg_input", "armscye_depth_input", "bust_apex_y_input", "dart_intake_input",
)


def build_registry(config: dict | None = None) -> RuleRegistry:
    cfg = config or drafting_config()
    c = cfg["constants"]
    ranges = cfg["ranges"]

    rules: list[DraftingRule] = []

    def rule(
        rule_id: str,
        output_name: str,
        inputs: list[str],
        equation,
        explanation: str,
        source_note: str,
        valid_range: dict | None = None,
    ) -> None:
        rules.append(
            DraftingRule(
                id=rule_id,
                name=output_name.replace("der.", "Derived: "),
                output_name=output_name,
                required_inputs=inputs,
                equation=equation,
                explanation=explanation,
                source_note=source_note,
                valid_range=valid_range or {},
                group="derived",
            )
        )

    # -- ease ----------------------------------------------------------------
    rule("der.bust_with_ease", "der.bust_with_ease", ["bust", "ease_bust"],
         lambda v: v["bust"] + v["ease_bust"],
         "Bust girth plus the selected bust ease.",
         "Ease preset from config/drafting.yaml ease_presets.")
    rule("der.waist_with_ease", "der.waist_with_ease", ["waist", "ease_waist"],
         lambda v: v["waist"] + v["ease_waist"],
         "Waist girth plus the selected waist ease.",
         "Ease preset from config/drafting.yaml ease_presets.")

    # -- front / back width split at the side seam ---------------------------
    rule("der.back_bust_half_width", "der.back_bust_half_width", ["back_width"],
         lambda v: v["back_width"] / 2.0,
         "Back half-pattern width at the bust line = back_width / 2.",
         "Measurement dictionary: back_width is measured across the back at the bust line.",
         {"back_width": (150.0, 700.0)})
    rule("der.front_bust_half_width", "der.front_bust_half_width",
         ["der.bust_with_ease", "back_width"],
         lambda v: (v["der.bust_with_ease"] - v["back_width"]) / 2.0,
         "Front half-pattern width at the bust line = (bust + ease - back_width) / 2.",
         "Side seam placed where the front and back widths close on the total girth.",
         {"back_width": (150.0, 700.0)})
    rule("der.back_waist_half_width", "der.back_waist_half_width", ["waist_back_width"],
         lambda v: v["waist_back_width"] / 2.0,
         "Back half-pattern width at the waist line = waist_back_width / 2.",
         "Measurement dictionary: waist_back_width is measured across the back at the waist line.")
    rule("der.front_waist_half_width", "der.front_waist_half_width",
         ["der.waist_with_ease", "waist_back_width"],
         lambda v: (v["der.waist_with_ease"] - v["waist_back_width"]) / 2.0,
         "Front half-pattern width at the waist line = (waist + ease - waist_back_width) / 2.",
         "Same side-seam split rule as the bust line.")

    # -- neck ----------------------------------------------------------------
    rule("der.neck_shoulder_x", "der.neck_shoulder_x", ["neck_w"],
         lambda v: v["neck_w"] * c["neck_shoulder_share"],
         "Horizontal distance centre neck point to neck-shoulder point = neck_w * neck_shoulder_share.",
         "neck_w is read as half neck girth; see config/measurements.yaml.")
    rule("der.front_neck_depth", "der.front_neck_depth", ["der.neck_shoulder_x"],
         lambda v: v["der.neck_shoulder_x"] * c["front_neck_depth_ratio"],
         "Front neckline depth = neckline half width * front_neck_depth_ratio.",
         "config/drafting.yaml constants.front_neck_depth_ratio.")
    rule("der.front_neck_dip_x", "der.front_neck_dip_x", ["der.neck_shoulder_x"],
         lambda v: v["der.neck_shoulder_x"] * c["front_neck_dip_x_ratio"],
         "Horizontal position of the front neckline low point.",
         "config/drafting.yaml constants.front_neck_dip_x_ratio.")
    rule("der.back_neck_depth", "der.back_neck_depth", ["der.neck_shoulder_x"],
         lambda v: v["der.neck_shoulder_x"] * c["back_neck_depth_ratio"],
         "Back neckline depth = neckline half width * back_neck_depth_ratio.",
         "config/drafting.yaml constants.back_neck_depth_ratio.")
    rule("der.back_neck_dip_x", "der.back_neck_dip_x", ["der.neck_shoulder_x"],
         lambda v: v["der.neck_shoulder_x"] * c["back_neck_dip_x_ratio"],
         "Horizontal position of the back neckline low point.",
         "config/drafting.yaml constants.back_neck_dip_x_ratio.")

    # -- shoulder ------------------------------------------------------------
    rule("der.shoulder_slope_deg", "der.shoulder_slope_deg", ["shoulder_slope_deg_input"],
         lambda v: min(max(v["shoulder_slope_deg_input"], ranges["shoulder_slope_deg"][0]),
                       ranges["shoulder_slope_deg"][1]),
         "Shoulder slope clamped into the drafting window; the original value is reported in the trace.",
         "config/drafting.yaml ranges.shoulder_slope_deg (clamped).",
         {"shoulder_slope_deg": (ranges["shoulder_slope_deg"][0], ranges["shoulder_slope_deg"][1])})
    rule("der.shoulder_seam_length", "der.shoulder_seam_length", ["shoulder_w"],
         lambda v: v["shoulder_w"] - c["shoulder_neck_offset_mm"],
         "Drafted shoulder seam length = half shoulder width - shoulder_neck_offset_mm.",
         "shoulder_w is measured over the shoulder from the neck point; the seam starts at the neck-shoulder point.",
         {"der.shoulder_seam_length": (30.0, 300.0)})
    rule("der.shoulder_tip_dx", "der.shoulder_tip_dx",
         ["der.shoulder_seam_length", "der.shoulder_slope_deg"],
         lambda v: v["der.shoulder_seam_length"] * cos(radians(v["der.shoulder_slope_deg"])),
         "Horizontal run of the shoulder seam.", "Projected shoulder seam length.")
    rule("der.shoulder_tip_dy", "der.shoulder_tip_dy",
         ["der.shoulder_seam_length", "der.shoulder_slope_deg"],
         lambda v: v["der.shoulder_seam_length"] * sin(radians(v["der.shoulder_slope_deg"])),
         "Vertical drop of the shoulder seam.", "Projected shoulder seam length.")

    # -- armscye / bust shaping ---------------------------------------------
    rule("der.armscye_depth_mm", "der.armscye_depth_mm", ["armscye_depth_input"],
         lambda v: v["armscye_depth_input"],
         "Armscye depth used by the draft (measurement or explicit override).",
         "config/measurements.yaml armscye_depth: drop from the shoulder-tip level to the underarm.")
    rule("der.bust_projection_mm", "der.bust_projection_mm", ["bust_points"],
         lambda v: v["bust_points"],
         "Bust projection from the chest wall to the apex.", "config/measurements.yaml bust_points.")
    rule("der.bust_apex_x", "der.bust_apex_x",
         ["der.front_bust_half_width", "der.bust_projection_mm"],
         lambda v: v["der.front_bust_half_width"] - c["bust_apex_inset_ratio"] * v["der.bust_projection_mm"],
         "Bust apex x = front bust half width - bust_apex_inset_ratio * projection.",
         "config/drafting.yaml constants.bust_apex_inset_ratio.")
    rule("der.bust_apex_y", "der.bust_apex_y", ["bust_apex_y_input"],
         lambda v: v["bust_apex_y_input"],
         "Bust apex level used by the draft (vert_bust_line or explicit override).",
         "config/measurements.yaml vert_bust_line: drop from the neck point to the bust apex.")
    rule("der.dart_intake", "der.dart_intake",
         ["dart_intake_input", "der.bust_projection_mm"],
         lambda v: v["dart_intake_input"]
         if v["dart_intake_input"] > 0 else c["dart_intake_ratio"] * v["der.bust_projection_mm"],
         "Waist dart intake = dart_intake_ratio * bust projection, unless overridden.",
         "config/drafting.yaml constants.dart_intake_ratio.",
         {"der.dart_intake": (0.0, 200.0)})

    rules.extend(build_landmark_rules(c))
    return RuleRegistry(rules, external_inputs=EXTERNAL_INPUTS)


def registry_inputs(body: BodyMeasurements, ease, settings) -> dict[str, float]:
    """Measurement + ease values handed to the registry, all in millimetres."""
    values = body.rule_inputs()
    return {
        **values,
        "bust_level": values["bust_line"],
        "waist_level": values["waist_line"],
        "ease_bust": ease.bust_mm,
        "ease_waist": ease.waist_mm,
        "shoulder_slope_deg_input": (
            settings.shoulder_slope_deg if settings.shoulder_slope_deg is not None
            else values["shoulder_incl"]
        ),
        "armscye_depth_input": (
            settings.armscye_depth_mm if settings.armscye_depth_mm is not None
            else values["armscye_depth"]
        ),
        "bust_apex_y_input": (
            settings.bust_apex_y_mm if settings.bust_apex_y_mm is not None
            else values["vert_bust_line"]
        ),
        "dart_intake_input": settings.dart_intake_mm or 0.0,
    }


def input_warnings(body: BodyMeasurements, values: dict[str, float], resolved: dict[str, float]) -> list[MeasurementWarning]:
    """Report every place the draft had to interpret or clamp a measurement."""
    cfg = drafting_config()
    out: list[MeasurementWarning] = []
    lo, hi = cfg["ranges"]["shoulder_slope_deg"]
    if not lo <= values["shoulder_slope_deg_input"] <= hi:
        out.append(
            MeasurementWarning(
                "shoulder_slope_clamped",
                "shoulder_incl",
                f"shoulder_incl is {values['shoulder_slope_deg_input']:.1f} deg, outside the drafting "
                f"window {lo}-{hi} deg. Drafted at {resolved['der.shoulder_slope_deg']:.1f} deg; "
                f"override it in the garment settings to use the measured angle.",
                values["shoulder_slope_deg_input"],
                "clamped",
            )
        )
    ua_y = resolved["lm.front.underarm_y"]
    if ua_y < resolved["bust_level"]:
        out.append(
            MeasurementWarning(
                "armscye_above_bust",
                "armscye_depth",
                f"The underarm point lands at {ua_y:.0f} mm, above the bust line at "
                f"{resolved['bust_level']:.0f} mm. The bodice armhole will sit high; "
                f"increase armscye depth in the garment settings.",
                values["armscye_depth_input"],
            )
        )
    return out
