"""Derived drafting dimensions and the assembled rule registry.

Inputs are drafting variables from the semantics adapter (DraftingMeasurements), never
camera subsystem field names. Only girth-derived widths, angle handling and the
shoulder/neckline runs live here; panel geometry is built from the landmark values in
drafting/panels.py.
"""
from __future__ import annotations

from math import cos, radians, sin

from ..config import drafting_config
from .landmarks import build_landmark_rules
from .rules import DraftingRule, RuleRegistry
from .semantics import DraftingMeasurements, MeasurementWarning

# Drafting variables supplied by the adapter. Angles arrive in degrees, every length in
# millimetres; nothing else may enter the registry.
EXTERNAL_INPUTS = (
    # from DraftingMeasurements
    "bust_circumference", "waist_circumference", "hip_circumference", "underbust_circumference",
    "shoulder_width_full", "back_width_full", "waist_back_width_full", "hip_back_width_full",
    "neck_width_half", "neck_to_bust_level", "neck_to_waist_level",
    "neck_to_bust_apex_vertical", "bust_to_waist_vertical", "armscye_depth_vertical",
    "bust_apex_spacing_full", "shoulder_slope_degrees",
    # resolved ease and explicit operator overrides
    "ease_bust", "ease_waist",
    "shoulder_slope_degrees_override", "armscye_depth_override",
    "bust_apex_level_override", "dart_intake_override",
)


def build_registry(config: dict | None = None) -> RuleRegistry:
    cfg = config or drafting_config()
    c = cfg["constants"]
    ranges = cfg["ranges"]

    rules: list[DraftingRule] = []

    def rule(rule_id, output_name, inputs, equation, explanation, source_note, valid_range=None):
        rules.append(
            DraftingRule(
                id=rule_id, name=output_name.replace("der.", "Derived: "), output_name=output_name,
                required_inputs=inputs, equation=equation, explanation=explanation,
                source_note=source_note, valid_range=valid_range or {}, group="derived",
            )
        )

    # -- ease ----------------------------------------------------------------
    rule("der.bust_with_ease", "der.bust_with_ease", ["bust_circumference", "ease_bust"],
         lambda v: v["bust_circumference"] + v["ease_bust"],
         "Bust girth plus the selected bust ease.",
         "Ease preset from config/drafting.yaml ease_presets.")
    rule("der.waist_with_ease", "der.waist_with_ease", ["waist_circumference", "ease_waist"],
         lambda v: v["waist_circumference"] + v["ease_waist"],
         "Waist girth plus the selected waist ease.",
         "Ease preset from config/drafting.yaml ease_presets.")

    # -- front / back width split at the side seam ---------------------------
    rule("der.back_bust_half_width", "der.back_bust_half_width", ["back_width_full"],
         lambda v: v["back_width_full"] / 2.0,
         "Back half-pattern width at the bust line = back_width_full / 2.",
         "Semantics adapter variable back_width_full (canonical full-body representation).",
         {"back_width_full": (150.0, 900.0)})
    rule("der.front_bust_half_width", "der.front_bust_half_width",
         ["der.bust_with_ease", "back_width_full"],
         lambda v: (v["der.bust_with_ease"] - v["back_width_full"]) / 2.0,
         "Front half-pattern width at the bust line = (bust + ease - back_width_full) / 2.",
         "Side seam placed where the front and back widths close on the total girth.",
         {"back_width_full": (150.0, 900.0)})
    rule("der.back_waist_half_width", "der.back_waist_half_width", ["waist_back_width_full"],
         lambda v: v["waist_back_width_full"] / 2.0,
         "Back half-pattern width at the waist line = waist_back_width_full / 2.",
         "Semantics adapter variable waist_back_width_full.")
    rule("der.front_waist_half_width", "der.front_waist_half_width",
         ["der.waist_with_ease", "waist_back_width_full"],
         lambda v: (v["der.waist_with_ease"] - v["waist_back_width_full"]) / 2.0,
         "Front half-pattern width at the waist line = (waist + ease - waist_back_width_full) / 2.",
         "Same side-seam split rule as the bust line.")

    # -- neck ----------------------------------------------------------------
    rule("der.neck_shoulder_x", "der.neck_shoulder_x", ["neck_width_half"],
         lambda v: v["neck_width_half"] * c["neck_shoulder_share"],
         "Horizontal distance centre neck point to neck-shoulder point = neck_width_half * neck_shoulder_share.",
         "Semantics adapter variable neck_width_half (half neck girth).")
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
    rule("der.shoulder_slope_deg", "der.shoulder_slope_deg",
         ["shoulder_slope_degrees_override", "shoulder_slope_degrees"],
         lambda v: v["shoulder_slope_degrees_override"] if v["shoulder_slope_degrees_override"] > 0
         else min(max(v["shoulder_slope_degrees"], ranges["shoulder_slope_deg"][0]),
                  ranges["shoulder_slope_deg"][1]),
         "Shoulder slope: operator override if set, otherwise the confirmed slope clamped "
         "into the drafting window.",
         "config/drafting.yaml ranges.shoulder_slope_deg (clamped); the original confirmed "
         "value is reported in the trace.",
         {"der.shoulder_slope_deg": (ranges["shoulder_slope_deg"][0], ranges["shoulder_slope_deg"][1])})
    rule("der.shoulder_half_width", "der.shoulder_half_width", ["shoulder_width_full"],
         lambda v: v["shoulder_width_full"] / 2.0,
         "Half shoulder width = shoulder_width_full / 2 (neck reference point to one shoulder tip).",
         "Semantics adapter variable shoulder_width_full.")
    rule("der.shoulder_seam_length", "der.shoulder_seam_length", ["der.shoulder_half_width"],
         lambda v: v["der.shoulder_half_width"] - c["shoulder_neck_offset_mm"],
         "Drafted shoulder seam length = half shoulder width - shoulder_neck_offset_mm.",
         "config/drafting.yaml constants.shoulder_neck_offset_mm.",
         {"der.shoulder_seam_length": (30.0, 400.0)})
    rule("der.shoulder_tip_dx", "der.shoulder_tip_dx",
         ["der.shoulder_seam_length", "der.shoulder_slope_deg"],
         lambda v: v["der.shoulder_seam_length"] * cos(radians(v["der.shoulder_slope_deg"])),
         "Horizontal run of the shoulder seam.", "Projected shoulder seam length.")
    rule("der.shoulder_tip_dy", "der.shoulder_tip_dy",
         ["der.shoulder_seam_length", "der.shoulder_slope_deg"],
         lambda v: v["der.shoulder_seam_length"] * sin(radians(v["der.shoulder_slope_deg"])),
         "Vertical drop of the shoulder seam.", "Projected shoulder seam length.")

    # -- armscye / bust shaping ---------------------------------------------
    rule("der.armscye_depth_mm", "der.armscye_depth_mm",
         ["armscye_depth_override", "armscye_depth_vertical"],
         lambda v: v["armscye_depth_override"] if v["armscye_depth_override"] > 0
         else v["armscye_depth_vertical"],
         "Armscye depth used by the draft (confirmed semantics value or operator override).",
         "Semantics adapter variable armscye_depth_vertical.")
    rule("der.bust_apex_spacing_mm", "der.bust_apex_spacing_mm", ["bust_apex_spacing_full"],
         lambda v: v["bust_apex_spacing_full"],
         "Bust apex spacing in canonical full representation.",
         "Semantics adapter variable bust_apex_spacing_full.")
    rule("der.bust_apex_x", "der.bust_apex_x",
         ["der.front_bust_half_width", "der.bust_apex_spacing_mm"],
         lambda v: v["der.front_bust_half_width"] - c["bust_apex_inset_ratio"] * v["der.bust_apex_spacing_mm"],
         "Bust apex x = front bust half width - bust_apex_inset_ratio * apex spacing.",
         "config/drafting.yaml constants.bust_apex_inset_ratio.")
    rule("der.bust_apex_y", "der.bust_apex_y",
         ["bust_apex_level_override", "neck_to_bust_apex_vertical"],
         lambda v: v["bust_apex_level_override"] if v["bust_apex_level_override"] > 0
         else v["neck_to_bust_apex_vertical"],
         "Bust apex level used by the draft (confirmed relative vertical distance or override).",
         "Semantics adapter variable neck_to_bust_apex_vertical.")
    rule("der.dart_intake", "der.dart_intake",
         ["dart_intake_override", "der.bust_apex_spacing_mm"],
         lambda v: v["dart_intake_override"] if v["dart_intake_override"] > 0
         else c["dart_intake_ratio"] * v["der.bust_apex_spacing_mm"],
         "Waist dart intake = dart_intake_ratio * apex spacing, unless overridden.",
         "config/drafting.yaml constants.dart_intake_ratio.",
         {"der.dart_intake": (0.0, 200.0)})

    rules.extend(build_landmark_rules(c))
    return RuleRegistry(rules, external_inputs=EXTERNAL_INPUTS)


def registry_inputs(measurements: DraftingMeasurements, ease, settings) -> dict[str, float]:
    """Drafting variables + resolved ease + overrides, all in millimetres/degrees."""
    data = measurements.model_dump()
    return {
        **{k: v for k, v in data.items() if isinstance(v, (int, float))},
        "ease_bust": ease.bust_mm,
        "ease_waist": ease.waist_mm,
        "shoulder_slope_degrees_override": settings.shoulder_slope_deg or 0.0,
        "armscye_depth_override": settings.armscye_depth_mm or 0.0,
        "bust_apex_level_override": settings.bust_apex_y_mm or 0.0,
        "dart_intake_override": settings.dart_intake_mm or 0.0,
    }


def input_warnings(measurements: DraftingMeasurements, resolved: dict[str, float]) -> list[MeasurementWarning]:
    """Report every place the draft had to interpret or clamp a confirmed measurement."""
    cfg = drafting_config()
    out: list[MeasurementWarning] = []
    lo, hi = cfg["ranges"]["shoulder_slope_deg"]
    if not lo <= measurements.shoulder_slope_degrees <= hi:
        out.append(MeasurementWarning(
            "shoulder_slope_clamped", "shoulder_incl",
            f"The confirmed shoulder slope is {measurements.shoulder_slope_degrees:.1f} deg, outside "
            f"the drafting window {lo}-{hi} deg. Drafted at {resolved['der.shoulder_slope_deg']:.1f} deg; "
            f"set the shoulder slope override in the garment settings to use the confirmed angle.",
            measurements.shoulder_slope_degrees, "clamped",
        ))
    ua_y = resolved["lm.front.underarm_y"]
    if ua_y < resolved["neck_to_bust_level"]:
        out.append(MeasurementWarning(
            "armscye_above_bust", "armscye_depth",
            f"The underarm point lands at {ua_y:.0f} mm, above the bust line at "
            f"{resolved['neck_to_bust_level']:.0f} mm. The bodice armhole will sit high; confirm a "
            f"different armscye_depth interpretation or set an override.",
            measurements.armscye_depth_vertical,
        ))
    return out
