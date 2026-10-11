"""Construction landmarks.

Coordinate system (per panel, local):
    origin  = the centre neck point of that panel (front: centre front, back: centre back)
    +x      = toward the side seam
    +y      = downward, so the waist line sits below the neck point
    unit    = mm

Every landmark is produced by a rule so it carries a rule id and its dependencies.
"""
from __future__ import annotations

from math import cos, hypot, sin

from .rules import DraftingRule, landmark_rule

SIDES = ("front", "back")


def build_landmark_rules(constants: dict) -> list[DraftingRule]:
    c = constants
    rules: list[DraftingRule] = []

    def lm(rule_id: str, fn, explanation: str, source: str | None = None) -> None:
        rules.append(landmark_rule(rule_id, rule_id, fn, explanation, source or "config/drafting.yaml constants"))

    # ---- shared frame -------------------------------------------------------
    lm("lm.origin_x", lambda v: 0.0, "Origin of the local frame: x = 0 on the centre line.")
    lm("lm.origin_y", lambda v: 0.0, "Origin of the local frame: y = 0 at the neck point level.")
    lm("lm.shoulder_dir_x", lambda v: cos(v["der.shoulder_slope_deg"] * 3.141592653589793 / 180.0),
       "Unit vector of the shoulder seam, x component.")
    lm("lm.shoulder_dir_y", lambda v: sin(v["der.shoulder_slope_deg"] * 3.141592653589793 / 180.0),
       "Unit vector of the shoulder seam, y component.")

    for s in SIDES:
        rise = -c["neck_rise_mm"]
        bust_half = "der.front_bust_half_width" if s == "front" else "der.back_bust_half_width"
        waist_half = "der.front_waist_half_width" if s == "front" else "der.back_waist_half_width"
        depth = "der.front_neck_depth" if s == "front" else "der.back_neck_depth"
        dip_x = "der.front_neck_dip_x" if s == "front" else "der.back_neck_dip_x"

        lm(f"lm.{s}.cf_neck_x", lambda v: 0.0, f"Centre {s} neck point sits on the centre line.")
        lm(f"lm.{s}.cf_neck_y", lambda v: 0.0, f"Centre {s} neck point is the frame origin.")
        lm(f"lm.{s}.cf_waist_x", lambda v: 0.0, f"Centre {s} waist point sits on the centre line.")
        lm(f"lm.{s}.cf_waist_y", lambda v: v["waist_level"], f"Centre {s} waist point is on the waist line.")

        lm(f"lm.{s}.neck_shoulder_x", lambda v: v["der.neck_shoulder_x"],
           "Neck-shoulder point x = neck_w * constants.neck_shoulder_share.")
        lm(f"lm.{s}.neck_shoulder_y", lambda v, r=rise: r,
           "Neck-shoulder point sits above the neck point by constants.neck_rise_mm; the same "
           "value on both panels keeps front and back side seams the same length.")
        lm(f"lm.{s}.shoulder_tip_x",
           lambda v, s=s: v[f"lm.{s}.neck_shoulder_x"] + v["der.shoulder_tip_dx"],
           "Shoulder tip x = neck-shoulder x + shoulder seam horizontal run.")
        lm(f"lm.{s}.shoulder_tip_y",
           lambda v, s=s: v[f"lm.{s}.neck_shoulder_y"] + v["der.shoulder_tip_dy"],
           "Shoulder tip y = neck-shoulder y + shoulder seam drop.")

        lm(f"lm.{s}.underarm_x", lambda v, b=bust_half: v[b],
           f"Underarm x = the {s} half-pattern width at the bust line.")
        lm(f"lm.{s}.underarm_y",
           lambda v, s=s: v[f"lm.{s}.shoulder_tip_y"] + v["der.armscye_depth_mm"],
           "Underarm y = shoulder tip y + armscye depth.")
        lm(f"lm.{s}.waist_side_x", lambda v, b=waist_half: v[b],
           f"Side seam at the waist = the {s} half-pattern width at the waist line.")
        lm(f"lm.{s}.waist_side_y", lambda v: v["waist_level"],
           "Side seam at the waist is on the waist line.")

        lm(f"lm.{s}.neck_dip_x", lambda v, d=dip_x: v[d],
           "Horizontal position of the neckline low point.")
        lm(f"lm.{s}.neck_dip_y",
           lambda v, s=s, d=depth: v[f"lm.{s}.neck_shoulder_y"] + v[d],
           "Neckline low point y = neck-shoulder y + neck depth.")

        lm(f"lm.{s}.armhole_chord",
           lambda v, s=s: hypot(v[f"lm.{s}.underarm_x"] - v[f"lm.{s}.shoulder_tip_x"],
                                v[f"lm.{s}.underarm_y"] - v[f"lm.{s}.shoulder_tip_y"]),
           "Chord length of the armscye.")
        lm(f"lm.{s}.armhole_handle_shoulder",
           lambda v, s=s: c["armscye_handle_ratio"] * v[f"lm.{s}.armhole_chord"],
           "First armscye handle length, along the shoulder seam direction.")
        lm(f"lm.{s}.side_seam_length",
           lambda v, s=s: hypot(v[f"lm.{s}.waist_side_x"] - v[f"lm.{s}.underarm_x"],
                                v[f"lm.{s}.waist_side_y"] - v[f"lm.{s}.underarm_y"]),
           "Side seam length, underarm point to waist point.")
        lm(f"lm.{s}.armhole_handle_side",
           lambda v, s=s: 0.45 * v[f"lm.{s}.armhole_chord"],
           "Second armscye handle length, along the side seam direction.")
        lm(f"lm.{s}.armhole_c1_x",
           lambda v, s=s: v[f"lm.{s}.shoulder_tip_x"]
           + v[f"lm.{s}.armhole_handle_shoulder"] * v["lm.shoulder_dir_x"],
           "Armscye control 1: shoulder tip plus a handle along the shoulder seam direction.")
        lm(f"lm.{s}.armhole_c1_y",
           lambda v, s=s: v[f"lm.{s}.shoulder_tip_y"]
           + v[f"lm.{s}.armhole_handle_shoulder"] * v["lm.shoulder_dir_y"],
           "Armscye control 1 y.")
        lm(f"lm.{s}.armhole_c2_x",
           lambda v, s=s: v[f"lm.{s}.underarm_x"] - v[f"lm.{s}.armhole_handle_side"]
           * (v[f"lm.{s}.waist_side_x"] - v[f"lm.{s}.underarm_x"]) / max(v[f"lm.{s}.side_seam_length"], 1e-9),
           "Armscye control 2: underarm plus a handle back along the side seam direction.")
        lm(f"lm.{s}.armhole_c2_y",
           lambda v, s=s: v[f"lm.{s}.underarm_y"] - v[f"lm.{s}.armhole_handle_side"]
           * (v[f"lm.{s}.waist_side_y"] - v[f"lm.{s}.underarm_y"]) / max(v[f"lm.{s}.side_seam_length"], 1e-9),
           "Armscye control 2 y.")

        lm(f"lm.{s}.neckline_handle",
           lambda v, s=s: c["neckline_handle_ratio"] * abs(v[f"lm.{s}.neck_shoulder_x"] - v[f"lm.{s}.cf_neck_x"]),
           "Neckline handle length along the neck line.")
        lm(f"lm.{s}.neckline_c1_x",
           lambda v, s=s: v[f"lm.{s}.cf_neck_x"] + v[f"lm.{s}.neckline_handle"],
           "Neckline control 1 x.")
        lm(f"lm.{s}.neckline_c1_y",
           lambda v, s=s, d=depth: v[f"lm.{s}.cf_neck_y"] + v[d],
           "Neckline control 1 pulls the curve down into the neck depth.")
        lm(f"lm.{s}.neckline_c2_x",
           lambda v, s=s: v[f"lm.{s}.neck_shoulder_x"] - v[f"lm.{s}.neckline_handle"],
           "Neckline control 2 x.")
        lm(f"lm.{s}.neckline_c2_y",
           lambda v, s=s, d=depth: v[f"lm.{s}.neck_shoulder_y"] + v[d],
           "Neckline control 2 y.")

    # ---- front bust shaping / dart ----------------------------------------
    lm("lm.front.dart_top_y", lambda v: v["bust_level"] + c["dart_top_offset_mm"],
       "Upper dart leg sits on the bust line plus constants.dart_top_offset_mm.")
    lm("lm.front.dart_bottom_y", lambda v: v["waist_level"],
       "Lower dart leg sits on the waist line.")
    lm("lm.front.bust_apex_y",
       lambda v: min(max(v["der.bust_apex_y"], v["lm.front.dart_top_y"] + c["dart_min_height_mm"]),
                     v["lm.front.dart_bottom_y"] - c["dart_min_height_mm"]),
       "Bust apex y, held between the two dart leg levels with a minimum dart height.")
    lm("lm.front.bust_apex_x", lambda v: v["der.bust_apex_x"],
       "Bust apex x from the bust projection rule.")
    lm("lm.front.dart_half_intake", lambda v: v["der.dart_intake"] / 2.0,
       "Each dart leg is offset from the apex by half the total dart intake.")
    return rules
