"""The predetermined seam network.

One preset, `basic_torso_v1`. Nothing here discovers a seam: the whole network is a
function of the ring grid and four landmark positions, so every seam path follows mesh
columns or a curve sampled on them, and every seam has a stable semantic label.

The network is expressed as one trim curve per panel: the y at which the panel's top
boundary cuts each column. Below that curve the panel runs down to the hem. Walking the
trim curve from centre front outwards crosses three semantic intervals, and the parameter
that separates them is the column's x, not its index, so the front and the back share one
piece of code:

    x <= neck_x          neckline     scooped below the neck reference level
    neck_x < x <= tip_x  shoulder     straight at the confirmed shoulder inclination
    tip_x < x            armscye      eased down to the underarm level

Centre front, centre back and the side are ring columns by construction, and the waist is
a ring. That is the whole seam network:

    centre front | centre back | side seam | neckline | armscye | shoulder | waist
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..assumptions import AssumptionRegistry
from ..config import reconstruction_config
from ..reconstruction.landmarks import key_columns, ring_x
from ..reconstruction.lofting import RingSurface

SEAM_ROLES = (
    "center_front", "center_back", "side_seam", "neckline", "armscye", "shoulder", "waist",
)


class SeamError(Exception):
    def __init__(self, errors: list[dict]) -> None:
        self.errors = errors
        super().__init__("; ".join(e["message"] for e in errors))


@dataclass
class SeamNetwork:
    preset: str
    columns: int
    side_column: int
    back_column: int
    neck_column: int
    tip_column: int
    neck_x_mm: float
    tip_x_mm: float
    top_half_width_mm: float
    underarm_y_mm: float
    waist_y_mm: float
    hem_y_mm: float
    shoulder_slope_deg: float
    front_neck_drop_mm: float
    back_neck_drop_mm: float
    neckline_power: float
    armscye_power: float
    ring_x: np.ndarray

    # ---- the trim curve ---------------------------------------------------

    def neck_drop(self, panel: str) -> float:
        return self.front_neck_drop_mm if panel == "front" else self.back_neck_drop_mm

    def shoulder_tip_y_mm(self) -> float:
        return (self.tip_x_mm - self.neck_x_mm) * math.tan(math.radians(self.shoulder_slope_deg))

    def trim_y(self, panel: str, column: int) -> float:
        """y where the panel's top boundary crosses this column."""
        x = abs(float(self.ring_x[column % self.columns]))
        if x <= self.neck_x_mm:
            u = 0.0 if self.neck_x_mm <= 0 else x / self.neck_x_mm
            return self.neck_drop(panel) * max(0.0, 1.0 - u * u) ** (0.5 * self.neckline_power)
        if x <= self.tip_x_mm:
            return (x - self.neck_x_mm) * math.tan(math.radians(self.shoulder_slope_deg))
        tip_y = self.shoulder_tip_y_mm()
        span = self.top_half_width_mm - self.tip_x_mm
        v = 1.0 if span <= 0 else (x - self.tip_x_mm) / span
        shape = math.sin(0.5 * math.pi * min(1.0, v)) ** self.armscye_power
        return tip_y + (self.underarm_y_mm - tip_y) * shape

    def role_at(self, column: int) -> str:
        """Which semantic interval of the trim curve this column belongs to."""
        x = abs(float(self.ring_x[column % self.columns]))
        if x <= self.neck_x_mm:
            return "neckline"
        if x <= self.tip_x_mm:
            return "shoulder"
        return "armscye"

    def as_dict(self) -> dict:
        return {
            "preset": self.preset,
            "columns": self.columns,
            "keyColumns": {
                "centerFront": 0, "neckPoint": self.neck_column,
                "shoulderTip": self.tip_column, "side": self.side_column,
                "centerBack": self.back_column,
            },
            "neckHalfWidthMm": round(self.neck_x_mm, 4),
            "shoulderTipHalfWidthMm": round(self.tip_x_mm, 4),
            "shoulderTipDropMm": round(self.shoulder_tip_y_mm(), 4),
            "shoulderSlopeDeg": self.shoulder_slope_deg,
            "underarmYMm": round(self.underarm_y_mm, 4),
            "waistYMm": round(self.waist_y_mm, 4),
            "hemYMm": round(self.hem_y_mm, 4),
            "frontNeckDropMm": self.front_neck_drop_mm,
            "backNeckDropMm": self.back_neck_drop_mm,
            "roles": list(SEAM_ROLES),
            "trim": [
                {"column": j, "role": self.role_at(j),
                 "xMm": round(float(self.ring_x[j]), 4),
                 "frontYMm": round(self.trim_y("front", j), 4),
                 "backYMm": round(self.trim_y("back", j), 4)}
                for j in range(self.back_column + 1)
            ],
        }


def build_seam_network(
    surface: RingSurface,
    sections: dict,
    levels: dict,
    m,
    settings,
    underarm_drop_mm: float,
    registry: AssumptionRegistry,
) -> SeamNetwork:
    cfg = reconstruction_config()["seams"]
    if settings.seam_preset not in ("basic_torso_v1", "", None):
        raise SeamError([_err("seam_preset", "UNKNOWN_SEAM_PRESET",
                              f"Seam preset '{settings.seam_preset}' does not exist. The "
                              f"first release ships basic_torso_v1 only.")])

    keys = key_columns(surface.columns)
    xs = ring_x(float(surface.a[0]), float(surface.b[0]), surface.exponent, surface.columns)
    top_half_width = float(abs(xs[keys["right_side"]]))

    neck_x = sections["neck"].a_mm
    tip_x = m.shoulder_width_full / 2.0

    errors: list[dict] = []
    if neck_x <= 0:
        errors.append(_err("neck_w", "NECK_SECTION_DEGENERATE",
                           "The reconstructed neck section has no width."))
    if tip_x <= neck_x:
        errors.append(_err(
            "shoulder_w", "SHOULDER_TIP_INSIDE_NECK",
            f"The shoulder tip lands {tip_x:.1f} mm from centre, inside the "
            f"{neck_x:.1f} mm neck half width. Confirm which interpretation shoulder_w "
            f"and neck_w carry.",
        ))
    if tip_x >= top_half_width:
        errors.append(_err(
            "shoulder_w", "SHOULDER_TIP_OUTSIDE_SECTION",
            f"The shoulder tip lands {tip_x:.1f} mm from centre but the reconstructed "
            f"upper-chest section is only {top_half_width:.1f} mm wide at the side. The "
            f"across-shoulder measurement and the back width cannot both be right.",
        ))
    if errors:
        raise SeamError(errors)

    underarm_y = levels["underarm"].y_mm + underarm_drop_mm
    waist_y = levels["waist"].y_mm
    hem_drop = float(cfg["hem_drop_below_waist_mm"])
    hem_y = waist_y + hem_drop

    # The neck point and the shoulder tip snap to ring columns, and the trim curve then
    # changes regime at the snapped positions rather than at the measured ones. Otherwise
    # the neck point would sit a few millimetres off the neck reference level and the
    # seam corner would not land on a mesh vertex. The snapping residual is reported.
    neck_column = _nearest_column(xs, keys, neck_x)
    tip_column = _nearest_column(xs, keys, tip_x)
    snapped_neck_x = float(abs(xs[neck_column]))
    snapped_tip_x = float(abs(xs[tip_column]))

    network = SeamNetwork(
        preset="basic_torso_v1",
        columns=surface.columns,
        side_column=keys["right_side"],
        back_column=keys["center_back"],
        neck_column=neck_column,
        tip_column=tip_column,
        neck_x_mm=snapped_neck_x,
        tip_x_mm=snapped_tip_x,
        top_half_width_mm=top_half_width,
        underarm_y_mm=underarm_y,
        waist_y_mm=waist_y,
        hem_y_mm=hem_y,
        shoulder_slope_deg=m.shoulder_slope_degrees,
        front_neck_drop_mm=float(settings.front_neck_drop_mm
                                 if settings.front_neck_drop_mm is not None
                                 else cfg["front_neck_drop_mm"]),
        back_neck_drop_mm=float(settings.back_neck_drop_mm
                                if settings.back_neck_drop_mm is not None
                                else cfg["back_neck_drop_mm"]),
        neckline_power=float(cfg["neckline_power"]),
        armscye_power=float(cfg["armscye_power"]),
        ring_x=xs,
    )

    if not 0 < network.neck_column < network.tip_column < network.side_column:
        raise SeamError([_err(
            "cross_section_resolution", "SEAM_COLUMNS_COLLIDE",
            f"The neck point and the shoulder tip snap to ring columns "
            f"{network.neck_column} and {network.tip_column} out of "
            f"{network.side_column} per quarter, so a seam interval would be empty. "
            f"Raise the cross-section resolution.",
        )])

    tip_y = network.shoulder_tip_y_mm()
    if not tip_y < underarm_y:
        raise SeamError([_err(
            "armscye_depth", "ARMSCYE_ABOVE_SHOULDER_TIP",
            f"The underarm level ({underarm_y:.1f} mm) is not below the shoulder tip "
            f"({tip_y:.1f} mm), so no armhole can be cut.",
        )])
    if not underarm_y < waist_y:
        raise SeamError([_err(
            "armscye_depth", "ARMSCYE_BELOW_WAIST",
            f"The underarm level ({underarm_y:.1f} mm) is not above the waist "
            f"({waist_y:.1f} mm).",
        )])
    if not waist_y < hem_y:
        raise SeamError([_err(
            "hem_drop_below_waist_mm", "HEM_ABOVE_WAIST",
            f"The dress hem ({hem_y:.1f} mm) is not below the waist ({waist_y:.1f} mm). "
            f"Check dress_hem_drop_below_waist_mm.",
        )])

    registry.add(
        "seam.neckline", "Neckline depth", {
            "frontMm": network.front_neck_drop_mm, "backMm": network.back_neck_drop_mm,
        }, "mm",
        """body_measurements.yaml carries no neckline, so the front and back neck drops
        are chosen, not derived. They are the two numbers most likely to be changed by
        hand. The neckline is a quarter-ellipse between centre front and the neck point.""",
        [], True,
    )
    registry.add(
        "seam.column_snapping", "Seam corners snapped to ring columns",
        {"neckPointMm": round(snapped_neck_x - neck_x, 3),
         "shoulderTipMm": round(snapped_tip_x - tip_x, 3)}, "mm",
        """The neck point and the shoulder tip are moved to the nearest ring column so
        every seam corner lands on a mesh vertex. The figures are how far each moved;
        raising the cross-section resolution shrinks them.""",
        ["neck_w", "shoulder_w"], True,
    )
    registry.add(
        "seam.shoulder_tip", "Shoulder tip placement",
        round(snapped_tip_x, 2), "mm",
        """The shoulder tip is placed at half the confirmed across-shoulder breadth,
        measured horizontally from centre on the upper-chest section. The measurement is
        taken over the shoulder, so its horizontal projection is an interpretation; it
        avoids introducing a shoulder-length offset constant.""",
        ["shoulder_w", "shoulder_incl"], False,
    )
    registry.add(
        "seam.armscye_shape", "Armscye and neckline curve shapes",
        {"necklinePower": network.neckline_power, "armscyePower": network.armscye_power},
        None,
        """The two cut curves are a quarter-ellipse neckline and a sine-eased armscye. The
        underarm level and the shoulder tip are measured; the curve between them is
        drawn. The armscye meets the side seam tangentially so the two panels join at a
        right angle.""",
        ["armscye_depth"], True,
    )
    if underarm_drop_mm:
        registry.add(
            "seam.armhole_ease_drop", "Armhole ease as an underarm drop",
            round(underarm_drop_mm, 2), "mm",
            """Armhole ease is requested as a circumference but the geometry needs a
            level, so a fraction of it lowers the underarm. The fraction is chosen.""",
            [], True,
        )
    registry.add(
        "seam.dress_hem", "Dress hem drop below the waist", hem_drop, "mm",
        """body_measurements.yaml carries no garment length or hem level. The dress panel
        is extended below the waist by this configured drop. It is a product setting,
        not a body measurement.""",
        ["waist_line"], True,
    )
    return network


def _nearest_column(xs: np.ndarray, keys: dict, target_x: float) -> int:
    quarter = np.abs(xs[: keys["right_side"] + 1])
    return int(np.argmin(np.abs(quarter - target_x)))


def _err(field: str, code: str, message: str) -> dict:
    return {
        "field": field, "code": code, "message": " ".join(message.split()),
        "suggestedAction": "Confirm the shoulder, neck and armscye interpretations.",
        "severity": "blocking",
    }
