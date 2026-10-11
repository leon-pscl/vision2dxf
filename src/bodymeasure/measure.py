"""Turn landmarks plus silhouettes into the 26 GarmentCode body parameters (brief 5.7).

The input is ``landmarks.json`` and the view masks, so the review UI can edit a
landmark and call this again. Nothing here searches for landmarks: that is
``landmarks.py``. Internal unit is millimetres; output is centimetres to one
decimal, and degrees for the three angles.

Every view carries its own :class:`~bodymeasure.landmarks.ViewScale`, set from the
subject's measured height over that view's head-to-floor pixel span. Front and
side rows are therefore never compared as pixel rows: everything crossing between
views goes through the height above the floor, in mm (brief 5.4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from bodymeasure.geometry import (
    back_arc,
    contour_length,
    hull_length,
    run_containing,
    runs_in_row,
    two_semi_ellipse_girth,
    width_at,
)
from bodymeasure.landmarks import (
    THIGH_DEPTH_MM,
    Landmark,
    ViewScale,
    as_dict,
    balance_x_at,
    view_scale,
)
from bodymeasure.posegate import L_ELBOW, L_SHOULDER, R_ELBOW, R_SHOULDER

GIRTH_DISAGREEMENT_MM = 10.0  # two side views may differ by this much before a warning
BUST_POINTS_RATIO = 0.170  # GarmentCode mean_all.yaml: 16.95 / 99.84
BUM_POINTS_RATIO = 0.176  # GarmentCode mean_all.yaml: 18.23 / 103.48


@dataclass(frozen=True)
class Measurement:
    """One value with the provenance that goes into ``measurement_report.yaml``."""

    value_cm: float
    source: str
    confidence: str
    views: list[str] = field(default_factory=list)
    edited: bool = False
    note: str = ""


@dataclass(frozen=True)
class MeasurementSet:
    """Every measurement, plus the warnings the tailor should read."""

    values: dict[str, Measurement]
    warnings: list[str] = field(default_factory=list)

    def body(self) -> dict[str, float]:
        """Return the flat ``body:`` mapping for ``body_measurements.yaml``."""
        return {name: entry.value_cm for name, entry in sorted(self.values.items())}


@dataclass
class SideView:
    """One side view: its mask, its landmarks and the scale they imply."""

    name: str
    mask: np.ndarray
    landmarks: dict[str, Landmark]
    height_cm: float

    def __post_init__(self) -> None:
        """Work out this view's scale from the rows it was given."""
        self.scale = view_scale(list(self.landmarks.values()), self.height_cm)  # type: ignore[attr-defined]

    def balance_x(self, row: float) -> float:
        """Return the x of the balance line at ``row``."""
        return balance_x_at(row, self.landmarks["balance_top"], self.landmarks["balance_bottom"])

    def depths(self, row: int) -> tuple[float, float]:
        """Return the (front, back) depth in mm at ``row``."""
        centre = int(round(self.balance_x(row)))
        run = run_containing(self.mask, row, centre)
        if run is None:
            return 0.0, 0.0
        return (
            (run[1] - centre + 1) * self.scale.mm_per_px,
            (centre - run[0]) * self.scale.mm_per_px,
        )

    def edge(self, row: int, side: str) -> float | None:
        """Return the x of the front or back edge at ``row``, or ``None`` off the body."""
        centre = int(round(self.balance_x(row)))
        run = run_containing(self.mask, row, centre)
        if run is None:
            return None
        return float(run[1] if side == "front" else run[0])


def _cm(mm: float) -> float:
    """Round millimetres to centimetres, one decimal, as the contract requires."""
    return round(mm / 10.0, 1)


def _distance_mm(a: Landmark, b: Landmark, scale: ViewScale) -> float:
    return math.hypot((a.x - b.x) * scale.mm_per_px, (a.y - b.y) * scale.mm_per_px)


def _mean_depths(sides: list[SideView], h_mm: float) -> tuple[float, float, list[float]]:
    """Return the mean front and back depth at a height, and each girth's input."""
    fronts, backs = [], []
    for side in sides:
        front, back = side.depths(side.scale.row_of(h_mm))
        fronts.append(front)
        backs.append(back)
    return float(np.mean(fronts)), float(np.mean(backs)), fronts


def _girth_at(
    h_mm: float, front_mask: np.ndarray, torso_x: int, sides: list[SideView], front: ViewScale
) -> tuple[float, float]:
    """Return the two-semi-ellipse girth and the back arc in mm at a height."""
    width = width_at(front_mask, front.row_of(h_mm), torso_x, front.mm_per_px)
    depth_front, depth_back, _ = _mean_depths(sides, h_mm)
    return two_semi_ellipse_girth(width, depth_front, depth_back), back_arc(width, depth_back)


def _girth_spread_mm(
    h_mm: float, front_mask: np.ndarray, torso_x: int, sides: list[SideView], front: ViewScale
) -> float:
    """Return how far the two side views disagree about a girth, in mm."""
    if len(sides) <= 1:
        return 0.0
    width = width_at(front_mask, front.row_of(h_mm), torso_x, front.mm_per_px)
    girths = []
    for side in sides:
        depth_front, depth_back = side.depths(side.scale.row_of(h_mm))
        girths.append(two_semi_ellipse_girth(width, depth_front, depth_back))
    return max(girths) - min(girths)


def _edge_path(side: SideView, top_row: int, bottom_row: int, which: str) -> np.ndarray:
    """Return the front or back edge between two rows as an ``(N, 2)`` pixel path."""
    points = []
    for row in range(int(top_row), int(bottom_row) + 1):
        x = side.edge(row, which)
        if x is not None:
            points.append((x, float(row)))
    return np.array(points, dtype=float) if points else np.zeros((0, 2))


def _leg_girth_mm(
    row: int,
    front_mask: np.ndarray,
    torso_x: int,
    sides: list[SideView],
    front: ViewScale,
) -> float:
    """Return one thigh's girth at ``row``: front run of one leg, side leg depth."""
    runs = [run for run in runs_in_row(front_mask, row) if run[1] < torso_x]
    if not runs:
        return 0.0
    leg = max(runs, key=lambda run: run[1] - run[0])
    width = (leg[1] - leg[0] + 1) * front.mm_per_px
    depth_front, depth_back, _ = _mean_depths(sides, front.height_mm(row))
    return two_semi_ellipse_girth(width, depth_front, depth_back)


def _angle_above_horizontal(a: Landmark, b: Landmark, scale: ViewScale) -> float:
    """Return the angle in degrees of the segment a->b above the horizontal."""
    dx = abs(a.x - b.x) * scale.mm_per_px
    dy = abs(a.y - b.y) * scale.mm_per_px
    return math.degrees(math.atan2(dy, dx)) if dx or dy else 0.0


def _angle_from_vertical(a: Landmark, b: Landmark, scale: ViewScale) -> float:
    """Return the angle in degrees of the segment a->b away from vertical."""
    dx = abs(a.x - b.x) * scale.mm_per_px
    dy = abs(a.y - b.y) * scale.mm_per_px
    return math.degrees(math.atan2(dx, dy)) if dx or dy else 0.0


def measure(
    front_mask: np.ndarray,
    front_landmarks: list[Landmark],
    sides: list[SideView],
    keypoints: np.ndarray,
    height_cm: float,
) -> MeasurementSet:
    """Compute the 26 GarmentCode keys from landmarks and silhouettes.

    Raises:
        ValueError: if no side view is given, or a landmark the engine needs is
            missing from ``front_landmarks``.

    """
    if not sides:
        raise ValueError("at least one side view is required")
    front = as_dict(front_landmarks)
    missing = {"waist_row", "armpit_row", "crotch_row"} - set(front)
    if missing:
        raise ValueError(f"missing front landmarks: {', '.join(sorted(missing))}")

    front_scale = view_scale(front_landmarks, height_cm)
    side_names = [side.name for side in sides]
    both = ["front", *side_names]
    torso_x = int(round(front["waist_row"].x))
    first = sides[0]

    # Heights above the floor, in mm, are what the two views share; each one is
    # converted back to a row in whichever view is being measured.
    h = {
        "armpit": front_scale.height_mm(front["armpit_row"].y),
        "waist": front_scale.height_mm(front["waist_row"].y),
        "crotch": front_scale.height_mm(front["crotch_row"].y),
        "bust": first.scale.height_mm(first.landmarks["bust_row"].y),
        "underbust": first.scale.height_mm(first.landmarks["underbust_row"].y),
        "hips": first.scale.height_mm(first.landmarks["hip_row"].y),
    }

    values: dict[str, Measurement] = {}
    warnings: list[str] = []

    def add(
        name: str,
        value: float,
        confidence: str,
        views: list[str],
        source: str = "geometric",
        unit: str = "mm",
    ) -> None:
        """Store one value, converting mm to cm but leaving degrees alone."""
        values[name] = Measurement(
            _cm(value) if unit == "mm" else round(value, 1), source, confidence, views
        )

    # --- heights and lengths -------------------------------------------------
    values["height"] = Measurement(round(height_cm, 1), "tailor_input", "high", ["front"])
    add(
        "head_l",
        _distance_mm(
            first.landmarks["nape"], first.landmarks["head_row"], first.scale
        ),
        "medium",
        [first.name],
    )
    add(
        "neck_w",
        _distance_mm(front["neck_base_left"], front["neck_base_right"], front_scale),
        "medium",
        ["front"],
    )
    add(
        "shoulder_w",
        _distance_mm(front["shoulder_left"], front["shoulder_right"], front_scale),
        "medium",
        ["front"],
    )
    add(
        "shoulder_incl",
        float(
            np.mean(
                [
                    _angle_above_horizontal(
                        front["neck_base_left"], front["shoulder_left"], front_scale
                    ),
                    _angle_above_horizontal(
                        front["neck_base_right"], front["shoulder_right"], front_scale
                    ),
                ]
            )
        ),
        "medium",
        ["front"],
        unit="deg",
    )
    add(
        "armscye_depth",
        float(
            np.mean(
                [
                    front_scale.height_mm(front["shoulder_left"].y),
                    front_scale.height_mm(front["shoulder_right"].y),
                ]
            )
        )
        - h["armpit"],
        "medium",
        ["front"],
    )

    arm_lengths, pose_angles = [], []
    for name, shoulder_index, elbow_index in (
        ("left", L_SHOULDER, L_ELBOW),
        ("right", R_SHOULDER, R_ELBOW),
    ):
        shoulder = front[f"shoulder_{name}"]
        elbow = Landmark(
            "elbow", "front", float(keypoints[elbow_index, 0]), float(keypoints[elbow_index, 1])
        )
        wrist_a = front[f"wrist_{name}_a"]
        wrist_b = front[f"wrist_{name}_b"]
        wrist = Landmark(
            "wrist", "front", (wrist_a.x + wrist_b.x) / 2, (wrist_a.y + wrist_b.y) / 2
        )
        arm_lengths.append(
            _distance_mm(shoulder, elbow, front_scale) + _distance_mm(elbow, wrist, front_scale)
        )
        keypoint_shoulder = Landmark(
            "shoulder",
            "front",
            float(keypoints[shoulder_index, 0]),
            float(keypoints[shoulder_index, 1]),
        )
        pose_angles.append(_angle_above_horizontal(keypoint_shoulder, elbow, front_scale))

    add("arm_length", float(np.mean(arm_lengths)), "medium", ["front"])
    add("arm_pose_angle", float(np.mean(pose_angles)), "high", ["front"], unit="deg")

    # --- girths --------------------------------------------------------------
    for name, height, confidence in (
        ("bust", h["bust"], "medium"),
        ("underbust", h["underbust"], "low"),
        ("waist", h["waist"], "medium"),
        ("hips", h["hips"], "medium"),
    ):
        girth, arc = _girth_at(height, front_mask, torso_x, sides, front_scale)
        add(name, girth, confidence, both)
        if name in ("bust", "waist", "hips"):
            key = {
                "bust": "back_width",
                "waist": "waist_back_width",
                "hips": "hip_back_width",
            }[name]
            add(key, arc, "low", both)
        spread = _girth_spread_mm(height, front_mask, torso_x, sides, front_scale)
        if spread > GIRTH_DISAGREEMENT_MM:
            warnings.append(
                f"{name} depth differs {spread / 10:.1f} cm between the two side views"
            )

    thigh_rows = range(
        int(round(front["crotch_row"].y)),
        int(round(front["crotch_row"].y)) + int(round(THIGH_DEPTH_MM / front_scale.mm_per_px)) + 1,
    )
    leg_girths = [
        _leg_girth_mm(row, front_mask, torso_x, sides, front_scale)
        for row in thigh_rows
        if row < front_mask.shape[0]
    ]
    add("leg_circ", max(leg_girths, default=0.0), "medium", both)

    wrist_width = _distance_mm(front["wrist_left_a"], front["wrist_left_b"], front_scale)
    wrist_width_right = _distance_mm(front["wrist_right_a"], front["wrist_right_b"], front_scale)
    add("wrist", math.pi * (wrist_width + wrist_width_right) / 2, "low", ["front"])

    # --- contour lengths and vertical differences ----------------------------
    nape = first.landmarks["nape"]
    hps = first.landmarks["hps"]
    waist_in_side = first.scale.row_of(h["waist"])
    mm = first.scale.mm_per_px  # the front_scale is isotropic, so x and y share it
    add(
        "waist_line",
        contour_length(_edge_path(first, nape.y, waist_in_side, "back"), mm, mm),
        "medium",
        [first.name],
    )
    add(
        "waist_over_bust_line",
        hull_length(_edge_path(first, hps.y, waist_in_side, "front"), mm, mm),
        "low",
        [first.name],
    )
    add(
        "bust_line",
        hull_length(_edge_path(first, hps.y, first.landmarks["bust_row"].y, "front"), mm, mm),
        "low",
        [first.name],
    )
    add(
        "vert_bust_line",
        first.scale.height_mm(nape.y) - h["bust"],
        "medium",
        [first.name],
    )
    add("hips_line", h["waist"] - h["hips"], "medium", both)
    add("crotch_hip_diff", h["hips"] - h["crotch"], "medium", both)

    waist_row = front_scale.row_of(h["waist"])
    hip_row = front_scale.row_of(h["hips"])
    inclinations = []
    for name in ("left", "right"):
        waist_run = run_containing(front_mask, waist_row, torso_x)
        hip_run = run_containing(front_mask, hip_row, torso_x)
        if waist_run is None or hip_run is None:
            continue
        index = 0 if name == "left" else 1
        waist_edge = Landmark("w", "front", float(waist_run[index]), float(waist_row))
        hip_edge = Landmark("h", "front", float(hip_run[index]), float(hip_row))
        inclinations.append(_angle_from_vertical(waist_edge, hip_edge, front_scale))
    add(
        "hip_inclination",
        float(np.mean(inclinations or [0.0])),
        "medium",
        ["front"],
        unit="deg",
    )

    # --- not visible in a silhouette ----------------------------------------
    values["bust_points"] = Measurement(
        round(values["bust"].value_cm * BUST_POINTS_RATIO, 1), "default_ratio", "estimated", []
    )
    values["bum_points"] = Measurement(
        round(values["hips"].value_cm * BUM_POINTS_RATIO, 1), "default_ratio", "estimated", []
    )

    return MeasurementSet(values=values, warnings=warnings)


def apply_overrides(
    measurements: MeasurementSet, overrides: dict[str, dict[str, object]]
) -> MeasurementSet:
    """Return the set with the tailor's own values substituted in.

    An overridden value keeps its schema confidence but is marked ``edited``, so
    the report shows plainly which numbers a person chose rather than measured.
    """
    values = dict(measurements.values)
    for name, edit in overrides.items():
        if name not in values:
            continue
        current = values[name]
        values[name] = Measurement(
            value_cm=round(float(edit["value"]), 1),
            source="tailor",
            confidence=current.confidence,
            views=current.views,
            edited=True,
            note=str(edit.get("note", "")),
        )
    return MeasurementSet(values=values, warnings=measurements.warnings)
