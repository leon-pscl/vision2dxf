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
    montazerian_girth,
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


# Allowed ``status`` values of a provenance record (brief 14).
ACQUIRED = "acquired"
OPERATOR_CONFIRMED = "operator_confirmed"
MANUAL_INPUT = "manual_input"
ESTIMATED_PENDING = "estimated_pending_acceptance"
UNAVAILABLE = "unavailable"
INVALIDATED = "invalidated"
BLOCKING_STATUSES = (UNAVAILABLE, INVALIDATED, ESTIMATED_PENDING)

# Allowed ``source`` values (brief 14).
GEOMETRIC_2D = "geometric_2d"
MESH_VISUAL_HULL = "mesh_visual_hull"
OPERATOR_INPUT = "operator_input"
DERIVED_PATTERN_PARAMETER = "derived_pattern_parameter"


@dataclass(frozen=True)
class Measurement:
    """One value and the whole provenance record behind it (brief 14).

    ``value`` is the internal value: millimetres when ``unit`` is ``"mm"``,
    degrees when it is ``"deg"``, and ``None`` when the measurement could not be
    taken. A failed measurement is never zero and never a population average: it
    is ``None`` with the reason in ``warnings``.
    """

    value: float | None
    source: str = GEOMETRIC_2D
    confidence: str = "medium"
    views: list[str] = field(default_factory=list)
    edited: bool = False
    note: str = ""
    unit: str = "mm"
    status: str = ACQUIRED
    method: str = ""
    formula: str = ""
    inputs: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    auto: bool = True

    @property
    def value_cm(self) -> float | None:
        """Return the display value: centimetres to one decimal, or degrees."""
        if self.value is None:
            return None
        return round(self.value / 10.0, 1) if self.unit == "mm" else round(self.value, 1)

    @property
    def available(self) -> bool:
        """Return whether this value may be exported as a body measurement."""
        return self.value is not None and self.status not in BLOCKING_STATUSES

    def record(self, name: str) -> dict[str, object]:
        """Return the provenance record for ``measurement_report.yaml`` (brief 14)."""
        value_key = "value_mm" if self.unit == "mm" else "value_deg"
        return {
            "name": name,
            value_key: None if self.value is None else round(self.value, 2),
            "value_cm_display" if self.unit == "mm" else "value_deg_display": self.value_cm,
            "status": self.status,
            "source": self.source,
            "method": self.method,
            "views": list(self.views),
            "inputs": {key: round(value, 2) for key, value in self.inputs.items()},
            "formula": self.formula,
            "confidence": self.confidence,
            "validated_against_manual": False,
            "auto": self.auto,
            "edited": self.edited,
            "warnings": list(self.warnings),
            **({"note": self.note} if self.note else {}),
        }


@dataclass(frozen=True)
class MeasurementSet:
    """Every measurement, plus the warnings the operator should read."""

    values: dict[str, Measurement]
    warnings: list[str] = field(default_factory=list)

    def body(self) -> dict[str, float | None]:
        """Return the flat ``body:`` mapping for ``body_measurements.yaml``.

        A measurement that was not acquired is exported as ``null`` rather than
        omitted, so a pattern engine sees the gap instead of a plausible number.
        """
        return {
            name: entry.value_cm if entry.available else None
            for name, entry in sorted(self.values.items())
        }

    def missing(self, required: list[str]) -> dict[str, str]:
        """Return the required fields that block export, each with its reason."""
        blocked = {}
        for name in required:
            entry = self.values.get(name)
            if entry is None:
                blocked[name] = "not computed by this pipeline"
            elif not entry.available:
                blocked[name] = "; ".join(entry.warnings) or entry.status
        return blocked


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


@dataclass(frozen=True)
class Girth:
    """One circumference and every intermediate value behind it (brief 3.1)."""

    value_mm: float | None
    formula: str
    inputs: dict[str, float]
    back_arc_mm: float | None
    warnings: list[str] = field(default_factory=list)


def _girth_at(
    h_mm: float, front_mask: np.ndarray, torso_x: int, sides: list[SideView], front: ViewScale
) -> Girth:
    """Return the Montazerian circumference at a height, with its inputs.

    The breadth comes from the front silhouette and the depth from the side
    silhouette *at the same height above the floor*, never at the same pixel row
    (brief 10). The back arc is a separate, asymmetric model (brief 12.2) and is
    marked as such where it is stored.
    """
    front_row = front.row_of(h_mm)
    width = width_at(front_mask, front_row, torso_x, front.mm_per_px)
    depth_front, depth_back, _ = _mean_depths(sides, h_mm)
    depth = depth_front + depth_back
    if width <= 0 or depth <= 0:
        reason = "front breadth" if width <= 0 else "side depth"
        return Girth(None, "", {}, None, [f"no {reason} at this height: check the mask and rows"])

    girth, formula = montazerian_girth(width, depth)
    inputs = {
        "front_breadth_mm": width,
        "side_depth_mm": depth,
        "side_depth_front_mm": depth_front,
        "side_depth_back_mm": depth_back,
        "row_height_mm": h_mm,
        "front_row": float(front_row),
        "front_mm_per_px": front.mm_per_px,
    }
    for side in sides:
        inputs[f"{side.name}_row"] = float(side.scale.row_of(h_mm))
        inputs[f"{side.name}_mm_per_px"] = side.scale.mm_per_px
    return Girth(girth, formula, inputs, back_arc(width, depth_back))


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
        girths.append(montazerian_girth(width, depth_front + depth_back)[0])
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
        value: float | None,
        confidence: str,
        views: list[str],
        method: str,
        source: str = GEOMETRIC_2D,
        unit: str = "mm",
        inputs: dict[str, float] | None = None,
        formula: str = "",
        reasons: list[str] | None = None,
    ) -> None:
        """Store one provenance record, keeping the internal value unrounded."""
        reasons = reasons or []
        values[name] = Measurement(
            value=value,
            source=source,
            confidence=confidence,
            views=views,
            unit=unit,
            status=UNAVAILABLE if value is None else ACQUIRED,
            method=method,
            formula=formula,
            inputs=inputs or {},
            warnings=reasons,
        )

    # --- heights and lengths -------------------------------------------------
    values["height"] = Measurement(
        value=height_cm * 10.0,
        source=OPERATOR_INPUT,
        confidence="high",
        views=["front"],
        status=MANUAL_INPUT,
        method="operator_measured_stature",
        auto=False,
    )
    add(
        "head_l",
        _distance_mm(
            first.landmarks["nape"], first.landmarks["head_row"], first.scale
        ),
        "medium",
        [first.name],
        method="landmark_distance_nape_to_head_top",
    )
    add(
        "neck_w",
        _distance_mm(front["neck_base_left"], front["neck_base_right"], front_scale),
        "medium",
        ["front"],
        method="landmark_distance_neck_base_left_right",
    )
    add(
        "shoulder_w",
        _distance_mm(front["shoulder_left"], front["shoulder_right"], front_scale),
        "medium",
        ["front"],
        method="landmark_distance_shoulder_left_right",
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
        method="mean_angle_neck_base_to_shoulder_above_horizontal",
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
        method="height_difference_shoulder_to_armpit_row",
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

    add(
        "arm_length",
        float(np.mean(arm_lengths)),
        "medium",
        ["front"],
        method="polyline_shoulder_elbow_wrist_mean_of_sides",
        inputs={"left_mm": arm_lengths[0], "right_mm": arm_lengths[1]},
    )
    add(
        "arm_pose_angle",
        float(np.mean(pose_angles)),
        "high",
        ["front"],
        method="mean_upper_arm_angle_above_horizontal",
        unit="deg",
    )
    # Left and right are kept separately as well as averaged, because a real
    # body is asymmetric and the operator is the one who decides to accept the
    # mean (brief 3.1).
    for side_name, index in (("left", 0), ("right", 1)):
        add(
            f"shoulder_length_{side_name}",
            _distance_mm(
                front[f"neck_base_{side_name}"], front[f"shoulder_{side_name}"], front_scale
            ),
            "medium",
            ["front"],
            method="landmark_distance_neck_base_to_shoulder_point",
        )
        add(
            f"arm_length_{side_name}",
            arm_lengths[index],
            "medium",
            ["front"],
            method="polyline_shoulder_elbow_wrist",
        )
    add(
        "shoulder_length",
        float(
            np.mean([values["shoulder_length_left"].value, values["shoulder_length_right"].value])
        ),
        "medium",
        ["front"],
        method="mean_of_left_and_right_shoulder_length",
    )

    # --- girths --------------------------------------------------------------
    arc_keys = {"bust": "back_width", "waist": "waist_back_width", "hips": "hip_back_width"}
    for name, height, confidence in (
        ("chest", h["armpit"], "medium"),
        ("bust", h["bust"], "medium"),
        ("underbust", h["underbust"], "low"),
        ("waist", h["waist"], "medium"),
        ("hips", h["hips"], "medium"),
    ):
        girth = _girth_at(height, front_mask, torso_x, sides, front_scale)
        add(
            name,
            girth.value_mm,
            confidence,
            both,
            method="montazerian_ellipse",
            inputs=girth.inputs,
            formula=girth.formula,
            reasons=girth.warnings,
        )
        if name in arc_keys:
            add(
                arc_keys[name],
                girth.back_arc_mm,
                "low",
                both,
                method="semi_ellipse_back_arc_from_balance_line",
                inputs=girth.inputs,
                formula="semi_ellipse_ramanujan_ii",
                reasons=girth.warnings
                or ["method-specific approximation, not validated against a tape"],
            )
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
    add(
        "leg_circ",
        max(leg_girths) if leg_girths else None,
        "medium",
        both,
        method="montazerian_ellipse_thigh_section",
        reasons=[] if leg_girths else ["no thigh section below the crotch row"],
    )

    wrist_width = _distance_mm(front["wrist_left_a"], front["wrist_left_b"], front_scale)
    wrist_width_right = _distance_mm(front["wrist_right_a"], front["wrist_right_b"], front_scale)
    add(
        "wrist",
        math.pi * (wrist_width + wrist_width_right) / 2,
        "low",
        ["front"],
        method="circular_section_from_front_width",
        inputs={"left_width_mm": wrist_width, "right_width_mm": wrist_width_right},
        formula="circle_from_diameter",
        reasons=["modelled as a circle: a side view cannot see the wrist clear of the body"],
    )

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
        method="side_back_contour_nape_to_waist",
        formula="polyline_length",
    )
    add(
        "waist_over_bust_line",
        hull_length(_edge_path(first, hps.y, waist_in_side, "front"), mm, mm),
        "low",
        [first.name],
        method="side_front_tape_path_hps_to_waist",
        formula="convex_hull_path_length",
    )
    add(
        "bust_line",
        hull_length(_edge_path(first, hps.y, first.landmarks["bust_row"].y, "front"), mm, mm),
        "low",
        [first.name],
        method="side_front_tape_path_hps_to_bust",
        formula="convex_hull_path_length",
    )
    add(
        "vert_bust_line",
        first.scale.height_mm(nape.y) - h["bust"],
        "medium",
        [first.name],
        method="height_difference_nape_to_bust_row",
    )
    add(
        "hips_line",
        h["waist"] - h["hips"],
        "medium",
        both,
        method="height_difference_waist_row_to_hip_row",
    )
    add(
        "crotch_hip_diff",
        h["hips"] - h["crotch"],
        "medium",
        both,
        method="height_difference_hip_row_to_crotch_row",
    )

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
        float(np.mean(inclinations)) if inclinations else None,
        "medium",
        ["front"],
        method="mean_waist_to_hip_edge_angle_from_vertical",
        unit="deg",
        reasons=[] if inclinations else ["waist or hip row misses the torso: check the rows"],
    )

    # --- not visible in a silhouette ----------------------------------------
    # A population ratio is not a measurement. These two are offered as a
    # starting point the operator must accept or replace, and they stay
    # ``estimated_pending_acceptance`` until someone does (brief 2, 14).
    for name, from_key, ratio in (
        ("bust_points", "bust", BUST_POINTS_RATIO),
        ("bum_points", "hips", BUM_POINTS_RATIO),
    ):
        base = values[from_key].value
        values[name] = Measurement(
            value=None if base is None else base * ratio,
            source=DERIVED_PATTERN_PARAMETER,
            confidence="estimated",
            views=[],
            status=UNAVAILABLE if base is None else ESTIMATED_PENDING,
            method=f"population_ratio_of_{from_key}",
            inputs={"ratio": ratio, **({f"{from_key}_mm": base} if base is not None else {})},
            warnings=[
                "population ratio from the GarmentCode mean body, not measured: "
                "accept it or enter the measured value"
            ],
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
        entered = float(edit["value"])
        values[name] = Measurement(
            value=entered * 10.0 if current.unit == "mm" else entered,
            source=OPERATOR_INPUT,
            confidence=current.confidence,
            views=current.views,
            edited=True,
            note=str(edit.get("note", "")),
            unit=current.unit,
            status=MANUAL_INPUT,
            method="operator_entered_value",
            inputs=current.inputs,
            auto=False,
        )
    return MeasurementSet(values=values, warnings=measurements.warnings)
