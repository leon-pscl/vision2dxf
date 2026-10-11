"""Accept or reject a capture, with reasons the operator can act on (brief 5.2).

Every reason is written to be read verbatim by the person in front of the
camera, so they say what to change, not which check failed. Keypoint sides are
the subject's own, as COCO numbers them.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from bodymeasure.geometry import run_containing

KEYPOINT_NAMES = (
    "nose",
    "left eye",
    "right eye",
    "left ear",
    "right ear",
    "left shoulder",
    "right shoulder",
    "left elbow",
    "right elbow",
    "left wrist",
    "right wrist",
    "left hip",
    "right hip",
    "left knee",
    "right knee",
    "left ankle",
    "right ankle",
)
NOSE, L_EAR, R_EAR = 0, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16
SIDE_KEYPOINTS = {
    "left": (L_SHOULDER, L_HIP, L_KNEE, L_ANKLE),
    "right": (R_SHOULDER, R_HIP, R_KNEE, R_ANKLE),
}


@dataclass(frozen=True)
class GateResult:
    """Whether a capture may be used, and why not."""

    ok: bool
    reasons: list[str] = field(default_factory=list)


def _angle_from_torso(shoulder: np.ndarray, elbow: np.ndarray, torso: np.ndarray) -> float:
    """Return the angle in degrees between the upper arm and the torso line."""
    arm = elbow - shoulder
    norms = float(np.linalg.norm(arm)) * float(np.linalg.norm(torso))
    if norms == 0:
        return 0.0
    cosine = float(np.dot(arm, torso)) / norms
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _confidence_reasons(
    keypoints: np.ndarray, indices: tuple[int, ...], min_conf: float
) -> list[str]:
    missing = [KEYPOINT_NAMES[index] for index in indices if keypoints[index, 2] < min_conf]
    if not missing:
        return []
    return [f"Cannot see the {', '.join(missing)} clearly: check lighting and framing"]


def _framing_reasons(mask: np.ndarray, margin_frac: float) -> list[str]:
    """Check the body sits wholly inside the frame with a margin top and bottom."""
    rows = np.flatnonzero(np.asarray(mask).any(axis=1))
    columns = np.flatnonzero(np.asarray(mask).any(axis=0))
    if rows.size == 0:
        return ["No person found in the frame"]

    height, width = mask.shape
    margin_px = margin_frac * height
    reasons = []
    if rows.min() < margin_px:
        reasons.append("Head is too close to the top of the frame: step back or tilt the camera")
    if height - 1 - rows.max() < margin_px:
        reasons.append("Feet are too close to the bottom of the frame: step back")
    if columns.min() == 0 or columns.max() == width - 1:
        reasons.append("Body is cut off at the side of the frame: step back")
    return reasons


def _floor_reasons(mask: np.ndarray, floor_row: int, tolerance_px: float) -> list[str]:
    rows = np.flatnonzero(np.asarray(mask).any(axis=1))
    if rows.size == 0:
        return []
    offset = int(rows.max()) - floor_row
    if abs(offset) <= tolerance_px:
        return []
    direction = "behind" if offset < 0 else "in front of"
    return [f"Stand on the floor mark: feet are {abs(offset)} px {direction} it"]


def gate_front(
    mask: np.ndarray,
    keypoints: np.ndarray,
    thresholds: Mapping[str, float],
    floor_row: int,
) -> GateResult:
    """Gate a front capture: all keypoints visible, A-pose, framing, floor mark."""
    min_conf = thresholds["keypoint_conf"]
    reasons = _confidence_reasons(keypoints, tuple(range(len(KEYPOINT_NAMES))), min_conf)
    reasons += _framing_reasons(mask, thresholds["frame_margin_frac"])
    reasons += _floor_reasons(mask, floor_row, thresholds["floor_row_tol_px"])

    shoulder_mid = (keypoints[L_SHOULDER, :2] + keypoints[R_SHOULDER, :2]) / 2
    hip_mid = (keypoints[L_HIP, :2] + keypoints[R_HIP, :2]) / 2
    torso = hip_mid - shoulder_mid
    torso_x = int(round(hip_mid[0]))
    low, high = thresholds["arm_angle_min_deg"], thresholds["arm_angle_max_deg"]

    for side, shoulder_index, elbow_index, wrist_index in (
        ("Left", L_SHOULDER, L_ELBOW, L_WRIST),
        ("Right", R_SHOULDER, R_ELBOW, R_WRIST),
    ):
        angle = _angle_from_torso(
            keypoints[shoulder_index, :2], keypoints[elbow_index, :2], torso
        )
        if angle < low:
            reasons.append(
                f"{side} arm too close to body ({angle:.0f} degrees, need {low:.0f}-{high:.0f})"
            )
        elif angle > high:
            reasons.append(
                f"{side} arm too far from body ({angle:.0f} degrees, need {low:.0f}-{high:.0f})"
            )

        wrist = keypoints[wrist_index, :2]
        run = run_containing(mask, int(round(wrist[1])), torso_x)
        if run is not None and run[0] <= wrist[0] <= run[1]:
            reasons.append(f"{side} wrist overlaps the torso: move the hand away from the body")

    return GateResult(ok=not reasons, reasons=reasons)


def visible_side(keypoints: np.ndarray) -> str:
    """Return which side of the body faces the camera, by mean keypoint confidence."""
    scores = {
        side: float(np.mean([keypoints[index, 2] for index in indices]))
        for side, indices in SIDE_KEYPOINTS.items()
    }
    return max(scores, key=lambda side: scores[side])


def gate_side(
    mask: np.ndarray,
    keypoints: np.ndarray,
    thresholds: Mapping[str, float],
    floor_row: int,
    front_shoulder_px: float,
) -> GateResult:
    """Gate a side capture: visible side seen, truly side-on, framing, floor mark."""
    min_conf = thresholds["keypoint_conf"]
    side = visible_side(keypoints)
    reasons = _confidence_reasons(keypoints, SIDE_KEYPOINTS[side], min_conf)
    reasons += _framing_reasons(mask, thresholds["frame_margin_frac"])
    reasons += _floor_reasons(mask, floor_row, thresholds["floor_row_tol_px"])

    span = abs(float(keypoints[L_SHOULDER, 0] - keypoints[R_SHOULDER, 0]))
    limit = thresholds["side_shoulder_ratio_max"] * front_shoulder_px
    if front_shoulder_px > 0 and span > limit:
        reasons.append("Not side-on yet: turn further so one shoulder hides the other")

    return GateResult(ok=not reasons, reasons=reasons)


def faces_right(keypoints: np.ndarray) -> bool:
    """Return whether a side view already faces +x (brief 5.4).

    The nose sits in front of the ear, so the sign of nose-x minus ear-x gives
    the facing direction. A view that faces left is mirrored before landmarking.
    """
    side = visible_side(keypoints)
    ear = L_EAR if side == "left" else R_EAR
    return float(keypoints[NOSE, 0]) >= float(keypoints[ear, 0])
