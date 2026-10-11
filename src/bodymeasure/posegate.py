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
# Only these feed the landmarks and the measurements, so a soft ankle or a
# turned-away eye must not cost the operator a retake. The legs still have to be
# in the mask: the scale comes from its head-to-floor span, not from keypoints.
MEASURED_KEYPOINTS = (
    L_EAR,
    R_EAR,
    L_SHOULDER,
    R_SHOULDER,
    L_ELBOW,
    R_ELBOW,
    L_WRIST,
    R_WRIST,
    L_HIP,
    R_HIP,
)
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


def _framing_reasons(mask: np.ndarray, thresholds: Mapping[str, float]) -> list[str]:
    """Check the body fills the frame and stands in the middle of it.

    There is no floor mark to stand on any more: the scale comes from the
    subject's height, so what the gate must protect is the whole body being in
    frame at a usable size - top and bottom margins, a tall enough silhouette,
    and the body near the middle column so the lens sees it square on.
    """
    rows = np.flatnonzero(np.asarray(mask).any(axis=1))
    columns = np.flatnonzero(np.asarray(mask).any(axis=0))
    if rows.size == 0:
        return ["No person found in the frame"]

    height, width = mask.shape
    margin_px = thresholds["frame_margin_frac"] * height
    reasons = []
    if rows.min() < margin_px:
        reasons.append("Head is too close to the top of the frame: step back or tilt the camera")
    if height - 1 - rows.max() < margin_px:
        reasons.append("Feet are too close to the bottom of the frame: step back")
    if rows.max() - rows.min() + 1 < thresholds["body_height_frac"] * height:
        reasons.append("Body too small in the frame: step closer, or crop an uploaded photo")
    offset = abs((columns.min() + columns.max()) / 2 - width / 2)
    if offset > thresholds["body_centre_band_frac"] * width / 2:
        reasons.append("Move towards the middle of the frame: stand square to the camera")
    if columns.min() == 0 or columns.max() == width - 1:
        reasons.append("Body is cut off at the side of the frame: step back")
    return reasons


def gate_front(
    mask: np.ndarray,
    keypoints: np.ndarray,
    thresholds: Mapping[str, float],
) -> GateResult:
    """Gate a front capture: the measured keypoints visible, A-pose, and framing."""
    min_conf = thresholds["keypoint_conf"]
    reasons = _confidence_reasons(keypoints, MEASURED_KEYPOINTS, min_conf)
    reasons += _framing_reasons(mask, thresholds)

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
    front_shoulder_px: float,
) -> GateResult:
    """Gate a side capture: visible side seen, truly side-on, and framing."""
    min_conf = thresholds["keypoint_conf"]
    side = visible_side(keypoints)
    reasons = _confidence_reasons(keypoints, SIDE_KEYPOINTS[side], min_conf)
    reasons += _framing_reasons(mask, thresholds)

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
