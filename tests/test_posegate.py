"""The pose gate accepts a good capture and explains every bad one."""

import numpy as np
import pytest
from tests.synthetic.people import make_person

from bodymeasure.config import load_app_config
from bodymeasure.posegate import (
    L_SHOULDER,
    R_SHOULDER,
    faces_right,
    gate_front,
    gate_side,
    visible_side,
)


@pytest.fixture
def thresholds() -> dict[str, float]:
    return dict(load_app_config().thresholds)


def test_a_good_front_capture_passes(thresholds: dict[str, float]) -> None:
    person = make_person(arm_angle_deg=30.0)
    result = gate_front(person.mask, person.keypoints, thresholds)
    assert result.ok, result.reasons
    assert result.reasons == []


@pytest.mark.parametrize(
    ("angle", "fragment"),
    [(8.0, "too close to body"), (70.0, "too far from body")],
)
def test_arms_outside_the_a_pose_are_named(
    thresholds: dict[str, float], angle: float, fragment: str
) -> None:
    person = make_person(arm_angle_deg=angle)
    result = gate_front(person.mask, person.keypoints, thresholds)
    assert not result.ok
    assert sum(fragment in reason for reason in result.reasons) == 2  # both arms
    assert any(reason.startswith("Left arm") for reason in result.reasons)


def test_a_wrist_over_the_torso_is_rejected(thresholds: dict[str, float]) -> None:
    person = make_person(wrist_over_torso=True)
    result = gate_front(person.mask, person.keypoints, thresholds)
    assert not result.ok
    assert any("wrist overlaps the torso" in reason for reason in result.reasons)


def test_a_hidden_keypoint_is_named(thresholds: dict[str, float]) -> None:
    person = make_person()
    keypoints = person.keypoints.copy()
    keypoints[9, 2] = 0.2  # left wrist
    result = gate_front(person.mask, keypoints, thresholds)
    assert not result.ok
    assert any("left wrist" in reason for reason in result.reasons)


def test_a_hidden_leg_keypoint_still_passes(thresholds: dict[str, float]) -> None:
    """Nothing is measured off the legs or the face, so a soft ankle is no retake."""
    person = make_person(arm_angle_deg=30.0)
    keypoints = person.keypoints.copy()
    keypoints[[0, 1, 2, 13, 14, 15, 16], 2] = 0.05  # face, knees and ankles
    assert gate_front(person.mask, keypoints, thresholds).ok


def test_a_body_too_close_to_the_top_is_rejected(thresholds: dict[str, float]) -> None:
    """The configured margin of frame must stay clear above the head."""
    person = make_person()
    height = person.mask.shape[0]
    top = int(np.flatnonzero(person.mask.any(axis=1))[0])
    shift = top - int(thresholds["frame_margin_frac"] * height) + 1  # leaves under the margin
    cropped = np.zeros_like(person.mask)
    cropped[:-shift] = person.mask[shift:]
    keypoints = person.keypoints.copy()
    keypoints[:, 1] -= shift
    result = gate_front(cropped, keypoints, thresholds)
    assert not result.ok
    assert any("top of the frame" in reason for reason in result.reasons)


def test_a_body_too_short_in_the_frame_is_rejected(thresholds: dict[str, float]) -> None:
    """Below the configured span there are too few pixels per millimetre.

    The fix is to come closer, not to step back: the body has to fill the frame,
    or every millimetre the engine reports is quantised over too few pixels.
    """
    person = make_person()
    kept = int((thresholds["body_height_frac"] - 0.01) * person.mask.shape[0])
    squashed = np.zeros_like(person.mask)
    squashed[:kept] = person.mask[:kept]
    result = gate_front(squashed, person.keypoints, thresholds)
    assert not result.ok
    assert any("too small in the frame" in reason for reason in result.reasons)


def test_a_body_off_to_one_side_is_rejected(thresholds: dict[str, float]) -> None:
    """The body must sit in the middle of the frame, or the lens sees it at an angle."""
    person = make_person()
    height, width = person.mask.shape
    shifted = int(0.9 * width)  # body centre ends 360 px off centre in a frame twice as wide
    moved = np.zeros((height, 2 * width), dtype=bool)
    moved[:, shifted : shifted + width] = person.mask
    keypoints = person.keypoints.copy()
    keypoints[:, 0] += shifted
    result = gate_front(moved, keypoints, thresholds)
    assert not result.ok
    assert any("middle of the frame" in reason for reason in result.reasons)


def test_an_empty_mask_says_so(thresholds: dict[str, float]) -> None:
    person = make_person()
    result = gate_front(np.zeros_like(person.mask), person.keypoints, thresholds)
    assert not result.ok
    assert "No person found in the frame" in result.reasons


def test_a_good_side_capture_passes(thresholds: dict[str, float]) -> None:
    person = make_person(side_on=True)
    front_span = make_person().shoulder_span_px
    result = gate_side(
        person.mask, person.keypoints, thresholds, front_span
    )
    assert result.ok, result.reasons


def test_a_half_turned_subject_is_rejected(thresholds: dict[str, float]) -> None:
    person = make_person(side_on=True)
    keypoints = person.keypoints.copy()
    keypoints[L_SHOULDER, 0] -= 60
    keypoints[R_SHOULDER, 0] += 60
    result = gate_side(
        person.mask, keypoints, thresholds, person.shoulder_span_px
    )
    assert not result.ok
    assert any("side-on" in reason for reason in result.reasons)


def test_the_side_gate_only_needs_the_visible_side(thresholds: dict[str, float]) -> None:
    person = make_person(side_on=True)
    assert visible_side(person.keypoints) == "left"
    result = gate_side(
        person.mask, person.keypoints, thresholds, 190.0
    )
    assert result.ok, result.reasons


def test_the_side_gate_names_a_missing_visible_keypoint(thresholds: dict[str, float]) -> None:
    person = make_person(side_on=True)
    keypoints = person.keypoints.copy()
    keypoints[13, 2] = 0.1  # left knee, on the visible side
    result = gate_side(person.mask, keypoints, thresholds, 190.0)
    assert not result.ok
    assert any("left knee" in reason for reason in result.reasons)


def test_facing_direction_comes_from_nose_against_ear() -> None:
    person = make_person(side_on=True)
    keypoints = person.keypoints.copy()
    keypoints[0, 0] = keypoints[3, 0] + 50
    assert faces_right(keypoints)
    keypoints[0, 0] = keypoints[3, 0] - 50
    assert not faces_right(keypoints)


def test_shoulder_span_zero_skips_the_side_on_check(thresholds: dict[str, float]) -> None:
    """Without a front view there is nothing to compare against."""
    person = make_person(side_on=True)
    keypoints = person.keypoints.copy()
    keypoints[L_SHOULDER, 0] -= 200
    keypoints[R_SHOULDER, 0] += 200
    result = gate_side(person.mask, keypoints, thresholds, 0.0)
    assert result.ok, result.reasons
