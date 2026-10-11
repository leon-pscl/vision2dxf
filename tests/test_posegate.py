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
    result = gate_front(person.mask, person.keypoints, thresholds, person.floor_row)
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
    result = gate_front(person.mask, person.keypoints, thresholds, person.floor_row)
    assert not result.ok
    assert sum(fragment in reason for reason in result.reasons) == 2  # both arms
    assert any(reason.startswith("Left arm") for reason in result.reasons)


def test_a_wrist_over_the_torso_is_rejected(thresholds: dict[str, float]) -> None:
    person = make_person(wrist_over_torso=True)
    result = gate_front(person.mask, person.keypoints, thresholds, person.floor_row)
    assert not result.ok
    assert any("wrist overlaps the torso" in reason for reason in result.reasons)


def test_a_hidden_keypoint_is_named(thresholds: dict[str, float]) -> None:
    person = make_person()
    keypoints = person.keypoints.copy()
    keypoints[15, 2] = 0.2  # left ankle
    result = gate_front(person.mask, keypoints, thresholds, person.floor_row)
    assert not result.ok
    assert any("left ankle" in reason for reason in result.reasons)


def test_standing_off_the_floor_mark_is_rejected(thresholds: dict[str, float]) -> None:
    person = make_person(floor_offset_px=-40)
    result = gate_front(person.mask, person.keypoints, thresholds, person.floor_row)
    assert not result.ok
    assert any("floor mark" in reason for reason in result.reasons)


def test_a_tight_crop_is_rejected(thresholds: dict[str, float]) -> None:
    person = make_person()
    cropped = person.mask[100:, :]  # head now sits at the very top
    result = gate_front(cropped, person.keypoints, thresholds, person.floor_row)
    assert not result.ok
    assert any("top of the frame" in reason for reason in result.reasons)


def test_an_empty_mask_says_so(thresholds: dict[str, float]) -> None:
    person = make_person()
    result = gate_front(np.zeros_like(person.mask), person.keypoints, thresholds, person.floor_row)
    assert not result.ok
    assert "No person found in the frame" in result.reasons


def test_a_good_side_capture_passes(thresholds: dict[str, float]) -> None:
    person = make_person(side_on=True)
    front_span = make_person().shoulder_span_px
    result = gate_side(
        person.mask, person.keypoints, thresholds, person.floor_row, front_span
    )
    assert result.ok, result.reasons


def test_a_half_turned_subject_is_rejected(thresholds: dict[str, float]) -> None:
    person = make_person(side_on=True)
    keypoints = person.keypoints.copy()
    keypoints[L_SHOULDER, 0] -= 60
    keypoints[R_SHOULDER, 0] += 60
    result = gate_side(
        person.mask, keypoints, thresholds, person.floor_row, person.shoulder_span_px
    )
    assert not result.ok
    assert any("side-on" in reason for reason in result.reasons)


def test_the_side_gate_only_needs_the_visible_side(thresholds: dict[str, float]) -> None:
    person = make_person(side_on=True)
    assert visible_side(person.keypoints) == "left"
    result = gate_side(
        person.mask, person.keypoints, thresholds, person.floor_row, 190.0
    )
    assert result.ok, result.reasons


def test_the_side_gate_names_a_missing_visible_keypoint(thresholds: dict[str, float]) -> None:
    person = make_person(side_on=True)
    keypoints = person.keypoints.copy()
    keypoints[13, 2] = 0.1  # left knee, on the visible side
    result = gate_side(person.mask, keypoints, thresholds, person.floor_row, 190.0)
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
    result = gate_side(person.mask, keypoints, thresholds, person.floor_row, 0.0)
    assert result.ok, result.reasons
