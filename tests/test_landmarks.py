"""Landmark search on synthetic figures whose rows are known by construction."""

import json
from pathlib import Path

import numpy as np
import pytest
from tests.synthetic.people import (
    ANKLE_Y,
    ARM_THICKNESS,
    HIP_Y,
    SHOULDER_Y,
    WAIST_Y,
    make_person,
    make_side,
)

from bodymeasure.landmarks import (
    Landmark,
    armpit_row,
    as_dict,
    balance_x_at,
    crotch_row,
    find_front_landmarks,
    find_side_landmarks,
    load_landmarks,
    narrowest_cross_section,
    neck_base_row,
    save_landmarks,
    scale_landmarks,
    turning_angles,
    view_scale,
    waist_row,
)

HEIGHT_CM = 164.5  # the drawn figure spans 1645 rows at 1 mm/px
# Front heights in mm: floor row 1750 less the row each landmark lands on.
HEIGHTS = {"armpit": 1750.0 - 430, "waist": 1750.0 - 760, "crotch": 1750.0 - 1000}


@pytest.fixture
def front() -> object:
    return make_person()


def test_the_scale_rows_are_the_extremes_of_the_mask(front: object) -> None:
    marks = as_dict(scale_landmarks(front.mask, "front", 450.0))
    rows = np.flatnonzero(front.mask.any(axis=1))

    assert marks["floor_row"].y == rows.max()
    assert marks["head_row"].y == rows.min()


def test_the_scale_rows_reject_an_empty_mask() -> None:
    with pytest.raises(ValueError, match="empty mask"):
        scale_landmarks(np.zeros((10, 10), dtype=bool), "front", 5.0)


def test_turning_angles_are_zero_on_a_straight_line() -> None:
    line = np.stack([np.arange(100.0), np.zeros(100)], axis=1)
    assert turning_angles(line, span=5)[50] == pytest.approx(0.0, abs=1e-9)


def test_turning_angles_peak_at_a_corner() -> None:
    down = np.stack([np.zeros(60), np.arange(60.0)], axis=1)
    across = np.stack([np.arange(1.0, 61.0), np.full(60, 59.0)], axis=1)
    corner = np.vstack([down, across])
    angles = turning_angles(corner, span=10)
    assert angles[59] == pytest.approx(90.0, abs=5.0)
    assert angles[30] < 1.0


def test_the_neck_base_sits_between_the_ears_and_the_shoulders(front: object) -> None:
    row = neck_base_row(front.mask, ear_row=225, shoulder_row=SHOULDER_Y, torso_x=front.torso_x)
    assert 225 < row <= SHOULDER_Y
    # It is where the silhouette widens, so the row below the neck is wider.
    narrow = front.mask[row - 20].sum()
    assert front.mask[row].sum() > narrow


def test_the_armpit_is_the_first_row_with_air_on_both_sides(front: object) -> None:
    row = armpit_row(front.mask, SHOULDER_Y, front.torso_x, HIP_Y)
    assert SHOULDER_Y < row < HIP_Y
    assert front.mask[row - 5, front.torso_x]


def test_the_armpit_search_explains_arms_held_against_the_body() -> None:
    person = make_person(arm_angle_deg=0.0)
    pressed = person.mask.copy()
    pressed[SHOULDER_Y:HIP_Y, person.torso_x - 130 : person.torso_x + 130] = True
    with pytest.raises(ValueError, match="arms further out"):
        armpit_row(pressed, SHOULDER_Y, person.torso_x, HIP_Y)


def test_the_waist_is_the_narrowest_torso_row(front: object) -> None:
    armpit = armpit_row(front.mask, SHOULDER_Y, front.torso_x, HIP_Y)
    row = waist_row(front.mask, armpit, HIP_Y, front.torso_x)
    assert abs(row - WAIST_Y) <= 30
    assert front.mask[row].sum() <= front.mask[row - 50].sum()


def test_the_crotch_is_where_the_legs_part(front: object) -> None:
    row = crotch_row(front.mask, HIP_Y, front.torso_x)
    assert HIP_Y <= row < ANKLE_Y
    assert not front.mask[row, front.torso_x]


def test_the_crotch_search_explains_closed_legs() -> None:
    person = make_person()
    closed = person.mask.copy()
    closed[HIP_Y:, person.torso_x - 25 : person.torso_x + 25] = True
    with pytest.raises(ValueError, match="feet apart"):
        crotch_row(closed, HIP_Y, person.torso_x)


def test_the_narrowest_cross_section_measures_across_the_arm(front: object) -> None:
    elbow = front.keypoints[7, :2]
    wrist = front.keypoints[9, :2]
    start, end = narrowest_cross_section(front.mask, wrist, wrist - elbow, search_px=40)
    width = float(np.linalg.norm(end - start))
    assert width == pytest.approx(ARM_THICKNESS, abs=6)


def test_the_cross_section_survives_a_wrist_keypoint_off_the_arm(front: object) -> None:
    """Real masks stop short of the hand, leaving the keypoint just outside."""
    elbow = front.keypoints[7, :2]
    wrist = front.keypoints[9, :2] + np.array([ARM_THICKNESS, 0.0])  # off the silhouette
    start, end = narrowest_cross_section(front.mask, wrist, wrist - elbow, search_px=40)
    width = float(np.linalg.norm(end - start))
    assert width == pytest.approx(ARM_THICKNESS, abs=6)


def test_every_front_landmark_is_found(front: object) -> None:
    marks = as_dict(find_front_landmarks(front.mask, front.keypoints, HEIGHT_CM))
    expected = {
        "floor_row",
        "head_row",
        "neck_base_left",
        "neck_base_right",
        "armpit_row",
        "waist_row",
        "crotch_row",
        "shoulder_left",
        "shoulder_right",
        "wrist_left_a",
        "wrist_left_b",
        "wrist_right_a",
        "wrist_right_b",
    }
    assert set(marks) == expected
    assert all(mark.view == "front" and mark.auto for mark in marks.values())
    assert marks["neck_base_left"].x < front.torso_x < marks["neck_base_right"].x
    assert marks["shoulder_left"].x < marks["neck_base_left"].x
    assert marks["shoulder_right"].x > marks["neck_base_right"].x
    assert marks["head_row"].y < marks["armpit_row"].y < marks["waist_row"].y
    assert marks["waist_row"].y < marks["crotch_row"].y


def test_the_scale_comes_from_the_height_over_the_head_to_floor_span(front: object) -> None:
    """No station calibration: mm/px is the entered height over the drawn span."""
    marks = as_dict(find_front_landmarks(front.mask, front.keypoints, HEIGHT_CM))
    scale = view_scale(list(marks.values()), HEIGHT_CM)

    assert scale.floor_row == marks["floor_row"].y
    assert scale.head_row == marks["head_row"].y
    assert scale.mm_per_px == pytest.approx(HEIGHT_CM * 10 / (scale.floor_row - scale.head_row))
    assert scale.height_mm(scale.head_row) == pytest.approx(HEIGHT_CM * 10)


def test_dragging_the_head_row_shortens_the_span_and_stretches_the_scale(front: object) -> None:
    """The tailor pulls head_row down over hair; every measurement follows."""
    from dataclasses import replace

    marks = find_front_landmarks(front.mask, front.keypoints, HEIGHT_CM)
    before = view_scale(marks, HEIGHT_CM)
    lowered = [
        replace(mark, y=mark.y + 20, auto=False) if mark.name == "head_row" else mark
        for mark in marks
    ]

    after = view_scale(lowered, HEIGHT_CM)

    assert after.head_row == before.head_row + 20
    assert after.mm_per_px > before.mm_per_px, "the same height now spans fewer pixels"


def test_a_scale_without_its_rows_is_reported(front: object) -> None:
    with pytest.raises(ValueError, match="missing scale landmarks"):
        view_scale([Landmark("waist_row", "front", 450.0, 760.0)], HEIGHT_CM)


def test_an_inverted_scale_is_refused() -> None:
    marks = [
        Landmark("floor_row", "front", 0.0, 100.0),
        Landmark("head_row", "front", 0.0, 100.0),
    ]
    with pytest.raises(ValueError, match="below the head row"):
        view_scale(marks, HEIGHT_CM)


def test_balance_line_interpolates_and_extrapolates() -> None:
    top = Landmark("balance_top", "side_left", 100.0, 200.0)
    bottom = Landmark("balance_bottom", "side_left", 140.0, 400.0)
    assert balance_x_at(300.0, top, bottom) == pytest.approx(120.0)
    assert balance_x_at(100.0, top, bottom) == pytest.approx(80.0)
    flat = Landmark("balance_bottom", "side_left", 140.0, 200.0)
    assert balance_x_at(999.0, top, flat) == 100.0


def test_side_landmarks_find_the_drawn_rows() -> None:
    side = make_side()
    marks = as_dict(find_side_landmarks(side.mask, side.keypoints, "side_left", HEIGHTS, HEIGHT_CM))

    assert marks["bust_row"].y == pytest.approx(side.rows["bust"], abs=12)
    assert marks["underbust_row"].y == pytest.approx(side.rows["underbust"], abs=20)
    assert marks["hip_row"].y == pytest.approx(side.rows["hip"], abs=12)
    assert marks["bust_row"].y < marks["underbust_row"].y < marks["hip_row"].y


def test_the_nape_sits_behind_the_balance_line_above_the_shoulder() -> None:
    side = make_side()
    marks = as_dict(find_side_landmarks(side.mask, side.keypoints, "side_left", HEIGHTS, HEIGHT_CM))

    assert marks["nape"].x < side.balance_x
    assert 215 <= marks["nape"].y <= SHOULDER_Y


def test_the_hps_is_above_the_shoulder_keypoint() -> None:
    side = make_side()
    marks = as_dict(find_side_landmarks(side.mask, side.keypoints, "side_left", HEIGHTS, HEIGHT_CM))

    assert marks["hps"].y < SHOULDER_Y
    assert marks["head_row"].y < marks["hps"].y


def test_the_side_scale_is_the_sides_own_span() -> None:
    """Front and side are matched by height, so each view carries its own scale."""
    side = make_side()
    marks = as_dict(find_side_landmarks(side.mask, side.keypoints, "side_left", HEIGHTS, HEIGHT_CM))
    scale = view_scale(list(marks.values()), HEIGHT_CM)
    rows = np.flatnonzero(side.mask.any(axis=1))

    assert scale.floor_row == rows.max()
    assert scale.head_row == rows.min()
    assert scale.mm_per_px == pytest.approx(HEIGHT_CM * 10 / (rows.max() - rows.min()))


def test_landmarks_round_trip_through_json(tmp_path: Path) -> None:
    front = make_person()
    marks = find_front_landmarks(front.mask, front.keypoints, HEIGHT_CM)
    path = save_landmarks(marks, tmp_path / "session" / "landmarks.json", HEIGHT_CM)

    reloaded = load_landmarks(path)
    assert reloaded == marks
    assert path.name == "landmarks.json"


def test_the_stored_scale_matches_the_stored_rows(tmp_path: Path) -> None:
    front = make_person()
    marks = find_front_landmarks(front.mask, front.keypoints, HEIGHT_CM)
    path = save_landmarks(marks, tmp_path / "landmarks.json", HEIGHT_CM)

    stored = json.loads(path.read_text(encoding="utf-8"))["views"]["front"]
    assert stored == {
        "floor_row": view_scale(marks, HEIGHT_CM).floor_row,
        "head_row": view_scale(marks, HEIGHT_CM).head_row,
        "mm_per_px": pytest.approx(view_scale(marks, HEIGHT_CM).mm_per_px),
    }


def test_an_edited_landmark_survives_the_round_trip(tmp_path: Path) -> None:
    front = make_person()
    marks = find_front_landmarks(front.mask, front.keypoints, HEIGHT_CM)
    moved = [
        Landmark(mark.name, mark.view, mark.x, mark.y + 12, auto=False)
        if mark.name == "waist_row"
        else mark
        for mark in marks
    ]
    path = save_landmarks(moved, tmp_path / "landmarks.json", HEIGHT_CM)

    waist = as_dict(load_landmarks(path))["waist_row"]
    assert waist.auto is False
    assert waist.y == as_dict(marks)["waist_row"].y + 12


def test_shoulder_landmarks_follow_the_subject_not_the_image(front: object) -> None:
    """COCO sides are the subject's own.

    Naming landmarks by image side instead pairs each shoulder with the opposite
    arm in measure.py, which the GarmentCode bench showed as a 20 cm arm error.
    """
    from bodymeasure.posegate import L_SHOULDER, R_SHOULDER

    marks = as_dict(find_front_landmarks(front.mask, front.keypoints, HEIGHT_CM))
    for side, index in (("left", L_SHOULDER), ("right", R_SHOULDER)):
        keypoint_side = front.keypoints[index, 0] - front.torso_x
        assert (marks[f"shoulder_{side}"].x - front.torso_x) * keypoint_side > 0, side
        assert (marks[f"neck_base_{side}"].x - front.torso_x) * keypoint_side > 0, side


def test_the_shoulder_point_is_lateral_to_the_neck(front: object) -> None:
    """The neck-shoulder junction is a sharp corner and must not win the search."""
    marks = as_dict(find_front_landmarks(front.mask, front.keypoints, HEIGHT_CM))
    for side in ("left", "right"):
        neck = abs(marks[f"neck_base_{side}"].x - front.torso_x)
        shoulder = abs(marks[f"shoulder_{side}"].x - front.torso_x)
        assert shoulder > neck, side
