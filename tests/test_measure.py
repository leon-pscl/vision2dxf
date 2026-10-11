"""The 26 GarmentCode keys, computed from synthetic silhouettes."""

import math

import pytest
from tests.synthetic.people import FLOOR_ROW, make_person, make_side

from bodymeasure.config import Calibration, load_schema
from bodymeasure.landmarks import as_dict, find_front_landmarks, find_side_landmarks
from bodymeasure.measure import (
    BUM_POINTS_RATIO,
    BUST_POINTS_RATIO,
    MeasurementSet,
    SideView,
    measure,
)

CALIBRATION = Calibration(
    camera_matrix=[[1400.0, 0.0, 450.0], [0.0, 1400.0, 900.0], [0.0, 0.0, 1.0]],
    dist_coeffs=[0.0, 0.0, 0.0, 0.0, 0.0],
    image_size=[900, 1800],
    rotation=90,
    mm_per_px_x=1.0,
    mm_per_px_y=1.0,
    floor_row=FLOOR_ROW,
    calibrated_at="2026-10-11T10:00:00+08:00",
)


def _measure(side_names: tuple[str, ...] = ("side_left",)) -> MeasurementSet:
    front = make_person()
    front_marks = find_front_landmarks(front.mask, front.keypoints, CALIBRATION.mm_per_px_x)
    rows = as_dict(front_marks)
    front_rows = {
        "armpit": int(rows["armpit_row"].y),
        "waist": int(rows["waist_row"].y),
        "crotch": int(rows["crotch_row"].y),
    }
    sides = []
    for name in side_names:
        side = make_side(name)
        sides.append(
            SideView(
                name=name,
                mask=side.mask,
                landmarks=as_dict(
                    find_side_landmarks(side.mask, side.keypoints, name, front_rows)
                ),
            )
        )
    return measure(front.mask, front_marks, sides, front.keypoints, CALIBRATION)


@pytest.fixture(scope="module")
def result() -> MeasurementSet:
    return _measure()


def test_exactly_the_26_schema_keys_are_produced(result: MeasurementSet) -> None:
    assert set(result.body()) == set(load_schema())
    assert len(result.body()) == 26


def test_every_value_is_rounded_to_one_decimal(result: MeasurementSet) -> None:
    for name, value in result.body().items():
        assert value == round(value, 1), name


def test_the_angles_are_degrees_not_centimetres(result: MeasurementSet) -> None:
    schema = load_schema()
    for name in ("shoulder_incl", "arm_pose_angle", "hip_inclination"):
        assert schema[name]["unit"] == "deg"
        assert 0.0 <= result.values[name].value_cm <= 90.0


def test_the_height_matches_the_drawn_figure(result: MeasurementSet) -> None:
    """The figure's head top is row 105 and the floor is row 1750, at 1 mm/px."""
    assert result.values["height"].value_cm == pytest.approx(164.5, abs=0.5)


def test_the_arm_pose_angle_reports_the_drawn_a_pose(result: MeasurementSet) -> None:
    """Arms are drawn 30 degrees from vertical, so 60 degrees below horizontal."""
    assert result.values["arm_pose_angle"].value_cm == pytest.approx(60.0, abs=2.0)


def test_girths_are_ordered_like_a_body(result: MeasurementSet) -> None:
    body = result.body()
    assert body["underbust"] < body["bust"]
    assert body["waist"] < body["hips"]
    assert body["wrist"] < body["leg_circ"] < body["hips"]


def test_a_back_arc_is_part_of_its_girth(result: MeasurementSet) -> None:
    body = result.body()
    for girth, arc in (
        ("bust", "back_width"),
        ("waist", "waist_back_width"),
        ("hips", "hip_back_width"),
    ):
        assert 0 < body[arc] < body[girth], arc


def test_vertical_differences_are_positive(result: MeasurementSet) -> None:
    body = result.body()
    assert body["hips_line"] > 0, "the hip row sits below the waist row"
    assert body["crotch_hip_diff"] > 0, "the crotch sits below the hip row"
    assert body["vert_bust_line"] > 0, "the bust sits below the nape"


def test_the_tailor_entered_values_use_the_garmentcode_ratios(result: MeasurementSet) -> None:
    body = result.body()
    assert result.values["bust_points"].source == "default_ratio"
    assert result.values["bust_points"].confidence == "estimated"
    assert body["bust_points"] == pytest.approx(body["bust"] * BUST_POINTS_RATIO, abs=0.05)
    assert body["bum_points"] == pytest.approx(body["hips"] * BUM_POINTS_RATIO, abs=0.05)


def test_provenance_names_the_views_each_value_came_from(result: MeasurementSet) -> None:
    assert result.values["height"].views == ["front"]
    assert result.values["head_l"].views == ["side_left"]
    assert result.values["waist"].views == ["front", "side_left"]
    assert result.values["bust_points"].views == []
    assert all(not entry.edited for entry in result.values.values())


def test_confidence_follows_the_schema(result: MeasurementSet) -> None:
    schema = load_schema()
    for name, entry in result.values.items():
        assert entry.confidence == schema[name]["confidence"], name


def test_the_wrist_is_a_circle_around_the_measured_width(result: MeasurementSet) -> None:
    """Brief 5.7: wrist = pi x the narrowest wrist width."""
    assert result.values["wrist"].value_cm == pytest.approx(math.pi * 5.5, abs=1.0)


def test_two_side_views_are_averaged_without_warning() -> None:
    both = _measure(("side_left", "side_right"))
    one = _measure(("side_left",))
    assert both.body()["waist"] == pytest.approx(one.body()["waist"], abs=0.1)
    assert both.warnings == [], "identical side views cannot disagree"


def test_a_measurement_needs_at_least_one_side_view() -> None:
    front = make_person()
    marks = find_front_landmarks(front.mask, front.keypoints, CALIBRATION.mm_per_px_x)
    with pytest.raises(ValueError, match="at least one side view"):
        measure(front.mask, marks, [], front.keypoints, CALIBRATION)


def test_missing_front_landmarks_are_named() -> None:
    front = make_person()
    marks = find_front_landmarks(front.mask, front.keypoints, CALIBRATION.mm_per_px_x)
    without_waist = [mark for mark in marks if mark.name != "waist_row"]
    side = make_side()
    view = SideView(
        name="side_left",
        mask=side.mask,
        landmarks=as_dict(
            find_side_landmarks(
                side.mask,
                side.keypoints,
                "side_left",
                {"armpit": 430, "waist": 760, "crotch": 1000},
            )
        ),
    )
    with pytest.raises(ValueError, match="waist_row"):
        measure(front.mask, without_waist, [view], front.keypoints, CALIBRATION)


def test_moving_the_waist_row_changes_only_what_depends_on_it() -> None:
    """Brief 8: dragging a landmark changes only the measurements that depend on it."""
    from dataclasses import replace

    front = make_person()
    marks = find_front_landmarks(front.mask, front.keypoints, CALIBRATION.mm_per_px_x)
    rows = as_dict(marks)
    front_rows = {
        "armpit": int(rows["armpit_row"].y),
        "waist": int(rows["waist_row"].y),
        "crotch": int(rows["crotch_row"].y),
    }
    side = make_side()
    view = SideView(
        name="side_left",
        mask=side.mask,
        landmarks=as_dict(find_side_landmarks(side.mask, side.keypoints, "side_left", front_rows)),
    )
    before = measure(front.mask, marks, [view], front.keypoints, CALIBRATION).body()

    moved = [
        replace(mark, y=mark.y - 40, auto=False) if mark.name == "waist_row" else mark
        for mark in marks
    ]
    after = measure(front.mask, moved, [view], front.keypoints, CALIBRATION).body()

    changed = {name for name in before if before[name] != after[name]}
    assert "waist" in changed
    assert "waist_back_width" in changed
    assert "hips_line" in changed
    assert not {"height", "head_l", "neck_w", "wrist", "arm_length"} & changed
