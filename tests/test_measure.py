"""The 26 GarmentCode keys, computed from synthetic silhouettes."""

import math

import pytest
from tests.synthetic.people import make_person, make_side

from bodymeasure.config import load_schema
from bodymeasure.landmarks import as_dict, find_front_landmarks, find_side_landmarks, view_scale
from bodymeasure.measure import (
    BUM_POINTS_RATIO,
    BUST_POINTS_RATIO,
    MeasurementSet,
    SideView,
    measure,
)

HEIGHT_CM = 164.5  # the drawn figure spans 1645 rows at 1 mm/px


def _measure(
    side_names: tuple[str, ...] = ("side_left",), height_cm: float = HEIGHT_CM
) -> MeasurementSet:
    """Return the measurements of the drawn figure, front plus the given sides."""
    front = make_person()
    front_marks = find_front_landmarks(front.mask, front.keypoints, height_cm)
    rows = as_dict(front_marks)
    scale = view_scale(front_marks, height_cm)
    front_heights = {
        name: scale.height_mm(rows[f"{name}_row"].y)
        for name in ("armpit", "waist", "crotch")
    }
    sides = []
    for name in side_names:
        side = make_side(name)
        sides.append(
            SideView(
                name=name,
                mask=side.mask,
                landmarks=as_dict(
                    find_side_landmarks(side.mask, side.keypoints, name, front_heights, height_cm)
                ),
                height_cm=height_cm,
            )
        )
    return measure(front.mask, front_marks, sides, front.keypoints, height_cm)


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


def test_the_height_is_the_number_the_tailor_measured(result: MeasurementSet) -> None:
    """Height is the input that sets the scale, not something derived from it."""
    entry = result.values["height"]
    assert entry.value_cm == HEIGHT_CM
    assert entry.source == "tailor_input"
    assert entry.confidence == "high"


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
    marks = find_front_landmarks(front.mask, front.keypoints, HEIGHT_CM)
    with pytest.raises(ValueError, match="at least one side view"):
        measure(front.mask, marks, [], front.keypoints, HEIGHT_CM)


def test_missing_front_landmarks_are_named() -> None:
    front = make_person()
    marks = find_front_landmarks(front.mask, front.keypoints, HEIGHT_CM)
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
                {"armpit": 1320.0, "waist": 990.0, "crotch": 750.0},
                HEIGHT_CM,
            )
        ),
        height_cm=HEIGHT_CM,
    )
    with pytest.raises(ValueError, match="waist_row"):
        measure(front.mask, without_waist, [view], front.keypoints, HEIGHT_CM)


def test_moving_the_waist_row_changes_only_what_depends_on_it() -> None:
    """Brief 8: dragging a landmark changes only the measurements that depend on it."""
    from dataclasses import replace

    front = make_person()
    marks = find_front_landmarks(front.mask, front.keypoints, HEIGHT_CM)
    rows = as_dict(marks)
    scale = view_scale(marks, HEIGHT_CM)
    front_heights = {
        name: scale.height_mm(rows[f"{name}_row"].y)
        for name in ("armpit", "waist", "crotch")
    }
    side = make_side()
    view = SideView(
        name="side_left",
        mask=side.mask,
        landmarks=as_dict(
            find_side_landmarks(side.mask, side.keypoints, "side_left", front_heights, HEIGHT_CM)
        ),
        height_cm=HEIGHT_CM,
    )
    before = measure(front.mask, marks, [view], front.keypoints, HEIGHT_CM).body()

    moved = [
        replace(mark, y=mark.y - 40, auto=False) if mark.name == "waist_row" else mark
        for mark in marks
    ]
    after = measure(front.mask, moved, [view], front.keypoints, HEIGHT_CM).body()

    changed = {name for name in before if before[name] != after[name]}
    assert "waist" in changed
    assert "waist_back_width" in changed
    assert "hips_line" in changed
    assert not {"height", "head_l", "neck_w", "wrist", "arm_length"} & changed
