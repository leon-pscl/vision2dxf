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


def test_every_primary_schema_key_is_produced(result: MeasurementSet) -> None:
    """The 26 primary keys, plus extended ones that go to their own file (brief 3.3)."""
    assert set(load_schema()) <= set(result.body())
    assert set(result.body()) - set(load_schema()) == {
        "chest",
        "shoulder_length",
        "shoulder_length_left",
        "shoulder_length_right",
        "arm_length_left",
        "arm_length_right",
    }


def test_every_value_is_rounded_to_one_decimal(result: MeasurementSet) -> None:
    for name, value in result.body().items():
        if value is None:
            continue  # unavailable or unaccepted: see the status tests below
        assert value == round(value, 1), name


def test_the_internal_value_stays_unrounded_in_millimetres(result: MeasurementSet) -> None:
    """Brief 5: display is cm to one decimal; provenance keeps the real number."""
    waist = result.values["waist"]
    assert waist.unit == "mm"
    assert waist.value > 100.0
    assert waist.value_cm == round(waist.value / 10.0, 1)
    assert waist.record("waist")["value_mm"] == round(waist.value, 2)


def test_the_angles_are_degrees_not_centimetres(result: MeasurementSet) -> None:
    schema = load_schema()
    for name in ("shoulder_incl", "arm_pose_angle", "hip_inclination"):
        assert schema[name]["unit"] == "deg"
        assert result.values[name].unit == "deg"
        assert 0.0 <= result.values[name].value_cm <= 90.0


def test_the_height_is_the_number_the_tailor_measured(result: MeasurementSet) -> None:
    """Height is the input that sets the scale, not something derived from it."""
    entry = result.values["height"]
    assert entry.value_cm == HEIGHT_CM
    assert entry.source == "operator_input"
    assert entry.status == "manual_input"
    assert entry.auto is False
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


def test_a_population_ratio_is_offered_but_never_counted_as_acquired(
    result: MeasurementSet,
) -> None:
    """Brief 2: a demographic ratio needs an explicit operator acceptance."""
    for name, base, ratio in (
        ("bust_points", "bust", BUST_POINTS_RATIO),
        ("bum_points", "hips", BUM_POINTS_RATIO),
    ):
        entry = result.values[name]
        assert entry.source == "derived_pattern_parameter"
        assert entry.status == "estimated_pending_acceptance"
        assert entry.available is False, "it must block a pattern export until accepted"
        assert entry.warnings, "the operator has to be told where the number came from"
        assert entry.value_cm == pytest.approx(
            result.values[base].value_cm * ratio, abs=0.05
        )
    assert result.body()["bust_points"] is None, "an unaccepted estimate exports as null"


def test_provenance_names_the_views_each_value_came_from(result: MeasurementSet) -> None:
    assert result.values["height"].views == ["front"]
    assert result.values["head_l"].views == ["side_left"]
    assert result.values["waist"].views == ["front", "side_left"]
    assert result.values["bust_points"].views == []
    assert all(not entry.edited for entry in result.values.values())


def test_confidence_follows_the_schema(result: MeasurementSet) -> None:
    schema = load_schema()
    for name, entry in result.values.items():
        if name in schema:
            assert entry.confidence == schema[name]["confidence"], name


def test_a_girth_keeps_the_breadth_depth_and_rows_it_came_from(
    result: MeasurementSet,
) -> None:
    """Brief 3.1: the four circumferences must retain their intermediate values."""
    for name in ("bust", "waist", "hips", "underbust"):
        entry = result.values[name]
        assert entry.method == "montazerian_ellipse"
        assert entry.formula in ("ellipse_equation_1", "ellipse_equation_2")
        inputs = entry.inputs
        assert inputs["front_breadth_mm"] > 0
        assert inputs["side_depth_mm"] > 0
        assert inputs["side_depth_front_mm"] + inputs["side_depth_back_mm"] == pytest.approx(
            inputs["side_depth_mm"]
        )
        assert inputs["row_height_mm"] > 0
        assert inputs["front_mm_per_px"] > 0
        assert inputs["side_left_mm_per_px"] > 0, "every contributing view's scale"


def test_a_girth_matches_its_own_recorded_inputs(result: MeasurementSet) -> None:
    """The provenance record is enough to recompute the number from scratch."""
    from bodymeasure.geometry import montazerian_girth

    for name in ("bust", "waist", "hips"):
        entry = result.values[name]
        again, formula = montazerian_girth(
            entry.inputs["front_breadth_mm"], entry.inputs["side_depth_mm"]
        )
        assert again == pytest.approx(entry.value)
        assert formula == entry.formula


def test_left_and_right_are_kept_as_well_as_the_mean(result: MeasurementSet) -> None:
    """Brief 3.1: shoulder and arm length, left and right, plus the accepted mean."""
    for base in ("shoulder_length", "arm_length"):
        left = result.values[f"{base}_left"].value
        right = result.values[f"{base}_right"].value
        assert left > 0 and right > 0
        assert result.values[base].value == pytest.approx((left + right) / 2)


def test_a_row_off_the_body_is_unavailable_not_zero() -> None:
    """Brief 5: never replace a failed calculation with zero."""
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
    # Drag the waist row above the top of the head: there is no body there.
    moved = [
        replace(mark, y=-500.0, auto=False) if mark.name == "waist_row" else mark
        for mark in marks
    ]
    waist = measure(front.mask, moved, [view], front.keypoints, HEIGHT_CM).values["waist"]

    assert waist.value is None
    assert waist.status == "unavailable"
    assert waist.warnings, "an unavailable result carries its reason"
    assert waist.available is False


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
    assert "chest" not in changed, "the chest row is the armpit row, not the waist row"
    assert "waist" in changed
    assert "waist_back_width" in changed
    assert "hips_line" in changed
    assert not {"height", "head_l", "neck_w", "wrist", "arm_length"} & changed
