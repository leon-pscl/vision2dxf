"""The scale is the subject's height, not the station (changes 2, 3 and 7).

Two properties carry the whole design. First: a figure drawn at any size or
position measures the same, as long as the height entered matches what was
drawn - otherwise every measurement depends on how far the subject stood from the
lens, which is exactly what the physical calibration used to pin down. Second:
the scale really is that height, so a wrong height propagates to every
measurement in proportion.

``tests/synthetic/people.py`` and ``make_side`` are one body between them, which
matters here: front and side rows are matched by height above the floor, so the
two drawings have to agree about where the crotch is.
"""

import random

import pytest
from tests.synthetic.people import make_person, make_side, rescale

from bodymeasure.landmarks import (
    Landmark,
    as_dict,
    find_front_landmarks,
    find_side_landmarks,
    view_scale,
)
from bodymeasure.measure import MeasurementSet, SideView, measure

BASE_HEIGHT_CM = 164.5  # the drawn figure spans 1645 rows at 1 mm/px

# The shoulder point is the sharpest turn over a fixed 15 contour points
# (brief 5.5), so that window covers less of the shoulder when the subject stands
# further away. Resolution-dependent by construction, not a property of the scale.
# The bounds are the widest drift over the draws below, in cm or degrees.
# ponytail: make the window a fixed 15 mm instead of 15 pixels if a real station
# ever shows the shoulder point moving between two captures of one subject.
CURVATURE_WINDOW = {
    "arm_length": 5.0,
    "armscye_depth": 5.0,
    "shoulder_w": 5.0,
    "shoulder_incl": 3.5,
}

# One pixel of the drawing. Redrawing a silhouette moves the row a width is read
# at by a pixel or two, and the values are rounded to 0.1 cm on top, so nothing
# finer than a few pixels is observable. A scale that was actually wrong would move
# the girths by centimetres, so this still fails loudly if the scale stops being
# the subject's height.
ROW_TOLERANCE_CM = 0.5


def _front_heights(marks: list[Landmark], height_cm: float) -> dict[str, float]:
    """Return the front armpit, waist and crotch heights above the floor, in mm."""
    rows = as_dict(marks)
    scale = view_scale(marks, height_cm)
    return {
        name: scale.height_mm(rows[f"{name}_row"].y) for name in ("armpit", "waist", "crotch")
    }


def _pair(
    front_factor: float,
    front_offset: tuple[int, int],
    side_factor: float,
    side_offset: tuple[int, int],
    height_cm: float | None = None,
) -> MeasurementSet:
    """Measure the drawn figure after rescaling each view independently.

    The two views get their own size and their own scale, which is the point:
    front and side rows are matched by height above the floor, never by row
    number. A ``height_cm`` of ``None`` uses the height each drawing implies.
    """
    drawn_front, drawn_side = make_person(), make_side()
    front = rescale(drawn_front.mask, drawn_front.keypoints, front_factor, front_offset)
    side = rescale(drawn_side.mask, drawn_side.keypoints, side_factor, side_offset)
    front_cm = front.height_cm if height_cm is None else height_cm
    side_cm = side.height_cm if height_cm is None else height_cm

    front_marks = find_front_landmarks(front.mask, front.keypoints, front_cm)
    side_view = SideView(
        name="side_left",
        mask=side.mask,
        landmarks=as_dict(
            find_side_landmarks(
                side.mask,
                side.keypoints,
                "side_left",
                _front_heights(front_marks, front_cm),
                side_cm,
            )
        ),
        height_cm=side_cm,
    )
    return measure(front.mask, front_marks, [side_view], front.keypoints, front_cm)


@pytest.fixture(scope="module")
def baseline() -> MeasurementSet:
    return _pair(1.0, (0, 0), 1.0, (0, 0))


def _drawings() -> list[tuple[float, tuple[int, int], float, tuple[int, int]]]:
    """Return the sizes and offsets a station really produces, drawn at random.

    A subject steps towards or away from the lens and stands a little to one side
    of the frame centre; the two views are captured independently, so each gets
    its own draw.
    """
    rng = random.Random(11)
    return [
        (
            round(rng.uniform(0.85, 1.2), 2),
            (round(rng.uniform(-30, 30)), round(rng.uniform(-100, 180))),
            round(rng.uniform(0.85, 1.2), 2),
            (round(rng.uniform(-30, 30)), round(rng.uniform(-100, 180))),
        )
        for _ in range(8)
    ]


@pytest.mark.parametrize(
    ("front_factor", "front_offset", "side_factor", "side_offset"), _drawings()
)
def test_a_figure_at_another_size_measures_the_same(
    baseline: MeasurementSet,
    front_factor: float,
    front_offset: tuple[int, int],
    side_factor: float,
    side_offset: tuple[int, int],
) -> None:
    """No floor mark and no mm/px on the station: the answer does not move."""
    other = _pair(front_factor, front_offset, side_factor, side_offset)
    tolerance = {name: ROW_TOLERANCE_CM for name in baseline.body()}
    tolerance.update(CURVATURE_WINDOW)

    for name, value in baseline.body().items():
        assert other.body()[name] == pytest.approx(value, abs=tolerance[name]), name


def test_the_implied_height_tracks_the_drawing() -> None:
    """Half the pixels means half the millimetres, so the same 164.5 cm stands up."""
    smaller = _pair(0.5, (0, 0), 0.5, (0, 0))

    assert smaller.body()["height"] == pytest.approx(BASE_HEIGHT_CM, abs=1.0)
    assert smaller.body()["waist"] == pytest.approx(
        _pair(1.0, (0, 0), 1.0, (0, 0)).body()["waist"], abs=0.2
    )


def test_one_centimetre_of_height_error_stretches_the_body(baseline: MeasurementSet) -> None:
    """A +1 cm height grows every length by 1/height.

    The scale is checked exactly, and tightly: nothing rounds there. The waist is
    checked end to end, where the 0.1 cm the output is rounded to is the finest
    thing observable - a 0.32 cm expected change cannot be pinned to 10% of itself
    on a 0.1 cm grid, so the bound is the larger of 10% and one step.
    """
    stretched = _pair(1.0, (0, 0), 1.0, (0, 0), height_cm=BASE_HEIGHT_CM + 1.0)
    marks = find_front_landmarks(make_person().mask, make_person().keypoints, BASE_HEIGHT_CM)

    assert stretched.body()["height"] == BASE_HEIGHT_CM + 1.0
    assert view_scale(marks, BASE_HEIGHT_CM + 1.0).mm_per_px == pytest.approx(
        view_scale(marks, BASE_HEIGHT_CM).mm_per_px * (1 + 1 / BASE_HEIGHT_CM)
    )

    waist = baseline.body()["waist"]
    expected = waist / BASE_HEIGHT_CM  # cm: every millimetre grows by 1/height
    assert stretched.body()["waist"] - waist == pytest.approx(expected, rel=0.10, abs=0.1)