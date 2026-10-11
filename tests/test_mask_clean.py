"""Mask clean-up keeps the body, drops noise, and preserves the gaps that matter."""

import numpy as np
from tests.synthetic.people import make_person

from bodymeasure.mask_clean import clean_mask, fill_small_holes, keep_largest_component


def test_keep_largest_component_drops_the_smaller_blob() -> None:
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[20:120, 20:120] = 1  # 10000 px
    mask[160:180, 160:180] = 1  # 400 px
    kept = keep_largest_component(mask)
    assert kept[50, 50] == 1
    assert kept[170, 170] == 0


def test_keep_largest_component_handles_an_empty_mask() -> None:
    assert keep_largest_component(np.zeros((10, 10), dtype=np.uint8)).sum() == 0


def test_fill_small_holes_fills_the_small_one_only() -> None:
    mask = np.ones((300, 300), dtype=np.uint8)
    mask[50:60, 50:60] = 0  # 100 px: a hole
    mask[150:200, 150:200] = 0  # 2500 px: a real gap
    filled = fill_small_holes(mask, max_area_px=500)
    assert filled[55, 55] == 1
    assert filled[175, 175] == 0


def test_fill_small_holes_never_fills_the_background() -> None:
    """A background region touching the edge is outside the body, not a hole."""
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[10:30, 10:30] = 1
    filled = fill_small_holes(mask, max_area_px=10_000)
    assert filled[0, 0] == 0
    assert filled.sum() == 400


def test_clean_mask_removes_speckle_and_keeps_the_leg_gap() -> None:
    person = make_person()
    noisy = person.mask.astype(np.uint8).copy()
    rng = np.random.default_rng(7)
    for _ in range(60):
        y, x = rng.integers(0, 1700), rng.integers(0, 880)
        noisy[y : y + 3, x : x + 3] = 1  # speckle outside the body
    noisy[500:505, 440:445] = 0  # a pinhole inside the torso

    cleaned = clean_mask(noisy, kernel_px=5, hole_fill_max_area_px=500)

    assert cleaned.dtype == bool
    assert cleaned[person.rows["waist"], 450]
    assert cleaned[502, 442], "pinhole should be filled"
    gap_row = (person.rows["hip"] + person.rows["knee"]) // 2
    assert not cleaned[gap_row, 450], "the gap between the legs must stay background"
    assert cleaned.sum() < noisy.sum() + 500
