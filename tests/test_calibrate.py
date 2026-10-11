"""Calibration against rendered checkerboards with a known camera and scale."""

import math
from pathlib import Path

import numpy as np
import pytest
from tests.synthetic.boards import (
    PATTERN,
    SQUARE_MM,
    calibration_frames,
    camera_matrix,
    lowest_corner_row,
    render_board,
    render_upright_board,
)

from bodymeasure.calibrate import (
    Intrinsics,
    PlaneScale,
    compute_intrinsics,
    compute_plane_scale,
    find_corners,
    load_intrinsics,
    save_calibration,
    save_intrinsics,
)
from bodymeasure.config import Calibration, load_calibration
from bodymeasure.preprocess import decode_image, prepare, rotate

IMAGE_SIZE = (1200, 900)


def _frames() -> list[np.ndarray]:
    return calibration_frames(IMAGE_SIZE)


def test_find_corners_finds_the_whole_pattern() -> None:
    corners = find_corners(render_board(IMAGE_SIZE), PATTERN)
    assert corners is not None
    assert corners.shape == (PATTERN[0] * PATTERN[1], 2)


def test_find_corners_returns_none_on_a_blank_frame() -> None:
    assert find_corners(np.full((900, 1200, 3), 255, dtype=np.uint8), PATTERN) is None


def test_compute_intrinsics_recovers_the_rendered_camera() -> None:
    intrinsics = compute_intrinsics(_frames(), PATTERN, SQUARE_MM)
    truth = camera_matrix(IMAGE_SIZE)
    recovered = np.array(intrinsics.camera_matrix)

    assert intrinsics.frames_used >= 10
    assert intrinsics.image_size == list(IMAGE_SIZE)
    assert intrinsics.rms_px < 1.0
    assert recovered[0, 0] == pytest.approx(truth[0, 0], rel=0.02)
    assert recovered[1, 1] == pytest.approx(truth[1, 1], rel=0.02)
    assert recovered[0, 2] == pytest.approx(truth[0, 2], rel=0.05)
    assert recovered[1, 2] == pytest.approx(truth[1, 2], rel=0.05)
    # The renderer has no tangential distortion, so p1 and p2 must stay at zero.
    # k3 is deliberately not asserted small: noiseless synthetic boards leave the
    # third radial term free and it absorbs the rasterisation residual.
    assert abs(intrinsics.dist_coeffs[2]) < 0.01
    assert abs(intrinsics.dist_coeffs[3]) < 0.01


def test_compute_intrinsics_rejects_too_few_boards() -> None:
    blank = np.full((900, 1200, 3), 255, dtype=np.uint8)
    with pytest.raises(ValueError, match="need 10"):
        compute_intrinsics([blank] * 12, PATTERN, SQUARE_MM)


def test_compute_intrinsics_rejects_mixed_frame_sizes() -> None:
    frames = [render_board(IMAGE_SIZE), render_board((800, 600))]
    with pytest.raises(ValueError, match="same size"):
        compute_intrinsics(frames, PATTERN, SQUARE_MM)


def test_compute_intrinsics_rejects_no_frames() -> None:
    with pytest.raises(ValueError, match="no frames"):
        compute_intrinsics([], PATTERN, SQUARE_MM)


@pytest.mark.parametrize("px_per_square", [40, 60, 75])
def test_plane_scale_recovers_the_drawn_pitch(px_per_square: int) -> None:
    board = render_upright_board(px_per_square=px_per_square)
    scale = compute_plane_scale(board, PATTERN, SQUARE_MM)
    expected = SQUARE_MM / px_per_square

    assert scale.mm_per_px_x == pytest.approx(expected, rel=0.01)
    assert scale.mm_per_px_y == pytest.approx(expected, rel=0.01)
    assert scale.corners_found == PATTERN[0] * PATTERN[1]
    assert abs(scale.floor_row - lowest_corner_row(px_per_square)) <= 1


def test_plane_scale_puts_the_floor_below_the_board_by_the_margin() -> None:
    board = render_upright_board(px_per_square=60)
    flush = compute_plane_scale(board, PATTERN, SQUARE_MM)
    raised = compute_plane_scale(board, PATTERN, SQUARE_MM, bottom_margin_mm=100.0)
    expected_px = 100.0 / flush.mm_per_px_y

    assert raised.floor_row - flush.floor_row == pytest.approx(expected_px, abs=2)


def test_plane_scale_reports_a_missing_board() -> None:
    blank = np.full((900, 1200, 3), 255, dtype=np.uint8)
    with pytest.raises(ValueError, match="checkerboard not found"):
        compute_plane_scale(blank, PATTERN, SQUARE_MM)


def test_intrinsics_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "intrinsics.yaml"
    assert load_intrinsics(path) is None

    intrinsics = compute_intrinsics(_frames(), PATTERN, SQUARE_MM)
    save_intrinsics(intrinsics, path)
    assert load_intrinsics(path) == intrinsics


def test_save_calibration_writes_the_brief_contract(tmp_path: Path) -> None:
    intrinsics = Intrinsics(
        camera_matrix=[[1400.0, 0.0, 600.0], [0.0, 1400.0, 450.0], [0.0, 0.0, 1.0]],
        dist_coeffs=[0.0, 0.0, 0.0, 0.0, 0.0],
        image_size=[1200, 900],
        rms_px=0.21,
        frames_used=12,
    )
    scale = PlaneScale(mm_per_px_x=1.04, mm_per_px_y=1.04, floor_row=1880, corners_found=54)
    path = tmp_path / "calibration.yaml"

    written = save_calibration(intrinsics, scale, rotation=90, path=path)
    reloaded = load_calibration(path)

    assert isinstance(reloaded, Calibration)
    assert reloaded == written
    assert reloaded.floor_row == 1880
    assert reloaded.rotation == 90
    assert reloaded.calibrated_at.startswith(str(written.calibrated_at)[:4])


def test_rotation_turns_the_frame_and_rejects_odd_angles() -> None:
    frame = np.zeros((900, 1200, 3), dtype=np.uint8)
    assert rotate(frame, 90).shape[:2] == (1200, 900)
    assert rotate(frame, 180).shape[:2] == (900, 1200)
    assert rotate(frame, 0) is frame
    with pytest.raises(ValueError, match="0, 90, 180 or 270"):
        rotate(frame, 45)


def test_prepare_rotates_then_undistorts() -> None:
    calibration = Calibration(
        camera_matrix=[[1400.0, 0.0, 450.0], [0.0, 1400.0, 600.0], [0.0, 0.0, 1.0]],
        dist_coeffs=[-0.2, 0.05, 0.0, 0.0, 0.0],
        image_size=[900, 1200],
        rotation=90,
        mm_per_px_x=1.04,
        mm_per_px_y=1.04,
        floor_row=1100,
        calibrated_at="2026-10-11T10:00:00+08:00",
    )
    prepared = prepare(render_board((1200, 900)), calibration)
    assert prepared.shape[:2] == (1200, 900)


def test_decode_image_round_trips_and_rejects_junk() -> None:
    import cv2

    board = render_upright_board()
    encoded = cv2.imencode(".png", board)[1].tobytes()
    assert decode_image(encoded).shape == board.shape
    with pytest.raises(ValueError, match="not a readable image"):
        decode_image(b"definitely not a png")


def test_scale_is_consistent_with_the_geometry_height_rule() -> None:
    """A point one square above the lowest corner is one square above the floor."""
    from bodymeasure.geometry import height_of_row

    board = render_upright_board(px_per_square=60)
    scale = compute_plane_scale(board, PATTERN, SQUARE_MM)
    row = lowest_corner_row(60) - 60
    assert height_of_row(row, scale.floor_row, scale.mm_per_px_y) == pytest.approx(
        SQUARE_MM, abs=math.ceil(scale.mm_per_px_y)
    )


def test_undistorting_with_the_fit_keeps_the_board_grid_uniform() -> None:
    """The property that matters downstream: undistort must not warp the scale."""
    import cv2

    intrinsics = compute_intrinsics(_frames(), PATTERN, SQUARE_MM)
    undistorted = cv2.undistort(
        render_board(IMAGE_SIZE),
        np.array(intrinsics.camera_matrix),
        np.array(intrinsics.dist_coeffs),
    )
    corners = find_corners(undistorted, PATTERN)
    assert corners is not None

    grid = corners.reshape(PATTERN[1], PATTERN[0], 2)
    dx = np.diff(grid[:, :, 0], axis=1)
    dy = np.diff(grid[:, :, 1], axis=0)
    assert abs(dx.std() / dx.mean()) < 0.005
    assert abs(dy.std() / dy.mean()) < 0.005


def test_plane_scale_survives_a_rotated_board() -> None:
    """The station rotates frames, so the board can arrive on its side."""
    import cv2

    upright = render_upright_board(px_per_square=60)
    straight = compute_plane_scale(upright, PATTERN, SQUARE_MM)
    sideways = compute_plane_scale(
        cv2.rotate(upright, cv2.ROTATE_90_CLOCKWISE), PATTERN, SQUARE_MM
    )

    assert sideways.mm_per_px_x == pytest.approx(straight.mm_per_px_x, rel=0.01)
    assert sideways.mm_per_px_y == pytest.approx(straight.mm_per_px_y, rel=0.01)
