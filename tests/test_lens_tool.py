"""The optional lens tool: does it recover a camera it was shown.

Rendered boards stand in for the ones the monitor shows, because the tool's
physical size comes from the screen's millimetre width and that cannot be faked
on a test machine; what matters here is that the fit finds back the camera that
projected the board.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from tests.synthetic.boards import PATTERN, calibration_frames
from tools.calibrate_lens_on_monitor import (
    compute_intrinsics,
    draw_board,
    find_corners,
    object_points,
    pitch_px,
    poses,
    screen_size,
    square_mm,
    write_calibration,
)

from bodymeasure.config import Calibration, load_calibration

SIZE = (1920, 1080)
MONITOR_WIDTH_MM = 530.0  # a 24-inch 16:9 panel


def test_the_tool_imports_without_a_camera_or_a_monitor() -> None:
    """It is optional: importing it must not need the hardware."""
    assert screen_size()[0] > 0


def test_the_drawn_square_is_the_size_it_reports() -> None:
    """The fit is only as good as this number, so it must follow from the screen."""
    pitch = pitch_px(SIZE, MONITOR_WIDTH_MM)
    assert pitch == pytest.approx(square_mm(SIZE, MONITOR_WIDTH_MM) * SIZE[0] / MONITOR_WIDTH_MM)
    # 30 mm would be 109 px, but the board has to fit while tilted, and it does not.
    assert pitch <= 30.0 * SIZE[0] / MONITOR_WIDTH_MM


def test_a_wider_screen_draws_a_smaller_square() -> None:
    """The same pixels over more millimetres is a smaller square on the wall."""
    assert pitch_px(SIZE, 1060.0) < pitch_px(SIZE, 530.0)


def test_a_board_drawn_on_the_screen_is_found_in_it() -> None:
    board = draw_board(SIZE, pitch_px(SIZE, MONITOR_WIDTH_MM), 0.0, (0, 0))
    assert board.shape == (SIZE[1], SIZE[0], 3)
    assert find_corners(board) is not None
    assert object_points(square_mm(SIZE, MONITOR_WIDTH_MM)).shape == (
        PATTERN[0] * PATTERN[1],
        3,
    )


def test_a_tilted_board_is_still_found() -> None:
    """Every pose the tool shows has to be detectable, or the fit has nothing."""
    pitch = pitch_px(SIZE, MONITOR_WIDTH_MM)
    for angle, offset in poses():
        assert find_corners(draw_board(SIZE, pitch, angle, offset)) is not None, angle


def test_every_pose_fits_on_the_screen_it_is_drawn_on() -> None:
    """A board clipped by its own tilt has no corners, and the pose is wasted."""
    for size in ((1280, 1024), (1920, 1080), (2560, 1440)):
        pitch = pitch_px(size, MONITOR_WIDTH_MM)
        for angle, offset in poses():
            board = draw_board(size, pitch, angle, offset)
            inked = (board < 128).any(axis=2)
            assert inked.any(), (size, angle, offset)
            assert not inked[0].any() and not inked[-1].any(), (size, angle, offset)
            assert not inked[:, 0].any() and not inked[:, -1].any(), (size, angle, offset)


def test_the_fit_recovers_the_camera_that_projected_the_board() -> None:
    frames = calibration_frames((1200, 900))
    intrinsics = compute_intrinsics(frames, square_mm=25.0)

    assert intrinsics.frames_used >= 10
    assert intrinsics.rms_px < 1.0
    assert intrinsics.camera_matrix[0][0] == pytest.approx(1400.0, rel=0.05)
    assert intrinsics.camera_matrix[1][1] == pytest.approx(1400.0, rel=0.05)


def test_too_few_views_are_refused_rather_than_fitted() -> None:
    blank = np.full((900, 1200, 3), 255, dtype=np.uint8)
    with pytest.raises(ValueError, match="need 10"):
        compute_intrinsics([blank, blank, blank], square_mm=25.0)


def test_frames_of_different_sizes_are_refused() -> None:
    with pytest.raises(ValueError, match="same size"):
        compute_intrinsics(
            [np.zeros((900, 1200, 3), np.uint8), np.zeros((600, 800, 3), np.uint8)],
            square_mm=25.0,
        )


def test_writing_the_calibration_leaves_only_the_lens_model(tmp_path: Path) -> None:
    frames = calibration_frames((1200, 900))
    intrinsics = compute_intrinsics(frames, square_mm=25.0)
    path = write_calibration(intrinsics, rotation=90, path=tmp_path / "c.yaml")

    saved = load_calibration(path)
    assert isinstance(saved, Calibration)
    assert saved.rotation == 90
    assert saved.has_lens_model
    assert not hasattr(saved, "mm_per_px_x")
    assert yaml.safe_load(path.read_text("utf-8")).keys() == {
        "camera_matrix",
        "dist_coeffs",
        "rotation",
    }
    new_matrix, _ = cv2.getOptimalNewCameraMatrix(
        np.array(saved.camera_matrix), np.array(saved.dist_coeffs), (1200, 900), 0
    )
    assert new_matrix.shape == (3, 3)