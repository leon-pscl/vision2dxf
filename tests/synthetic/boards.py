"""Render checkerboards whose camera matrix and scale are known exactly.

The brief asks M2 to be tested on saved sample frames. There are none in the
repository, so the boards are rendered instead: a known camera matrix projects a
known board, and the test asserts the calibration recovers what was projected.
"""

from __future__ import annotations

import cv2
import numpy as np

PATTERN = (9, 6)  # inner corners (columns, rows)
SQUARE_MM = 25.0


def camera_matrix(image_size: tuple[int, int], focal_px: float = 1400.0) -> np.ndarray:
    """Return a plausible pinhole camera matrix centred on the image."""
    width, height = image_size
    return np.array(
        [[focal_px, 0.0, width / 2], [0.0, focal_px, height / 2], [0.0, 0.0, 1.0]],
        dtype=float,
    )


def _square_corners(col: int, row: int) -> np.ndarray:
    """Return the 3D corners (mm, z=0) of one board square."""
    return np.array(
        [
            [col, row, 0.0],
            [col + 1, row, 0.0],
            [col + 1, row + 1, 0.0],
            [col, row + 1, 0.0],
        ],
        dtype=float,
    ) * SQUARE_MM


def render_board(
    image_size: tuple[int, int] = (1200, 900),
    rvec: tuple[float, float, float] = (0.0, 0.0, 0.0),
    tvec: tuple[float, float, float] = (-100.0, -75.0, 700.0),
    focal_px: float = 1400.0,
) -> np.ndarray:
    """Project the board at the given pose onto a white frame.

    ``PATTERN`` counts inner corners, so the drawn board is one square larger in
    each direction and carries a white margin that corner detection needs.
    """
    width, height = image_size
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    matrix = camera_matrix(image_size, focal_px)
    cols, rows = PATTERN[0] + 1, PATTERN[1] + 1

    for row in range(rows):
        for col in range(cols):
            if (row + col) % 2:
                continue
            projected, _ = cv2.projectPoints(
                _square_corners(col, row),
                np.array(rvec, dtype=float),
                np.array(tvec, dtype=float),
                matrix,
                np.zeros(5),
            )
            cv2.fillConvexPoly(image, projected.reshape(-1, 2).round().astype(np.int32), (0, 0, 0))
    return image


def render_upright_board(
    px_per_square: int = 60,
    image_size: tuple[int, int] = (1200, 900),
    top_left: tuple[int, int] = (300, 150),
) -> np.ndarray:
    """Draw a fronto-parallel board at an exact pixel pitch.

    No projection, so ``SQUARE_MM / px_per_square`` is the true mm/px on both
    axes and the lowest inner-corner row is known to the pixel.
    """
    width, height = image_size
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    x0, y0 = top_left
    for row in range(PATTERN[1] + 1):
        for col in range(PATTERN[0] + 1):
            if (row + col) % 2:
                continue
            x, y = x0 + col * px_per_square, y0 + row * px_per_square
            image[y : y + px_per_square, x : x + px_per_square] = 0
    return image


def lowest_corner_row(px_per_square: int = 60, top_left: tuple[int, int] = (300, 150)) -> int:
    """Return the image row of the lowest inner corner of :func:`render_upright_board`."""
    return top_left[1] + PATTERN[1] * px_per_square


# Angles, distances and positions an operator would actually cover. Varying the
# distance matters: boards all at one depth leave the distortion terms badly
# conditioned (k3 then absorbs the rasterisation noise).
_RVECS = [
    (0.0, 0.0, 0.0),
    (0.3, 0.0, 0.0),
    (-0.3, 0.0, 0.0),
    (0.0, 0.3, 0.0),
    (0.0, -0.3, 0.0),
    (0.2, 0.2, 0.05),
    (-0.2, 0.2, -0.05),
    (0.2, -0.2, 0.05),
    (-0.2, -0.2, -0.05),
    (0.1, 0.35, 0.1),
    (-0.35, -0.1, -0.1),
    (0.35, 0.1, 0.0),
    (0.0, 0.15, 0.2),
    (-0.15, 0.0, -0.2),
]
POSES: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = [
    (
        rvec,
        (
            -100.0 + 55.0 * (index % 3 - 1),
            -75.0 + 45.0 * (index % 4 - 1.5),
            550.0 + 90.0 * (index % 5),
        ),
    )
    for index, rvec in enumerate(_RVECS)
]


def calibration_frames(image_size: tuple[int, int] = (1200, 900)) -> list[np.ndarray]:
    """Return one rendered frame per pose in :data:`POSES`."""
    return [render_board(image_size, rvec=rvec, tvec=tvec) for rvec, tvec in POSES]
