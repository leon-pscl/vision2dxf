"""Checkerboard intrinsics and body-plane scale (brief M2).

Two steps, run from the calibration page:

1. 15-20 views of a checkerboard give the camera matrix and distortion
   coefficients. They are parked in ``data/calib/intrinsics.yaml``.
2. One frame of a board held upright on the floor mark gives the scale at the
   body plane and the image row of the floor. Together with step 1 this is
   written to ``config/calibration.yaml``.

Frames arriving here must already be rotated (brief 4: ``image_size`` is the
size after rotation).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from bodymeasure.config import CONFIG_DIR, ROOT, Calibration, load_app_config

MIN_FRAMES = 10
_COLOR_NDIM = 3  # a BGR frame, as opposed to an already-grey one
_CORNER_FLAGS = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY


@dataclass(frozen=True)
class Intrinsics:
    """Camera matrix and distortion, with the fit quality the page displays."""

    camera_matrix: list[list[float]]
    dist_coeffs: list[float]
    image_size: list[int]
    rms_px: float
    frames_used: int


@dataclass(frozen=True)
class PlaneScale:
    """Scale at the body plane and the floor row, from the upright board."""

    mm_per_px_x: float
    mm_per_px_y: float
    floor_row: int
    corners_found: int


def find_corners(image: np.ndarray, pattern_size: tuple[int, int]) -> np.ndarray | None:
    """Return the inner checkerboard corners as ``(N, 2)`` px, or ``None`` if not found.

    ``pattern_size`` is ``(columns, rows)`` of *inner* corners, so a board of
    10x7 squares is ``(9, 6)``.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == _COLOR_NDIM else image
    found, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags=_CORNER_FLAGS)
    if not found:
        return None
    return corners.reshape(-1, 2).astype(float)


def _object_points(pattern_size: tuple[int, int], square_mm: float) -> np.ndarray:
    """Return the board's corner coordinates in mm, z = 0."""
    cols, rows = pattern_size
    grid = np.zeros((rows * cols, 3), dtype=np.float32)
    grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm
    return grid


def compute_intrinsics(
    frames: list[np.ndarray],
    pattern_size: tuple[int, int],
    square_mm: float,
    min_frames: int = MIN_FRAMES,
) -> Intrinsics:
    """Calibrate from several checkerboard views.

    Raises:
        ValueError: if the frames differ in size, or if fewer than
            ``min_frames`` of them show a complete board.

    """
    if not frames:
        raise ValueError("no frames given")
    sizes = {(frame.shape[1], frame.shape[0]) for frame in frames}
    if len(sizes) > 1:
        raise ValueError(f"frames must all be the same size, got {sorted(sizes)}")
    image_size = sizes.pop()

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    for frame in frames:
        corners = find_corners(frame, pattern_size)
        if corners is None:
            continue
        object_points.append(_object_points(pattern_size, square_mm))
        image_points.append(corners.astype(np.float32))

    if len(image_points) < min_frames:
        raise ValueError(
            f"found the board in only {len(image_points)} of {len(frames)} frames, "
            f"need {min_frames}"
        )

    rms, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    return Intrinsics(
        camera_matrix=camera_matrix.tolist(),
        dist_coeffs=dist_coeffs.ravel().tolist(),
        image_size=list(image_size),
        rms_px=float(rms),
        frames_used=len(image_points),
    )


def compute_plane_scale(
    frame: np.ndarray,
    pattern_size: tuple[int, int],
    square_mm: float,
    intrinsics: Intrinsics | None = None,
    bottom_margin_mm: float = 0.0,
) -> PlaneScale:
    """Measure mm/px and the floor row from a board standing on the floor mark.

    Corner spacing along each grid axis gives the scale on the image axis that
    axis runs along, so the board may be detected in either orientation. The
    floor row is the lowest corner row plus ``bottom_margin_mm``, the distance
    the operator measures from the lowest corners down to the board's bottom
    edge.

    Raises:
        ValueError: if the board is not found.

    """
    if intrinsics is not None:
        frame = cv2.undistort(
            frame,
            np.array(intrinsics.camera_matrix, dtype=float),
            np.array(intrinsics.dist_coeffs, dtype=float),
        )
    corners = find_corners(frame, pattern_size)
    if corners is None:
        raise ValueError("checkerboard not found: check lighting, framing and pattern size")

    cols, rows = pattern_size
    grid = corners.reshape(rows, cols, 2)
    # Mean step vectors along each grid axis. The board can be detected in either
    # orientation (the station camera is mounted portrait and frames are rotated),
    # so assign each grid axis to the image axis it actually runs along instead of
    # assuming. Per-axis components, not Euclidean distance: each axis scales alone.
    along_cols = np.diff(grid, axis=1).reshape(-1, 2).mean(axis=0)
    along_rows = np.diff(grid, axis=0).reshape(-1, 2).mean(axis=0)
    step_x, step_y = (
        (along_cols, along_rows)
        if abs(along_cols[0]) >= abs(along_rows[0])
        else (along_rows, along_cols)
    )
    if abs(step_x[0]) < 1.0 or abs(step_y[1]) < 1.0:
        raise ValueError(
            "board is not square to the camera: rotate it so its rows are level"
        )
    mm_per_px_x = square_mm / abs(float(step_x[0]))
    mm_per_px_y = square_mm / abs(float(step_y[1]))
    floor_row = int(round(grid[:, :, 1].max() + bottom_margin_mm / mm_per_px_y))

    return PlaneScale(
        mm_per_px_x=round(mm_per_px_x, 4),
        mm_per_px_y=round(mm_per_px_y, 4),
        floor_row=floor_row,
        corners_found=len(corners),
    )


def _intrinsics_path() -> Path:
    return ROOT / load_app_config().paths.calib / "intrinsics.yaml"


def save_intrinsics(intrinsics: Intrinsics, path: Path | None = None) -> Path:
    """Park step 1's result so step 2 can pick it up. Returns the path written."""
    path = path or _intrinsics_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(asdict(intrinsics), sort_keys=False), encoding="utf-8")
    return path


def load_intrinsics(path: Path | None = None) -> Intrinsics | None:
    """Return the parked intrinsics, or ``None`` if step 1 has not run."""
    path = path or _intrinsics_path()
    if not path.exists():
        return None
    return Intrinsics(**yaml.safe_load(path.read_text(encoding="utf-8")))


def calibration_path() -> Path:
    """Return the path of the station calibration file."""
    return CONFIG_DIR / "calibration.yaml"


def save_calibration(
    intrinsics: Intrinsics,
    scale: PlaneScale,
    rotation: int,
    path: Path | None = None,
) -> Calibration:
    """Write ``config/calibration.yaml`` in the brief's format and return it validated."""
    calibration = Calibration(
        camera_matrix=intrinsics.camera_matrix,
        dist_coeffs=intrinsics.dist_coeffs,
        image_size=intrinsics.image_size,
        rotation=rotation,
        mm_per_px_x=scale.mm_per_px_x,
        mm_per_px_y=scale.mm_per_px_y,
        floor_row=scale.floor_row,
        calibrated_at=datetime.now().astimezone().isoformat(timespec="seconds"),
    )
    payload: dict[str, Any] = calibration.model_dump()
    path = path or calibration_path()
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return calibration
