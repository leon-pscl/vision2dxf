"""Frame preparation: decode, rotate, undistort (brief 5.1)."""

from __future__ import annotations

import cv2
import numpy as np

from bodymeasure.config import Calibration, load_app_config

_ROTATIONS = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def decode_image(buffer: bytes) -> np.ndarray:
    """Decode an uploaded image to BGR.

    Raises:
        ValueError: if the bytes are not a readable image.

    """
    image = cv2.imdecode(np.frombuffer(buffer, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("not a readable image")
    return image


def rotate(image: np.ndarray, degrees: int) -> np.ndarray:
    """Rotate clockwise by 0, 90, 180 or 270 degrees.

    Raises:
        ValueError: for any other angle, which would need resampling and would
            silently change the calibrated scale.

    """
    if degrees % 360 == 0:
        return image
    try:
        return cv2.rotate(image, _ROTATIONS[degrees % 360])
    except KeyError as exc:
        raise ValueError(f"rotation must be 0, 90, 180 or 270 degrees, got {degrees}") from exc


def undistort(image: np.ndarray, calibration: Calibration) -> np.ndarray:
    """Undistort an already-rotated frame with the stored lens model."""
    return cv2.undistort(
        image,
        np.array(calibration.camera_matrix, dtype=float),
        np.array(calibration.dist_coeffs, dtype=float),
    )


def prepare(
    image: np.ndarray, calibration: Calibration | None, upright: bool = False
) -> np.ndarray:
    """Rotate, then undistort if the station has a lens model.

    ``upright`` says the frame is already in final orientation and must not be
    turned: the station's camera is mounted portrait, so its live frames arrive
    sideways, but a photograph off the camera roll or off disk is the right way up.

    Rotation comes from the calibration file when there is one and from
    ``config/app.yaml`` otherwise, so an uncorrected station still gets portrait
    frames the pose gate can use.
    """
    config = load_app_config()
    degrees = 0 if upright else (calibration.rotation if calibration else config.camera.rotation)
    rotated = rotate(image, degrees)
    if calibration and calibration.has_lens_model:
        return undistort(rotated, calibration)
    return rotated
