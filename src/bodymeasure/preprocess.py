"""Frame preparation: decode, orient, undistort, enhance (brief 9.1).

Each step is kept as its own image rather than folded into the next one, because
the operator has to be able to see what was done to the photograph before any
measurement was taken from it. :class:`Prepared` carries them all.

Enhancement is deliberately restrained: white balance, local contrast and an
edge-preserving denoise change what a pixel *looks* like, never where the body
boundary *is*, so a measurement taken from the enhanced frame is a measurement
of the same silhouette.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from bodymeasure.config import Calibration, load_app_config

NOT_APPLIED = "not applied"

_ROTATIONS = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def decode_image(buffer: bytes) -> np.ndarray:
    """Decode an uploaded image to BGR, within the configured upload limits.

    Decoding to an array is also how nonessential metadata is stripped (brief
    9.1): nothing of the original container survives into the stage files.

    Raises:
        ValueError: if the bytes are not a readable image, or the file or its
            pixel count exceeds the configured limit.

    """
    limits = load_app_config().uploads
    if len(buffer) > limits.max_bytes:
        raise ValueError(f"image is larger than the {limits.max_bytes // 1_000_000} MB limit")
    image = cv2.imdecode(np.frombuffer(buffer, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("not a readable image")
    pixels = int(image.shape[0]) * int(image.shape[1])
    if pixels > limits.max_pixels:
        raise ValueError(f"image has more than the {limits.max_pixels} pixel limit")
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


def pad(image: np.ndarray, frac: float) -> np.ndarray:
    """Add a border of ``frac`` of the long side around the frame.

    A tight room leaves the subject filling the frame edge to edge, which both
    the detector and the framing gate read as a cut-off body. A border is air
    that costs no distance: millimetres per pixel come from the head-to-floor
    span, which a border does not move.
    """
    border = int(round(frac * max(image.shape[:2])))
    if border <= 0:
        return image
    return cv2.copyMakeBorder(
        image, border, border, border, border, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )


def enhance(image: np.ndarray, parameters: dict[str, float]) -> np.ndarray:
    """Normalise white balance and local contrast, then denoise, preserving edges.

    Grey-world white balance first, so a warm room does not bias the contrast
    step; CLAHE on lightness alone, so colour is not pushed around; then a
    bilateral filter, which smooths inside the body and inside the background
    but leaves the boundary between them where it was.
    """
    result = image
    if parameters.get("white_balance"):
        channels = cv2.split(result.astype(np.float32))
        means = [float(channel.mean()) or 1.0 for channel in channels]
        grey = float(np.mean(means))
        result = cv2.merge(
            [
                np.clip(channel * grey / mean, 0, 255)
                for channel, mean in zip(channels, means, strict=True)
            ]
        ).astype(np.uint8)

    clip = float(parameters.get("clahe_clip", 0.0))
    if clip > 0:
        grid = int(parameters.get("clahe_grid", 8))
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        lightness, a, b = cv2.split(lab)
        lightness = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(lightness)
        result = cv2.cvtColor(cv2.merge([lightness, a, b]), cv2.COLOR_LAB2BGR)

    diameter = int(parameters.get("bilateral_d", 0))
    if diameter > 0:
        result = cv2.bilateralFilter(
            result,
            diameter,
            float(parameters.get("bilateral_sigma_colour", 50)),
            float(parameters.get("bilateral_sigma_space", 50)),
        )
    return result


@dataclass(frozen=True)
class Prepared:
    """Every preprocessing stage of one view, plus what was done to it.

    ``enhanced`` is the frame the vision models see and every measurement is
    taken from; the earlier images exist so the operator can check that nothing
    moved on the way there.
    """

    raw: np.ndarray
    oriented: np.ndarray
    undistorted: np.ndarray
    enhanced: np.ndarray
    rotation_deg: int
    lens_corrected: bool
    pad_frac: float
    enhance_parameters: dict[str, float] = field(default_factory=dict)


def prepare_stages(
    image: np.ndarray, calibration: Calibration | None, upright: bool = False
) -> Prepared:
    """Run the whole preprocessing chain and keep each stage.

    Undistortion runs on the unpadded frame, because the lens model describes
    the sensor and not a border we added; the border is therefore part of the
    enhanced stage and is recorded in its parameters.
    """
    config = load_app_config()
    degrees = 0 if upright else (calibration.rotation if calibration else config.camera.rotation)
    oriented = rotate(image, degrees)
    lens_corrected = bool(calibration and calibration.has_lens_model)
    undistorted = undistort(oriented, calibration) if lens_corrected else oriented
    parameters = dict(config.enhance)
    return Prepared(
        raw=image,
        oriented=oriented,
        undistorted=undistorted,
        enhanced=enhance(pad(undistorted, config.camera.pad_frac), parameters),
        rotation_deg=degrees % 360,
        lens_corrected=lens_corrected,
        pad_frac=config.camera.pad_frac,
        enhance_parameters=parameters,
    )


def prepare(
    image: np.ndarray, calibration: Calibration | None, upright: bool = False
) -> np.ndarray:
    """Rotate, undistort if the station has a lens model, then pad.

    ``upright`` says the frame is already in final orientation and must not be
    turned: the station's camera is mounted portrait, so its live frames arrive
    sideways, but a photograph off the camera roll or off disk is the right way up.

    Rotation comes from the calibration file when there is one and from
    ``config/app.yaml`` otherwise, so an uncorrected station still gets portrait
    frames the pose gate can use.
    """
    return prepare_stages(image, calibration, upright).enhanced
