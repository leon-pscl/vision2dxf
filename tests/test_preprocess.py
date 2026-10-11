"""Frame preparation: rotate, then undistort when there is a lens model.

The distinction that matters is which way up the frame arrives. The station's
camera is mounted portrait, so a live capture comes in sideways and is turned; an
uploaded photograph or a stored sample is already the right way up and must not
be turned again, or the subject ends up lying down.
"""

import numpy as np
import pytest

from bodymeasure.config import Calibration, load_app_config
from bodymeasure.preprocess import prepare, rotate

ROTATION = load_app_config().camera.rotation
LANDSCAPE = np.zeros((720, 1280, 3), dtype=np.uint8)
PORTRAIT = np.zeros((1280, 720, 3), dtype=np.uint8)


def _marker(shape: tuple[int, ...]) -> np.ndarray:
    """Return a frame with one bright pixel, so its position is identifiable."""
    image = np.zeros(shape, dtype=np.uint8)
    image[10, 20] = 255
    return image


def test_a_live_frame_is_turned_the_configured_way() -> None:
    assert prepare(_marker(LANDSCAPE.shape), Calibration(rotation=90)).shape == PORTRAIT.shape
    assert prepare(_marker(LANDSCAPE.shape), None).shape == PORTRAIT.shape


def test_an_upright_frame_is_left_alone() -> None:
    prepared = prepare(_marker(PORTRAIT.shape), Calibration(rotation=90), upright=True)
    assert prepared.shape == PORTRAIT.shape
    assert np.array_equal(prepared, _marker(PORTRAIT.shape))


def test_upright_still_undistorts_when_the_station_has_a_lens_model() -> None:
    """Turning is skipped; the lens correction is not."""
    lens = Calibration(
        camera_matrix=[[1400.0, 0.0, 360.0], [0.0, 1400.0, 640.0], [0.0, 0.0, 1.0]],
        dist_coeffs=[0.01, -0.02, 0.0, 0.0, 0.0],
        rotation=90,
    )
    prepared = prepare(_marker(PORTRAIT.shape), lens, upright=True)
    assert prepared.shape == PORTRAIT.shape
    assert not np.array_equal(prepared, _marker(PORTRAIT.shape)), "not undistorted"


def test_rotation_still_refuses_an_angle_that_would_resample() -> None:
    with pytest.raises(ValueError, match="0, 90, 180 or 270"):
        rotate(_marker(LANDSCAPE.shape), 45)