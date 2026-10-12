"""Frame preparation: rotate, then undistort when there is a lens model.

The distinction that matters is which way up the frame arrives. The station's
camera is mounted portrait, so a live capture comes in sideways and is turned; an
uploaded photograph or a stored sample is already the right way up and must not
be turned again, or the subject ends up lying down.
"""

import numpy as np
import pytest

from bodymeasure.config import Calibration, load_app_config
from bodymeasure.preprocess import (
    NOT_APPLIED,
    decode_image,
    enhance,
    pad,
    prepare,
    prepare_stages,
    rotate,
)

ROTATION = load_app_config().camera.rotation
PAD_FRAC = load_app_config().camera.pad_frac
ENHANCE = load_app_config().enhance
LANDSCAPE = np.zeros((720, 1280, 3), dtype=np.uint8)
PORTRAIT = np.zeros((1280, 720, 3), dtype=np.uint8)
PADDED = pad(PORTRAIT, PAD_FRAC).shape


def _marker(shape: tuple[int, ...]) -> np.ndarray:
    """Return a frame with one bright pixel, so its position is identifiable."""
    image = np.zeros(shape, dtype=np.uint8)
    image[10, 20] = 255
    return image


def test_a_live_frame_is_turned_the_configured_way() -> None:
    assert prepare(_marker(LANDSCAPE.shape), Calibration(rotation=90)).shape == PADDED
    assert prepare(_marker(LANDSCAPE.shape), None).shape == PADDED


def test_an_upright_frame_is_left_alone() -> None:
    prepared = prepare(_marker(PORTRAIT.shape), Calibration(rotation=90), upright=True)
    assert prepared.shape == PADDED
    enhanced = enhance(pad(_marker(PORTRAIT.shape), PAD_FRAC), dict(ENHANCE))
    assert np.array_equal(prepared, enhanced), "only the border and the enhancement"


def test_upright_still_undistorts_when_the_station_has_a_lens_model() -> None:
    """Turning is skipped; the lens correction is not."""
    lens = Calibration(
        camera_matrix=[[1400.0, 0.0, 360.0], [0.0, 1400.0, 640.0], [0.0, 0.0, 1.0]],
        dist_coeffs=[0.01, -0.02, 0.0, 0.0, 0.0],
        rotation=90,
    )
    prepared = prepare(_marker(PORTRAIT.shape), lens, upright=True)
    assert prepared.shape == PADDED
    assert not np.array_equal(prepared, pad(_marker(PORTRAIT.shape), PAD_FRAC)), "not undistorted"


def test_a_body_touching_the_edge_gains_a_margin() -> None:
    """The border is what lets a frame-filling body pass the framing gate."""
    full = np.ones((1280, 720, 3), dtype=np.uint8) * 255
    padded = pad(full, PAD_FRAC)
    border = (padded.shape[0] - full.shape[0]) // 2
    assert border > 0
    assert padded[0, 0].tolist() == [114, 114, 114]
    assert np.array_equal(padded[border:-border, border:-border], full)


def test_rotation_still_refuses_an_angle_that_would_resample() -> None:
    with pytest.raises(ValueError, match="0, 90, 180 or 270"):
        rotate(_marker(LANDSCAPE.shape), 45)

def _silhouette() -> np.ndarray:
    """Return a dim, warm-tinted photo with a bright rectangle in it."""
    image = np.full((200, 120, 3), (70, 60, 40), dtype=np.uint8)
    image[50:150, 30:90] = (150, 140, 120)
    return image


def test_enhancement_does_not_move_the_body_boundary() -> None:
    """Brief 9.1: enhancement may change how a pixel looks, never where the edge is."""
    photo = _silhouette()
    before = photo.mean(axis=2) > 100
    after = enhance(photo, dict(ENHANCE)).mean(axis=2) > 100

    assert before.sum() > 0
    moved = int(np.logical_xor(before, after).sum())
    assert moved <= int(0.01 * before.sum()), f"{moved} px of the silhouette moved"


def test_enhancement_is_off_when_every_parameter_is_zero() -> None:
    photo = _silhouette()
    off = {"white_balance": 0, "clahe_clip": 0.0, "bilateral_d": 0}
    assert np.array_equal(enhance(photo, off), photo)


def test_every_preprocessing_stage_is_kept_and_labelled() -> None:
    """The operator must be able to see each step, including one not applied."""
    stages = prepare_stages(_marker(LANDSCAPE.shape), None)

    assert stages.raw.shape == LANDSCAPE.shape, "the raw frame is never rewritten"
    assert stages.oriented.shape[:2] == PORTRAIT.shape[:2]
    assert np.array_equal(stages.undistorted, stages.oriented), "no lens model, so no change"
    assert stages.lens_corrected is False
    assert stages.enhanced.shape == PADDED
    assert stages.rotation_deg == ROTATION % 360
    assert NOT_APPLIED == "not applied"


def test_an_upload_over_the_pixel_limit_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    import cv2

    from bodymeasure import preprocess

    limits = load_app_config().uploads
    monkeypatch.setattr(
        preprocess, "load_app_config", lambda: type("C", (), {"uploads": limits})()
    )
    big = cv2.imencode(".png", np.zeros((40, 40, 3), dtype=np.uint8))[1].tobytes()
    assert decode_image(big).shape == (40, 40, 3)

    monkeypatch.setattr(limits, "max_pixels", 100)
    with pytest.raises(ValueError, match="pixel limit"):
        decode_image(big)
    monkeypatch.setattr(limits, "max_pixels", 10_000_000)
    monkeypatch.setattr(limits, "max_bytes", 10)
    with pytest.raises(ValueError, match="MB limit"):
        decode_image(big)


def test_unreadable_bytes_are_refused() -> None:
    with pytest.raises(ValueError, match="not a readable image"):
        decode_image(b"this is not a picture")
