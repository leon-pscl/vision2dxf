"""The stage files and the manifest that indexes them (brief 7, 8)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from bodymeasure import artifacts, store


@pytest.fixture(autouse=True)
def _sessions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(store, "data_dir", lambda: tmp_path)
    monkeypatch.setattr(store, "sessions_root", lambda: tmp_path / "sessions")
    store.initialise()


@pytest.fixture
def session_id() -> str:
    session = store.create_session("S014", "F", "2026-10-11", 164.5)
    store.session_folder(session).mkdir(parents=True, exist_ok=True)
    return session


def _photo() -> np.ndarray:
    return np.full((40, 30, 3), 120, dtype=np.uint8)


def test_the_stage_list_is_the_pipeline_the_brief_specifies() -> None:
    assert artifacts.STAGE_IDS == (
        "00_raw",
        "01_oriented",
        "02_undistorted",
        "03_enhanced",
        "04_detection",
        "05_roi",
        "06_mask_raw",
        "07_mask_clean",
        "08_body_isolated",
        "09_edges",
        "10_keypoints",
        "11_landmarks",
        "12_measurement_overlay",
    )
    assert artifacts.stage("00_raw").suffix == ".jpg", "the raw capture stays as it arrived"
    assert all(stage.label and stage.description for stage in artifacts.STAGES)


def test_a_recorded_stage_is_on_disk_and_in_the_manifest(session_id: str) -> None:
    path = artifacts.record(
        session_id,
        "front",
        "07_mask_clean",
        np.zeros((40, 30), dtype=bool),
        "clean_mask",
        {"kernel_px": 5, "max_hole_area_px": 500},
        ["views/front/06_mask_raw.png"],
        runtime_ms=12.345,
    )

    assert path == artifacts.stage_path(session_id, "front", "07_mask_clean")
    assert cv2.imread(str(path)) is not None, "a mask must be viewable as a picture"

    entry = artifacts.manifest(session_id)[0]
    assert entry["stage_id"] == "07_mask_clean"
    assert entry["view"] == "front"
    assert entry["path"] == "views/front/07_mask_clean.png"
    assert entry["algorithm"] == "clean_mask"
    assert entry["parameters"] == {"kernel_px": 5, "max_hole_area_px": 500}
    assert entry["source_files"] == ["views/front/06_mask_raw.png"]
    assert entry["runtime_ms"] == 12.3
    assert (entry["width"], entry["height"]) == (30, 40)
    assert entry["created_at"].endswith("+00:00")


def test_recording_the_same_stage_again_replaces_its_entry(session_id: str) -> None:
    """A retake must not leave two entries claiming the same file."""
    for _ in range(3):
        artifacts.record(session_id, "front", "00_raw", _photo(), "camera_or_upload")

    entries = artifacts.manifest(session_id)
    assert len(entries) == 1
    assert entries[0]["stage_id"] == "00_raw"


def test_dropping_a_view_removes_its_files_and_its_entries(session_id: str) -> None:
    artifacts.record(session_id, "front", "00_raw", _photo(), "camera_or_upload")
    artifacts.record(session_id, "side_left", "00_raw", _photo(), "camera_or_upload")

    artifacts.drop_view(session_id, "front")

    assert [entry["view"] for entry in artifacts.manifest(session_id)] == ["side_left"]
    assert not artifacts.stage_path(session_id, "front", "00_raw").exists()
    assert artifacts.stage_path(session_id, "side_left", "00_raw").exists()


def test_dropping_a_stage_clears_it_on_every_view(session_id: str) -> None:
    """Landmarks are found for all views together, so they are lost together."""
    for view in ("front", "side_left"):
        artifacts.record(session_id, view, "00_raw", _photo(), "camera_or_upload")
        artifacts.record(session_id, view, "11_landmarks", _photo(), "find_landmarks")

    artifacts.drop_stages(session_id, ("11_landmarks",))

    assert {entry["stage_id"] for entry in artifacts.manifest(session_id)} == {"00_raw"}
    for view in ("front", "side_left"):
        assert not artifacts.stage_path(session_id, view, "11_landmarks").exists()
        assert artifacts.stage_path(session_id, view, "00_raw").exists()


def test_an_unknown_stage_cannot_name_a_file(session_id: str) -> None:
    with pytest.raises(KeyError):
        artifacts.stage_path(session_id, "front", "../subject")
    with pytest.raises(KeyError):
        artifacts.record(session_id, "front", "99_made_up", _photo(), "nothing")


def test_a_mask_and_a_contour_are_both_saved_as_pictures(session_id: str) -> None:
    """A boolean array and a single-channel image must still be viewable."""
    mask = np.zeros((20, 10), dtype=bool)
    mask[5:15, 2:8] = True
    grey = (mask * 255).astype(np.uint8)

    artifacts.record(session_id, "front", "06_mask_raw", mask, "yolo26_seg")
    artifacts.record(session_id, "front", "09_edges", grey, "findContours")

    for stage_id in ("06_mask_raw", "09_edges"):
        image = cv2.imread(str(artifacts.stage_path(session_id, "front", stage_id)))
        assert image is not None and image.shape == (20, 10, 3), stage_id


def test_the_manifest_is_valid_json_after_every_write(session_id: str) -> None:
    artifacts.record(session_id, "front", "00_raw", _photo(), "camera_or_upload")
    text = (store.session_folder(session_id) / artifacts.MANIFEST).read_text(encoding="utf-8")
    assert isinstance(json.loads(text), list)


def test_a_session_with_no_stages_has_an_empty_manifest(session_id: str) -> None:
    assert artifacts.manifest(session_id) == []
