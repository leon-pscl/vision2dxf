"""End-to-end: capture, measure, export, delete.

``analyse`` is stubbed throughout. The models are exercised in M3's tests; what
matters here is the session plumbing, and YOLO cannot see a drawn silhouette
anyway.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml
from tests.synthetic.people import FLOOR_ROW, make_person, make_side

from bodymeasure import session as session_module
from bodymeasure.config import Calibration, load_schema
from bodymeasure.landmarks import as_dict
from bodymeasure.session import (
    Subject,
    captured_views,
    create_session,
    delete_session,
    export,
    find_landmarks,
    load_subject,
    measure_session,
    save_view,
    session_dir,
)
from bodymeasure.vision import ViewAnalysis

CALIBRATION = Calibration(
    camera_matrix=[[1400.0, 0.0, 450.0], [0.0, 1400.0, 900.0], [0.0, 0.0, 1.0]],
    dist_coeffs=[0.0, 0.0, 0.0, 0.0, 0.0],
    image_size=[900, 1800],
    rotation=0,
    mm_per_px_x=1.0,
    mm_per_px_y=1.0,
    floor_row=FLOOR_ROW,
    calibrated_at="2026-10-11T10:00:00+08:00",
)


def _photo(mask: np.ndarray) -> np.ndarray:
    """Return a grey-on-white image the size of the silhouette."""
    return np.where(mask[..., None], 90, 230).astype(np.uint8).repeat(3, axis=2)


@pytest.fixture
def station(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Return a calibrated station whose sessions and vision are under test control."""
    front = make_person()
    sides = {name: make_side(name) for name in ("side_left", "side_right")}

    monkeypatch.setattr(session_module, "sessions_root", lambda: tmp_path / "sessions")
    monkeypatch.setattr(session_module, "_require_calibration", lambda: CALIBRATION)

    def fake_analyse(image: np.ndarray) -> ViewAnalysis:
        """Return the figure whose silhouette matches the image size and ink."""
        for figure in (front, *sides.values()):
            if np.array_equal(image, _photo(figure.mask)):
                return ViewAnalysis(figure.mask, figure.keypoints, 1.0, 1.0)
        raise ValueError("No person found in the frame")

    monkeypatch.setattr(session_module, "analyse", fake_analyse)
    return {"front": front, "sides": sides}


def _capture(station: dict[str, object], views: tuple[str, ...]) -> str:
    subject = Subject(id="S014", sex="F", consent=True, notes="tight clothing")
    session_id = create_session(subject)
    for view in views:
        figure = station["front"] if view == "front" else station["sides"][view]
        save_view(session_id, view, _photo(figure.mask))
    return session_id


def test_consent_is_required_before_any_capture() -> None:
    with pytest.raises(ValueError, match="consent is required"):
        create_session(Subject(id="S014", sex="F", consent=False))


def test_a_subject_id_cannot_escape_the_sessions_folder() -> None:
    with pytest.raises(ValueError, match="letters, digits"):
        create_session(Subject(id="../../etc", sex="F", consent=True))
    with pytest.raises(ValueError, match="letters, digits"):
        session_dir("../secrets")


def test_a_session_records_its_subject(station: dict[str, object]) -> None:
    session_id = _capture(station, ("front",))
    subject = load_subject(session_id)
    assert subject.id == "S014"
    assert subject.consent is True
    assert session_id.startswith("S014-")


def test_capturing_a_view_writes_image_mask_keypoints_and_overlay(
    station: dict[str, object],
) -> None:
    session_id = _capture(station, ("front",))
    folder = session_dir(session_id)

    assert (folder / "front.jpg").exists()
    assert (folder / "front_mask.png").exists()
    assert (folder / "front_kp.json").exists()
    assert (folder / "debug" / "front_gate.png").exists()
    assert captured_views(session_id) == ["front"]


def test_a_good_front_capture_passes_the_gate(station: dict[str, object]) -> None:
    subject = Subject(id="S014", sex="F", consent=True)
    session_id = create_session(subject)
    result = save_view(session_id, "front", _photo(station["front"].mask))
    assert result.ok, result.reasons


def test_an_unknown_view_is_refused(station: dict[str, object]) -> None:
    session_id = _capture(station, ("front",))
    with pytest.raises(ValueError, match="view must be one of"):
        save_view(session_id, "back", _photo(station["front"].mask))


def test_landmarks_are_found_for_every_captured_view(station: dict[str, object]) -> None:
    session_id = _capture(station, ("front", "side_left"))
    marks = find_landmarks(session_id)
    folder = session_dir(session_id)

    views = {mark.view for mark in marks}
    assert views == {"front", "side_left"}
    assert (folder / "landmarks.json").exists()
    assert (folder / "debug" / "front_marks.png").exists()
    assert (folder / "debug" / "side_left_marks.png").exists()


def test_measuring_needs_a_side_view(station: dict[str, object]) -> None:
    session_id = _capture(station, ("front",))
    with pytest.raises(ValueError, match="at least one side view"):
        measure_session(session_id)


def test_front_plus_one_side_exports_both_files(station: dict[str, object]) -> None:
    session_id = _capture(station, ("front", "side_left"))
    measurements = measure_session(session_id)
    body_path, report_path = export(session_id, measurements)

    body = yaml.safe_load(body_path.read_text(encoding="utf-8"))
    assert set(body) == {"body"}
    assert set(body["body"]) == set(load_schema())
    assert not any(key.startswith("_") for key in body["body"]), "pygarment recomputes those"

    report = yaml.safe_load(report_path.read_text(encoding="utf-8"))
    assert report["subject_id"] == "S014"
    assert report["side_views_used"] == ["left"]
    assert report["values"]["bust_points"]["source"] == "default_ratio"
    assert report["values"]["height"]["confidence"] == "high"


def test_front_plus_both_sides_also_exports(station: dict[str, object]) -> None:
    session_id = _capture(station, ("front", "side_left", "side_right"))
    measurements = measure_session(session_id)
    body_path, report_path = export(session_id, measurements)

    body = yaml.safe_load(body_path.read_text(encoding="utf-8"))
    report = yaml.safe_load(report_path.read_text(encoding="utf-8"))
    assert len(body["body"]) == 26
    assert report["side_views_used"] == ["left", "right"]


def test_edited_landmarks_change_the_measurements(station: dict[str, object]) -> None:
    from dataclasses import replace

    session_id = _capture(station, ("front", "side_left"))
    marks = find_landmarks(session_id)
    before = measure_session(session_id, marks).body()

    moved = [
        replace(mark, y=mark.y - 50, auto=False) if mark.name == "waist_row" else mark
        for mark in marks
    ]
    after = measure_session(session_id, moved).body()

    assert after["waist"] != before["waist"]
    assert after["height"] == before["height"]
    saved = as_dict(
        __import__("bodymeasure.landmarks", fromlist=["load_landmarks"]).load_landmarks(
            session_dir(session_id) / "landmarks.json"
        )
    )
    assert saved["waist_row"].auto is False


def test_deleting_a_session_removes_every_image(station: dict[str, object]) -> None:
    session_id = _capture(station, ("front", "side_left"))
    folder = session_dir(session_id)
    assert list(folder.glob("*.jpg"))

    assert delete_session(session_id) is True
    assert not folder.exists()
    assert delete_session(session_id) is False


def test_an_uncalibrated_station_refuses_to_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(session_module, "sessions_root", lambda: tmp_path / "sessions")
    monkeypatch.setattr(session_module, "load_calibration", lambda: None)
    session_id = create_session(Subject(id="S020", sex="M", consent=True))
    with pytest.raises(ValueError, match="not calibrated"):
        save_view(session_id, "front", np.zeros((10, 10, 3), dtype=np.uint8))
