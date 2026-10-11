"""API skeleton and the calibration endpoints."""

import uuid
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient
from tests.synthetic.boards import (
    PATTERN,
    SQUARE_MM,
    calibration_frames,
    render_upright_board,
)
from tests.synthetic.people import make_person, make_side
from tests.test_session import CALIBRATION as SESSION_CALIBRATION

from bodymeasure import calibrate, store
from bodymeasure import session as session_module
from bodymeasure.api import app
from bodymeasure.config import load_calibration
from bodymeasure.vision import ViewAnalysis

client = TestClient(app)


def test_health() -> None:
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert len(body["measurement_keys"]) == 26


def test_index_page_is_served() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Body measurement" in response.text


def _png(image: np.ndarray) -> bytes:
    return cv2.imencode(".png", image)[1].tobytes()


@pytest.fixture
def calib_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Point both calibration artefacts at a temporary directory."""
    intrinsics = tmp_path / "intrinsics.yaml"
    calibration = tmp_path / "calibration.yaml"
    monkeypatch.setattr(calibrate, "_intrinsics_path", lambda: intrinsics)
    monkeypatch.setattr(calibrate, "calibration_path", lambda: calibration)
    return {"intrinsics": intrinsics, "calibration": calibration}


def _board_params() -> dict[str, float | int]:
    return {
        "pattern_cols": PATTERN[0],
        "pattern_rows": PATTERN[1],
        "square_mm": SQUARE_MM,
    }


def test_checkerboard_endpoint_returns_and_parks_intrinsics(calib_paths: dict[str, Path]) -> None:
    frames = [
        ("frames", (f"f{index}.png", _png(frame), "image/png"))
        for index, frame in enumerate(calibration_frames())
    ]
    response = client.post("/api/calibration/checkerboard", files=frames, data=_board_params())
    body = response.json()

    assert response.status_code == 200, body
    assert body["frames_used"] >= 10
    # app.yaml rotates frames 90 degrees, so the calibrated size is portrait.
    assert body["image_size"] == [900, 1200]
    assert body["rms_px"] < 1.0
    assert calib_paths["intrinsics"].exists()


def test_checkerboard_endpoint_explains_too_few_boards(calib_paths: dict[str, Path]) -> None:
    blank = np.full((900, 1200, 3), 255, dtype=np.uint8)
    frames = [("frames", (f"f{index}.png", _png(blank), "image/png")) for index in range(3)]

    response = client.post("/api/calibration/checkerboard", files=frames, data=_board_params())

    assert response.status_code == 422
    assert "need 10" in response.json()["detail"]
    assert not calib_paths["intrinsics"].exists()


def test_scale_endpoint_shows_numbers_without_saving(calib_paths: dict[str, Path]) -> None:
    board = ("frame", ("board.png", _png(render_upright_board()), "image/png"))

    response = client.post("/api/calibration/scale", files=[board], data=_board_params())
    body = response.json()

    assert response.status_code == 200, body
    assert body["saved"] is False
    assert body["corners_found"] == PATTERN[0] * PATTERN[1]
    assert body["mm_per_px_x"] == pytest.approx(SQUARE_MM / 60, rel=0.02)
    assert not calib_paths["calibration"].exists()


def test_scale_endpoint_saves_the_station_calibration(calib_paths: dict[str, Path]) -> None:
    frames = [
        ("frames", (f"f{index}.png", _png(frame), "image/png"))
        for index, frame in enumerate(calibration_frames())
    ]
    client.post("/api/calibration/checkerboard", files=frames, data=_board_params())
    board = ("frame", ("board.png", _png(render_upright_board()), "image/png"))

    response = client.post(
        "/api/calibration/scale",
        files=[board],
        data={**_board_params(), "save": "true", "bottom_margin_mm": 20.0},
    )
    body = response.json()

    assert response.status_code == 200, body
    assert body["saved"] is True
    saved = load_calibration(calib_paths["calibration"])
    assert saved is not None
    assert saved.rotation == 90
    assert saved.floor_row == body["floor_row"]


def test_scale_endpoint_refuses_to_save_without_intrinsics(calib_paths: dict[str, Path]) -> None:
    board = ("frame", ("board.png", _png(render_upright_board()), "image/png"))

    response = client.post(
        "/api/calibration/scale", files=[board], data={**_board_params(), "save": "true"}
    )

    assert response.status_code == 409
    assert "checkerboard step first" in response.json()["detail"]
    assert not calib_paths["calibration"].exists()


@pytest.fixture
def station(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Return a calibrated station with vision stubbed, as in test_session."""
    front = make_person()
    sides = {name: make_side(name) for name in ("side_left", "side_right")}
    monkeypatch.setattr(store, "sessions_root", lambda: tmp_path / "sessions")
    monkeypatch.setattr(store, "data_dir", lambda: tmp_path)
    monkeypatch.setattr(session_module, "_require_calibration", lambda: SESSION_CALIBRATION)
    store.initialise()

    def fake_analyse(image: np.ndarray) -> ViewAnalysis:
        for figure in (front, *sides.values()):
            if np.array_equal(image, _photo(figure.mask)):
                return ViewAnalysis(figure.mask, figure.keypoints, 1.0, 1.0)
        raise ValueError("No person found in the frame")

    monkeypatch.setattr(session_module, "analyse", fake_analyse)
    return {"front": front, "sides": sides}


def _photo(mask: np.ndarray) -> np.ndarray:
    return np.where(mask[..., None], 90, 230).astype(np.uint8).repeat(3, axis=2)


def _open_session() -> str:
    response = client.post(
        "/api/sessions", data={"subject_code": "S014", "sex": "F", "consent": "true"}
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


def _send(session_id: str, view: str, figure: object) -> dict[str, object]:
    file = ("image", (f"{view}.png", _png(_photo(figure.mask)), "image/png"))
    response = client.post(f"/api/sessions/{session_id}/views/{view}", files=[file])
    assert response.status_code == 200, response.text
    return response.json()


def _capture_all(station: dict[str, object], views: tuple[str, ...]) -> str:
    session_id = _open_session()
    for view in views:
        figure = station["front"] if view == "front" else station["sides"][view]
        _send(session_id, view, figure)
    return session_id


def test_the_capture_page_offers_consent_and_three_views() -> None:
    page = client.get("/").text
    assert 'id="consent"' in page
    for view in ("front", "side_left", "side_right"):
        assert f'data-view="{view}"' in page
    assert 'id="measure"' in page


def test_health_reports_the_calibration_for_the_floor_guide() -> None:
    body = client.get("/api/health").json()
    assert "calibration" in body
    assert body["calibration"] is None or "floor_row" in body["calibration"]


def test_a_session_cannot_start_without_consent(station: dict[str, object]) -> None:
    response = client.post(
        "/api/sessions", data={"subject_code": "S014", "sex": "F", "consent": "false"}
    )
    assert response.status_code == 422
    assert "consent is required" in response.json()["detail"]


def test_a_subject_id_with_path_characters_is_refused(station: dict[str, object]) -> None:
    response = client.post(
        "/api/sessions", data={"subject_code": "../etc", "sex": "F", "consent": "true"}
    )
    assert response.status_code == 422


def test_capturing_a_view_returns_the_gate_verdict(station: dict[str, object]) -> None:
    session_id = _open_session()
    body = _send(session_id, "front", station["front"])
    assert body["ok"] is True
    assert body["reasons"] == []
    assert body["views"] == ["front"]


def test_capturing_into_a_missing_session_is_a_404(station: dict[str, object]) -> None:
    file = ("image", ("front.png", _png(_photo(station["front"].mask)), "image/png"))
    unknown = uuid.uuid4().hex
    response = client.post(f"/api/sessions/{unknown}/views/front", files=[file])
    assert response.status_code == 404


def test_a_malformed_session_id_is_refused(station: dict[str, object]) -> None:
    """Only UUID hex reaches the filesystem, so a crafted id cannot traverse."""
    file = ("image", ("front.png", _png(_photo(station["front"].mask)), "image/png"))
    response = client.post("/api/sessions/not-a-uuid/views/front", files=[file])
    assert response.status_code == 422


def test_measure_returns_landmarks_and_the_26_keys(station: dict[str, object]) -> None:
    session_id = _capture_all(station, ("front", "side_left"))
    body = client.post(f"/api/sessions/{session_id}/measure").json()

    assert len(body["measurements"]) == 26
    assert body["measurements"]["height"]["confidence"] == "high"
    assert body["measurements"]["bust_points"]["source"] == "default_ratio"
    assert {mark["view"] for mark in body["landmarks"]} == {"front", "side_left"}


def test_measuring_without_a_side_view_explains_why(station: dict[str, object]) -> None:
    session_id = _capture_all(station, ("front",))
    response = client.post(f"/api/sessions/{session_id}/measure")
    assert response.status_code == 422
    assert "side view" in response.json()["detail"]


def test_edited_landmarks_recompute_through_the_api(station: dict[str, object]) -> None:
    session_id = _capture_all(station, ("front", "side_left"))
    first = client.post(f"/api/sessions/{session_id}/measure").json()

    edited = [
        {**mark, "y": mark["y"] - 50, "auto": False} if mark["name"] == "waist_row" else mark
        for mark in first["landmarks"]
    ]
    second = client.put(f"/api/sessions/{session_id}/landmarks", json=edited).json()

    assert second["measurements"]["waist"]["value"] != first["measurements"]["waist"]["value"]
    assert second["measurements"]["height"]["value"] == first["measurements"]["height"]["value"]


def test_the_exports_are_yaml_and_only_appear_after_measuring(
    station: dict[str, object],
) -> None:
    session_id = _capture_all(station, ("front", "side_left"))
    assert client.get(f"/api/sessions/{session_id}/export").status_code == 404

    client.post(f"/api/sessions/{session_id}/measure")
    body = yaml.safe_load(client.get(f"/api/sessions/{session_id}/export").text)
    report = yaml.safe_load(client.get(f"/api/sessions/{session_id}/report").text)

    assert len(body["body"]) == 26
    assert report["subject_id"] == "S014"


def test_overlays_are_served_and_missing_ones_are_404(station: dict[str, object]) -> None:
    session_id = _capture_all(station, ("front", "side_left"))
    client.post(f"/api/sessions/{session_id}/measure")

    response = client.get(f"/api/sessions/{session_id}/overlay/front_marks")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert client.get(f"/api/sessions/{session_id}/overlay/nope").status_code == 404


def test_deleting_a_session_is_final(station: dict[str, object]) -> None:
    session_id = _capture_all(station, ("front", "side_left"))
    assert client.delete(f"/api/sessions/{session_id}").json() == {"deleted": True}
    assert client.delete(f"/api/sessions/{session_id}").status_code == 404


def test_the_review_page_is_served_under_its_session_path(station: dict[str, object]) -> None:
    session_id = _capture_all(station, ("front", "side_left"))
    response = client.get(f"/review/{session_id}")
    assert response.status_code == 200
    assert 'id="review"' in response.text


def test_measuring_twice_keeps_edited_landmarks(station: dict[str, object]) -> None:
    """The review page measures on load; that must not discard the tailor's work."""
    session_id = _capture_all(station, ("front", "side_left"))
    first = client.post(f"/api/sessions/{session_id}/measure").json()
    edited = [
        {**mark, "y": mark["y"] - 50, "auto": False} if mark["name"] == "waist_row" else mark
        for mark in first["landmarks"]
    ]
    client.put(f"/api/sessions/{session_id}/landmarks", json=edited)

    again = client.post(f"/api/sessions/{session_id}/measure").json()
    waist = next(mark for mark in again["landmarks"] if mark["name"] == "waist_row")
    assert waist["auto"] is False
    assert waist["y"] == next(
        mark["y"] for mark in first["landmarks"] if mark["name"] == "waist_row"
    ) - 50


def test_a_new_capture_clears_stale_landmarks(station: dict[str, object]) -> None:
    session_id = _capture_all(station, ("front", "side_left"))
    client.post(f"/api/sessions/{session_id}/measure")
    assert (session_module.session_dir(session_id) / "landmarks.json").exists()

    _send(session_id, "front", station["front"])
    assert not (session_module.session_dir(session_id) / "landmarks.json").exists()


def test_a_typed_value_is_recorded_as_the_tailors(station: dict[str, object]) -> None:
    session_id = _capture_all(station, ("front", "side_left"))
    client.post(f"/api/sessions/{session_id}/measure")

    body = client.put(
        f"/api/sessions/{session_id}/values",
        json={"bust_points": {"value": 17.4, "note": "measured by tape"}},
    ).json()
    entry = body["measurements"]["bust_points"]

    assert entry["value"] == 17.4
    assert entry["edited"] is True
    assert entry["source"] == "tailor"
    assert entry["note"] == "measured by tape"

    report = yaml.safe_load(client.get(f"/api/sessions/{session_id}/report").text)
    assert report["values"]["bust_points"]["note"] == "measured by tape"
    assert yaml.safe_load(client.get(f"/api/sessions/{session_id}/export").text)["body"][
        "bust_points"
    ] == 17.4


def test_a_typed_value_survives_a_later_landmark_drag(station: dict[str, object]) -> None:
    session_id = _capture_all(station, ("front", "side_left"))
    first = client.post(f"/api/sessions/{session_id}/measure").json()
    client.put(
        f"/api/sessions/{session_id}/values", json={"bum_points": {"value": 19.9, "note": "tape"}}
    )

    edited = [
        {**mark, "y": mark["y"] - 30, "auto": False} if mark["name"] == "waist_row" else mark
        for mark in first["landmarks"]
    ]
    body = client.put(f"/api/sessions/{session_id}/landmarks", json=edited).json()

    assert body["measurements"]["bum_points"]["value"] == 19.9
    assert body["measurements"]["bum_points"]["edited"] is True


def test_the_index_lists_sessions_and_filters_by_status(station: dict[str, object]) -> None:
    session_id = _capture_all(station, ("front", "side_left"))
    client.post(f"/api/sessions/{session_id}/measure")

    listed = client.get("/api/sessions").json()
    assert [row["id"] for row in listed] == [session_id]
    assert listed[0]["status"] == "measured"
    assert listed[0]["views"] == ["front", "side_left"]
    assert client.get("/api/sessions?status=confirmed").json() == []


def test_confirming_starts_the_retention_clock(station: dict[str, object]) -> None:
    session_id = _capture_all(station, ("front", "side_left"))
    client.post(f"/api/sessions/{session_id}/measure")

    body = client.post(f"/api/sessions/{session_id}/confirm").json()

    assert body["status"] == "confirmed"
    assert body["images_purge_after"] > date.today().isoformat()
    assert not (session_module.session_dir(session_id) / "debug").exists(), (
        "keep_debug_after_confirm is false"
    )


def test_confirming_an_unknown_session_is_a_404(station: dict[str, object]) -> None:
    assert client.post(f"/api/sessions/{uuid.uuid4().hex}/confirm").status_code == 404


def test_the_csv_export_covers_confirmed_sessions_only(station: dict[str, object]) -> None:
    session_id = _capture_all(station, ("front", "side_left"))
    client.post(f"/api/sessions/{session_id}/measure")
    assert "S014" not in client.get("/api/export.csv").text

    client.post(f"/api/sessions/{session_id}/confirm")
    text = client.get("/api/export.csv").text

    assert "S014" in text
    assert session_id in text
    assert text.splitlines()[0].count("_edited") == 26


def test_edits_reach_the_audit_trail(station: dict[str, object]) -> None:
    session_id = _capture_all(station, ("front", "side_left"))
    first = client.post(f"/api/sessions/{session_id}/measure").json()
    edited = [
        {**mark, "y": mark["y"] - 40, "auto": False} if mark["name"] == "waist_row" else mark
        for mark in first["landmarks"]
    ]
    client.put(f"/api/sessions/{session_id}/landmarks", json=edited)
    client.put(
        f"/api/sessions/{session_id}/values",
        json={"bust_points": {"value": 17.4, "note": "tape"}},
    )

    trail = store.list_edits(session_id)
    kinds = {entry["kind"] for entry in trail}
    assert kinds == {"landmark", "value"}
    assert any(entry["target"] == "front.waist_row" for entry in trail)
    assert any(entry["note"] == "tape" for entry in trail)


def test_deleting_through_the_api_records_the_reason(station: dict[str, object]) -> None:
    session_id = _capture_all(station, ("front", "side_left"))
    client.delete(f"/api/sessions/{session_id}")

    deletions = store.list_deletions()
    assert [entry["reason"] for entry in deletions] == ["user_request"]
    assert [entry["scope"] for entry in deletions] == ["session"]
