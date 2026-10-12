"""API skeleton: the session, capture, measure and export routes."""

import uuid
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient
from tests.synthetic.people import as_captured, make_person, make_side

from bodymeasure import session as session_module
from bodymeasure import store
from bodymeasure.api import app
from bodymeasure.artifacts import STAGE_IDS
from bodymeasure.config import Calibration
from bodymeasure.vision import ViewAnalysis

client = TestClient(app)

HEIGHT_CM = 164.5


def test_health() -> None:
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert len(body["measurement_keys"]) == 26
    assert body["validation_status"] == "unvalidated_against_manual_anthropometry"
    assert body["garmentcode_commit"]
    assert body["profiles"] and body["scale_modes"] == ["stature"]
    assert [stage["stage_id"] for stage in body["stages"]] == list(STAGE_IDS)


def test_index_page_is_served() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Capture a session" in response.text


def _png(image: np.ndarray) -> bytes:
    return cv2.imencode(".png", image)[1].tobytes()


@pytest.fixture
def station(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Return a station with its sessions and vision stubbed, as in test_session."""
    front = make_person()
    sides = {name: make_side(name) for name in ("side_left", "side_right")}
    monkeypatch.setattr(store, "sessions_root", lambda: tmp_path / "sessions")
    monkeypatch.setattr(store, "data_dir", lambda: tmp_path)
    # An uncorrected station: frames are rotated but not undistorted.
    monkeypatch.setattr(session_module, "load_calibration", lambda: Calibration(rotation=0))
    store.initialise()

    def fake_analyse(image: np.ndarray) -> ViewAnalysis:
        for figure in (front, *sides.values()):
            if np.array_equal(image, as_captured(_photo(figure.mask))):
                mask, keypoints = as_captured(figure.mask, figure.keypoints)
                return ViewAnalysis(mask, keypoints, 1.0, 1.0)
        raise ValueError("No person found in the frame")

    monkeypatch.setattr(session_module, "analyse", fake_analyse)
    return {"front": front, "sides": sides}


def _photo(mask: np.ndarray) -> np.ndarray:
    return np.where(mask[..., None], 90, 230).astype(np.uint8).repeat(3, axis=2)


def _open_session(height_cm: float = HEIGHT_CM, profile: str = "full") -> str:
    response = client.post(
        "/api/sessions",
        data={
            "subject_code": "S014",
            "sex": "F",
            "consent": "true",
            "height_cm": str(height_cm),
            "profile": profile,
        },
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
    assert 'id="go-process"' in page


def test_health_reports_whether_the_lens_is_corrected() -> None:
    body = client.get("/api/health").json()
    assert isinstance(body["calibrated"], bool)
    assert "camera_rotation" in body


def test_a_session_cannot_start_without_consent(station: dict[str, object]) -> None:
    response = client.post(
        "/api/sessions",
        data={
            "subject_code": "S014",
            "sex": "F",
            "consent": "false",
            "height_cm": str(HEIGHT_CM),
        },
    )
    assert response.status_code == 422
    assert "consent is required" in response.json()["detail"]


def test_a_session_cannot_start_without_a_plausible_height(
    station: dict[str, object],
) -> None:
    response = client.post(
        "/api/sessions",
        data={"subject_code": "S014", "sex": "F", "consent": "true", "height_cm": "250"},
    )
    assert response.status_code == 422
    assert "height must be" in response.json()["detail"]


def test_a_subject_id_with_path_characters_is_refused(station: dict[str, object]) -> None:
    response = client.post(
        "/api/sessions",
        data={
            "subject_code": "../etc",
            "sex": "F",
            "consent": "true",
            "height_cm": str(HEIGHT_CM),
        },
    )
    assert response.status_code == 422


def test_capturing_a_view_returns_the_gate_verdict(station: dict[str, object]) -> None:
    session_id = _open_session()
    body = _send(session_id, "front", station["front"])
    assert body["ok"] is True
    assert body["reasons"] == []
    assert body["views"] == ["front"]

    stored = client.get(f"/api/sessions/{session_id}/views/front/gate").json()
    assert stored == {"ok": True, "reasons": [], "captured": True}
    missing = client.get(f"/api/sessions/{session_id}/views/side_left/gate").json()
    assert missing["captured"] is False
    assert missing["reasons"][0]["code"] == "NOT_CAPTURED"


def test_an_unknown_view_never_reaches_a_path(station: dict[str, object]) -> None:
    session_id = _open_session()
    assert client.get(f"/api/sessions/{session_id}/views/..%2F..%2Fetc/gate").status_code == 404
    assert client.get(f"/api/sessions/{session_id}/artifacts/front/..%2Fsubject").status_code == 404


def test_an_unaccepted_content_type_is_refused(station: dict[str, object]) -> None:
    session_id = _open_session()
    file = ("image", ("front.gif", b"GIF89a", "image/gif"))
    response = client.post(f"/api/sessions/{session_id}/views/front", files=[file])
    assert response.status_code == 415
    assert "not accepted" in response.json()["detail"]


def test_every_stage_the_ui_shows_has_a_manifest_entry_and_a_file(
    station: dict[str, object],
) -> None:
    """Brief 20.3: a displayed stage must correspond to a readable file."""
    session_id = _capture_all(station, ("front", "side_left"))
    client.post(f"/api/sessions/{session_id}/process")

    payload = client.get(f"/api/sessions/{session_id}/artifacts").json()
    assert [stage["stage_id"] for stage in payload["stages"]] == list(STAGE_IDS)
    assert payload["manifest"], "capture and processing both write entries"
    for entry in payload["manifest"]:
        response = client.get(
            f"/api/sessions/{session_id}/artifacts/{entry['view']}/{entry['stage_id']}"
        )
        assert response.status_code == 200, entry
        assert response.headers["content-type"].startswith("image/")
        assert len(response.content) > 0
    for view in ("front", "side_left"):
        produced = {e["stage_id"] for e in payload["manifest"] if e["view"] == view}
        assert produced == set(STAGE_IDS), f"{view} is missing {set(STAGE_IDS) - produced}"


def test_the_process_status_names_what_is_still_missing(
    station: dict[str, object],
) -> None:
    session_id = _capture_all(station, ("front",))
    status = client.get(f"/api/sessions/{session_id}/process/status").json()

    assert status["measured"] is False
    assert status["views"]["front"]["stages_missing"] == [
        "11_landmarks",
        "12_measurement_overlay",
    ]
    assert status["views"]["front"]["gate"]["ok"] is True
    assert "side_left" not in status["views"]


def test_an_upright_upload_is_not_turned_again(station: dict[str, object]) -> None:
    """The live camera is mounted portrait; a photograph off disk already is not.

    The station fixture rotates nothing, so the flag is checked against the
    configured rotation the app would otherwise apply.
    """
    session_id = _open_session()
    file = ("image", ("front.jpg", _png(_photo(station["front"].mask)), "image/png"))
    response = client.post(
        f"/api/sessions/{session_id}/views/front", files=[file], data={"upright": "true"}
    )
    assert response.status_code == 200, response.text
    assert response.json()["ok"] is True, response.json()["reasons"]


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


def test_processing_returns_landmarks_measurements_and_the_gate(
    station: dict[str, object],
) -> None:
    session_id = _capture_all(station, ("front", "side_left"))
    body = client.post(f"/api/sessions/{session_id}/process").json()

    assert len(body["primary_keys"]) == 26
    assert set(body["primary_keys"]) <= set(body["measurements"])
    assert body["measurements"]["height"]["confidence"] == "high"
    assert body["measurements"]["bust_points"]["source"] == "derived_pattern_parameter"
    assert {mark["view"] for mark in body["landmarks"]} == {"front", "side_left"}

    waist = body["measurements"]["waist"]
    assert waist["method"] == "montazerian_ellipse"
    assert waist["validated_against_manual"] is False
    assert waist["inputs"]["front_breadth_mm"] > 0
    assert body["export"]["profile"] == "full"


def test_the_export_gate_blocks_an_unaccepted_estimate(station: dict[str, object]) -> None:
    """Brief 15.3: an estimate has to be accepted before a pattern file is written."""
    session_id = _open_session(profile="core_upper_body")
    for view in ("front", "side_left"):
        _send(session_id, view, station["front"] if view == "front" else station["sides"][view])
    body = client.post(f"/api/sessions/{session_id}/process").json()
    assert body["export"]["ok"] is True, body["export"]["blocked"]

    full = _capture_all(station, ("front", "side_left"))
    blocked = client.post(f"/api/sessions/{full}/process").json()["export"]
    assert blocked["ok"] is False
    assert "bust_points" in blocked["blocked"]
    assert client.get(f"/api/sessions/{full}/export/body").status_code == 404
    assert client.get(f"/api/sessions/{full}/export/report").status_code == 200, (
        "the report must be available even for a blocked session"
    )

    client.put(
        f"/api/sessions/{full}/measurements/bust_points",
        json={"value": 17.4, "note": "measured by tape"},
    )
    after = client.put(
        f"/api/sessions/{full}/measurements/bum_points",
        json={"value": 18.9, "note": "measured by tape"},
    ).json()
    assert after["export"]["ok"] is True, after["export"]["blocked"]
    assert client.get(f"/api/sessions/{full}/export/body").status_code == 200


def test_measuring_without_a_side_view_explains_why(station: dict[str, object]) -> None:
    session_id = _capture_all(station, ("front",))
    response = client.post(f"/api/sessions/{session_id}/process")
    assert response.status_code == 422
    assert "side view" in response.json()["detail"]
    assert "Traceback" not in response.text


def test_edited_landmarks_recompute_through_the_api(station: dict[str, object]) -> None:
    session_id = _capture_all(station, ("front", "side_left"))
    first = client.post(f"/api/sessions/{session_id}/process").json()

    edited = [
        {**mark, "y": mark["y"] - 50, "auto": False} if mark["name"] == "waist_row" else mark
        for mark in first["landmarks"]
    ]
    second = client.put(f"/api/sessions/{session_id}/landmarks", json=edited).json()

    assert second["measurements"]["waist"]["value"] != first["measurements"]["waist"]["value"]
    assert second["measurements"]["height"]["value"] == first["measurements"]["height"]["value"]


def test_resetting_landmarks_throws_away_every_edit(station: dict[str, object]) -> None:
    """Brief 11.4: reset to automatic, within the review session."""
    session_id = _capture_all(station, ("front", "side_left"))
    first = client.post(f"/api/sessions/{session_id}/process").json()
    edited = [
        {**mark, "y": mark["y"] - 50, "auto": False} if mark["name"] == "waist_row" else mark
        for mark in first["landmarks"]
    ]
    client.put(f"/api/sessions/{session_id}/landmarks", json=edited)
    client.put(
        f"/api/sessions/{session_id}/measurements/bust_points",
        json={"value": 17.4, "note": "tape"},
    )

    back = client.post(f"/api/sessions/{session_id}/landmarks/reset").json()

    assert all(mark["auto"] for mark in back["landmarks"])
    assert back["measurements"]["waist"]["value"] == first["measurements"]["waist"]["value"]
    assert back["measurements"]["bust_points"]["edited"] is False


def test_the_exports_are_yaml_and_only_appear_after_processing(
    station: dict[str, object],
) -> None:
    session_id = _open_session(profile="core_upper_body")
    for view in ("front", "side_left"):
        _send(session_id, view, station["front"] if view == "front" else station["sides"][view])
    assert client.get(f"/api/sessions/{session_id}/export/body").status_code == 404

    client.post(f"/api/sessions/{session_id}/process")
    body = yaml.safe_load(client.get(f"/api/sessions/{session_id}/export/body").text)
    report = yaml.safe_load(client.get(f"/api/sessions/{session_id}/export/report").text)
    extended = yaml.safe_load(client.get(f"/api/sessions/{session_id}/export/extended").text)

    assert len(body["body"]) == 26
    assert body["metadata"]["session_id"] == session_id
    assert report["metadata"]["validated_against_manual"] is False
    assert report["scale"]["stature_cm"] == HEIGHT_CM
    assert report["artifacts"], "the report links every artifact on disk"
    assert "chest" in extended["measurements"]


def test_the_session_csv_carries_the_provenance_columns(
    station: dict[str, object],
) -> None:
    session_id = _capture_all(station, ("front", "side_left"))
    client.post(f"/api/sessions/{session_id}/process")
    text = client.get(f"/api/sessions/{session_id}/export/csv").text

    header = text.splitlines()[0].split(",")
    assert header[:4] == ["name", "value", "unit", "status"]
    assert "validated_against_manual" in header
    assert "montazerian_ellipse" in text


def test_a_session_detail_names_the_profile_and_its_required_fields(
    station: dict[str, object],
) -> None:
    session_id = _capture_all(station, ("front",))
    body = client.get(f"/api/sessions/{session_id}").json()

    assert body["profile"] == "full"
    assert body["scale_mode"] == "stature"
    assert body["height_cm"] == HEIGHT_CM
    assert body["views_captured"] == ["front"]
    assert "waist" in body["required_fields"]
    assert client.get(f"/api/sessions/{uuid.uuid4().hex}").status_code == 404


def test_a_stage_that_was_never_produced_is_a_404(station: dict[str, object]) -> None:
    session_id = _capture_all(station, ("front",))
    assert (
        client.get(f"/api/sessions/{session_id}/artifacts/front/12_measurement_overlay").status_code
        == 404
    )
    assert client.get(f"/api/sessions/{session_id}/artifacts/front/99_nope").status_code == 404


def test_deleting_a_session_is_final(station: dict[str, object]) -> None:
    session_id = _capture_all(station, ("front", "side_left"))
    assert client.delete(f"/api/sessions/{session_id}").json() == {"deleted": True}
    assert client.delete(f"/api/sessions/{session_id}").status_code == 404


def test_the_review_page_is_served_under_its_session_path(station: dict[str, object]) -> None:
    session_id = _capture_all(station, ("front", "side_left"))
    response = client.get(f"/review/{session_id}")
    assert response.status_code == 200
    assert 'id="review"' in response.text
    assert 'id="process"' in client.get(f"/process/{session_id}").text


def test_measuring_twice_keeps_edited_landmarks(station: dict[str, object]) -> None:
    """The review page measures on load; that must not discard the tailor's work."""
    session_id = _capture_all(station, ("front", "side_left"))
    first = client.post(f"/api/sessions/{session_id}/process").json()
    edited = [
        {**mark, "y": mark["y"] - 50, "auto": False} if mark["name"] == "waist_row" else mark
        for mark in first["landmarks"]
    ]
    client.put(f"/api/sessions/{session_id}/landmarks", json=edited)

    again = client.post(f"/api/sessions/{session_id}/measurements/recompute").json()
    waist = next(mark for mark in again["landmarks"] if mark["name"] == "waist_row")
    assert waist["auto"] is False
    assert waist["y"] == next(
        mark["y"] for mark in first["landmarks"] if mark["name"] == "waist_row"
    ) - 50


def test_a_new_capture_clears_stale_landmarks(station: dict[str, object]) -> None:
    session_id = _capture_all(station, ("front", "side_left"))
    client.post(f"/api/sessions/{session_id}/process")
    assert (session_module.session_dir(session_id) / "landmarks.json").exists()

    _send(session_id, "front", station["front"])
    assert not (session_module.session_dir(session_id) / "landmarks.json").exists()


def test_a_typed_value_is_recorded_as_the_operators(station: dict[str, object]) -> None:
    session_id = _capture_all(station, ("front", "side_left"))
    client.post(f"/api/sessions/{session_id}/process")

    body = client.put(
        f"/api/sessions/{session_id}/measurements/bust_points",
        json={"value": 17.4, "note": "measured by tape"},
    ).json()
    entry = body["measurements"]["bust_points"]

    assert entry["value"] == 17.4
    assert entry["edited"] is True
    assert entry["source"] == "operator_input"
    assert entry["status"] == "manual_input"
    assert entry["auto"] is False
    assert entry["note"] == "measured by tape"

    records = yaml.safe_load(client.get(f"/api/sessions/{session_id}/export/report").text)
    by_name = {record["name"]: record for record in records["measurements"]}
    assert by_name["bust_points"]["note"] == "measured by tape"


def test_a_typed_value_without_a_note_is_refused(station: dict[str, object]) -> None:
    """Brief 16.3: a direct numeric override requires a note."""
    session_id = _capture_all(station, ("front", "side_left"))
    client.post(f"/api/sessions/{session_id}/process")

    response = client.put(
        f"/api/sessions/{session_id}/measurements/bust_points",
        json={"value": 17.4, "note": "   "},
    )
    assert response.status_code == 422
    assert "note is required" in response.json()["detail"]


def test_a_typed_value_survives_a_later_landmark_drag(station: dict[str, object]) -> None:
    session_id = _capture_all(station, ("front", "side_left"))
    first = client.post(f"/api/sessions/{session_id}/process").json()
    client.put(
        f"/api/sessions/{session_id}/measurements/bum_points",
        json={"value": 19.9, "note": "tape"},
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
    client.post(f"/api/sessions/{session_id}/process")

    listed = client.get("/api/sessions").json()
    assert [row["id"] for row in listed] == [session_id]
    assert listed[0]["status"] == "measured"
    assert listed[0]["views"] == ["front", "side_left"]
    assert client.get("/api/sessions?status=confirmed").json() == []


def test_confirming_starts_the_retention_clock(station: dict[str, object]) -> None:
    session_id = _capture_all(station, ("front", "side_left"))
    client.post(f"/api/sessions/{session_id}/process")

    body = client.post(f"/api/sessions/{session_id}/confirm").json()

    assert body["status"] == "confirmed"
    assert body["images_purge_after"] > date.today().isoformat()
    assert not (session_module.session_dir(session_id) / "debug").exists(), (
        "keep_debug_after_confirm is false"
    )


def test_confirming_an_unknown_session_is_a_404(station: dict[str, object]) -> None:
    assert client.post(f"/api/sessions/{uuid.uuid4().hex}/confirm").status_code == 404


def test_the_csv_export_covers_confirmed_sessions_only(station: dict[str, object]) -> None:
    session_id = _open_session(profile="core_upper_body")
    for view in ("front", "side_left"):
        _send(session_id, view, station["front"] if view == "front" else station["sides"][view])
    client.post(f"/api/sessions/{session_id}/process")
    assert "S014" not in client.get("/api/export.csv").text

    client.post(f"/api/sessions/{session_id}/confirm")
    text = client.get("/api/export.csv").text

    assert "S014" in text
    assert session_id in text
    assert text.splitlines()[0].count("_edited") == 26


def test_edits_reach_the_audit_trail(station: dict[str, object]) -> None:
    session_id = _capture_all(station, ("front", "side_left"))
    first = client.post(f"/api/sessions/{session_id}/process").json()
    edited = [
        {**mark, "y": mark["y"] - 40, "auto": False} if mark["name"] == "waist_row" else mark
        for mark in first["landmarks"]
    ]
    client.put(f"/api/sessions/{session_id}/landmarks", json=edited)
    client.put(
        f"/api/sessions/{session_id}/measurements/bust_points",
        json={"value": 17.4, "note": "tape"},
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
