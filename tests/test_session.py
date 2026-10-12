"""End-to-end: capture, measure, export, delete.

``analyse`` is stubbed throughout. The models are exercised in M3's tests; what
matters here is the session plumbing, and YOLO cannot see a drawn silhouette
anyway.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from tests.synthetic.people import as_captured, make_person, make_side

from bodymeasure import artifacts, store
from bodymeasure import session as session_module
from bodymeasure.config import Calibration, load_schema
from bodymeasure.landmarks import as_dict, load_landmarks
from bodymeasure.session import (
    NO_DISTORTION_WARNING,
    Subject,
    captured_views,
    create_session,
    export,
    find_landmarks,
    load_subject,
    measure_session,
    save_view,
    session_dir,
)
from bodymeasure.vision import ViewAnalysis

HEIGHT_CM = 164.5  # the drawn figure spans 1645 rows at 1 mm/px


def _report(session_id: str) -> dict:
    """Return the provenance report of a session."""
    path = session_dir(session_id) / "measurement_report.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _photo(mask: np.ndarray) -> np.ndarray:
    """Return a grey-on-white image the size of the silhouette."""
    return np.where(mask[..., None], 90, 230).astype(np.uint8).repeat(3, axis=2)


@pytest.fixture
def station(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Return a station with its sessions and vision under test control."""
    front = make_person()
    sides = {name: make_side(name) for name in ("side_left", "side_right")}

    monkeypatch.setattr(store, "sessions_root", lambda: tmp_path / "sessions")
    monkeypatch.setattr(store, "data_dir", lambda: tmp_path)
    # An uncorrected station: frames are rotated but not undistorted.
    monkeypatch.setattr(session_module, "load_calibration", lambda: Calibration(rotation=0))
    store.initialise()

    def fake_analyse(image: np.ndarray) -> ViewAnalysis:
        """Return the figure whose silhouette matches the padded image."""
        for figure in (front, *sides.values()):
            if np.array_equal(image, as_captured(_photo(figure.mask))):
                mask, keypoints = as_captured(figure.mask, figure.keypoints)
                return ViewAnalysis(mask, keypoints, 1.0, 1.0)
        raise ValueError("No person found in the frame")

    monkeypatch.setattr(session_module, "analyse", fake_analyse)
    return {"front": front, "sides": sides}


def _capture(
    station: dict[str, object], views: tuple[str, ...], profile: str = "core_upper_body"
) -> str:
    subject = Subject(
        code="S014",
        sex="F",
        consent=True,
        height_cm=HEIGHT_CM,
        notes="tight clothing",
        profile=profile,
    )
    session_id = create_session(subject)
    for view in views:
        figure = station["front"] if view == "front" else station["sides"][view]
        save_view(session_id, view, _photo(figure.mask))
    return session_id


def test_consent_is_required_before_any_capture() -> None:
    with pytest.raises(ValueError, match="consent is required"):
        create_session(Subject(code="S014", sex="F", consent=False, height_cm=HEIGHT_CM))


def test_a_height_outside_the_range_is_refused() -> None:
    with pytest.raises(ValueError, match="height must be 100-220 cm"):
        create_session(Subject(code="S014", sex="F", consent=True, height_cm=95.0))


def test_a_subject_code_cannot_escape_the_sessions_folder() -> None:
    with pytest.raises(ValueError, match="letters, digits"):
        create_session(Subject(code="../../etc", sex="F", consent=True, height_cm=HEIGHT_CM))
    with pytest.raises(ValueError, match="UUID"):
        session_dir("../secrets")


def test_a_session_records_its_subject_and_height(station: dict[str, object]) -> None:
    session_id = _capture(station, ("front",))
    subject = load_subject(session_id)
    assert subject.code == "S014"
    assert subject.consent is True
    assert subject.height_cm == HEIGHT_CM
    assert "S014" not in session_id, "a subject code must never appear in a path"
    assert store.get_session(session_id)["subject_code"] == "S014"
    assert store.get_session(session_id)["height_cm"] == HEIGHT_CM
    assert store.get_session(session_id)["status"] == "capturing"


def test_capturing_a_view_writes_every_stage_and_manifests_it(
    station: dict[str, object],
) -> None:
    """Brief 7 and 8: a measurement the operator cannot trace is not reviewable."""
    session_id = _capture(station, ("front",))
    folder = session_dir(session_id)

    entries = artifacts.manifest(session_id)
    assert [entry["stage_id"] for entry in entries] == list(artifacts.STAGE_IDS[:11]), (
        "capture produces stages 00 to 10; landmarks and measurements come later"
    )
    for entry in entries:
        path = folder / str(entry["path"])
        assert path.exists(), entry["path"]
        assert path.stat().st_size > 0
        assert entry["algorithm"] and entry["created_at"]
        assert entry["width"] > 0 and entry["height"] > 0
    assert (folder / "predictions" / "front_detections.json").exists()
    assert (folder / "predictions" / "front_keypoints.json").exists()
    assert (folder / "debug" / "front_gate.png").exists()
    assert captured_views(session_id) == ["front"]


def test_the_raw_capture_is_never_the_measured_image(station: dict[str, object]) -> None:
    """Stage 00 must survive orientation, padding and enhancement untouched."""
    session_id = _capture(station, ("front",))
    raw = cv2.imread(str(artifacts.stage_path(session_id, "front", "00_raw")))
    enhanced = cv2.imread(str(artifacts.stage_path(session_id, "front", "03_enhanced")))
    assert raw.shape[:2] == _photo(station["front"].mask).shape[:2]
    assert enhanced.shape[0] > raw.shape[0], "the enhanced frame carries the border"


def test_a_retake_leaves_no_stage_from_the_previous_capture(
    station: dict[str, object],
) -> None:
    session_id = _capture(station, ("front", "side_left"))
    find_landmarks(session_id)
    assert "11_landmarks" in {e["stage_id"] for e in artifacts.manifest(session_id)}

    save_view(session_id, "front", _photo(station["front"].mask))
    stages = {e["stage_id"] for e in artifacts.manifest(session_id)}
    assert "11_landmarks" not in stages, "a retake invalidates the landmarks found before"
    assert "00_raw" in stages
    assert not artifacts.stage_path(session_id, "side_left", "11_landmarks").exists(), (
        "a stale overlay on the view that was not retaken would mislead the operator"
    )


def test_a_good_front_capture_passes_the_gate(station: dict[str, object]) -> None:
    subject = Subject(code="S014", sex="F", consent=True, height_cm=HEIGHT_CM)
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
    assert artifacts.stage_path(session_id, "front", "11_landmarks").exists()
    assert artifacts.stage_path(session_id, "side_left", "11_landmarks").exists()


def test_each_view_stores_the_scale_its_own_rows_imply(
    station: dict[str, object],
) -> None:
    """landmarks.json carries the mm/px per view, beside the rows that set it."""
    session_id = _capture(station, ("front", "side_left"))
    find_landmarks(session_id)
    folder = session_dir(session_id)
    stored = json.loads((folder / "landmarks.json").read_text(encoding="utf-8"))
    rows = {(mark.name, mark.view): mark for mark in load_landmarks(folder / "landmarks.json")}

    assert set(stored["views"]) == {"front", "side_left"}
    for view, scale in stored["views"].items():
        floor = rows[("floor_row", view)].y
        head = rows[("head_row", view)].y
        assert scale["floor_row"] == floor
        assert scale["head_row"] == head
        assert scale["mm_per_px"] == pytest.approx(HEIGHT_CM * 10 / (floor - head), abs=1e-6)


def test_measuring_needs_a_side_view(station: dict[str, object]) -> None:
    session_id = _capture(station, ("front",))
    with pytest.raises(ValueError, match="at least one side view"):
        measure_session(session_id)


def test_front_plus_one_side_exports_the_three_files(station: dict[str, object]) -> None:
    session_id = _capture(station, ("front", "side_left"))
    folder = session_dir(session_id)
    gate = export(session_id, measure_session(session_id))
    assert gate.ok, gate.reasons

    body = yaml.safe_load((folder / "body_measurements.yaml").read_text(encoding="utf-8"))
    assert set(body) == {"metadata", "body"}
    assert set(body["body"]) == set(load_schema())
    assert not any(key.startswith("_") for key in body["body"]), "pygarment recomputes those"
    assert body["body"]["height"] == HEIGHT_CM
    assert body["metadata"]["unit"] == "cm"
    assert body["metadata"]["validation_status"] == "unvalidated_against_manual_anthropometry"

    extended = yaml.safe_load((folder / "extended_measurements.yaml").read_text(encoding="utf-8"))
    assert "chest" in extended["measurements"], "chest is not a primary GarmentCode key"
    assert not set(extended["measurements"]) & set(load_schema()), (
        "brief 3.3: extended keys must never leak into the pattern file"
    )

    records = {record["name"]: record for record in _report(session_id)["measurements"]}
    assert records["waist"]["method"] == "montazerian_ellipse"
    assert records["waist"]["formula"].startswith("ellipse_equation_")
    assert records["waist"]["inputs"]["front_breadth_mm"] > 0
    assert records["waist"]["inputs"]["side_depth_mm"] > 0
    assert records["waist"]["validated_against_manual"] is False
    assert records["bust_points"]["source"] == "derived_pattern_parameter"
    assert records["bust_points"]["status"] == "estimated_pending_acceptance", (
        "a population ratio is never an acquired measurement"
    )
    assert records["height"]["source"] == "operator_input"


def test_an_uncorrected_lens_is_warned_about_in_the_report(
    station: dict[str, object],
) -> None:
    """The station fixture has no lens model: frames are measured as they arrive."""
    session_id = _capture(station, ("front", "side_left"))
    export(session_id, measure_session(session_id))

    report = _report(session_id)
    assert NO_DISTORTION_WARNING in report["warnings"]


def test_a_saved_lens_model_keeps_the_warning_off(
    station: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        session_module,
        "load_calibration",
        lambda: Calibration(
            camera_matrix=[[1400.0, 0.0, 450.0], [0.0, 1400.0, 1000.0], [0.0, 0.0, 1.0]],
            dist_coeffs=[0.0, 0.0, 0.0, 0.0, 0.0],
            rotation=0,
        ),
    )
    session_id = _capture(station, ("front", "side_left"))
    export(session_id, measure_session(session_id))

    report = _report(session_id)
    assert NO_DISTORTION_WARNING not in report["warnings"]


def test_front_plus_both_sides_also_exports(station: dict[str, object]) -> None:
    session_id = _capture(station, ("front", "side_left", "side_right"))
    gate = export(session_id, measure_session(session_id))

    body = yaml.safe_load(
        (session_dir(session_id) / "body_measurements.yaml").read_text(encoding="utf-8")
    )
    assert len(body["body"]) == 26, "the pattern file carries the primary keys and no more"
    assert set(body["body"]) == set(load_schema())
    assert gate.ok, gate.reasons
    views = {view for record in _report(session_id)["measurements"] for view in record["views"]}
    assert {"side_left", "side_right"} <= views


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
    assert list(folder.glob("views/*/*.jpg"))

    assert store.delete_session(session_id, reason="user_request") is True
    assert not folder.exists()
    assert store.delete_session(session_id, reason="user_request") is False


def test_capturing_into_a_missing_session_is_refused(
    station: dict[str, object],
) -> None:
    session_id = _capture(station, ("front",))
    store.delete_session(session_id, reason="user_request")
    with pytest.raises(FileNotFoundError, match="no such session"):
        save_view(session_id, "front", _photo(station["front"].mask))


def test_a_frame_with_nobody_in_it_is_refused(station: dict[str, object]) -> None:
    session_id = _capture(station, ("front",))
    with pytest.raises(ValueError, match="No person found"):
        save_view(session_id, "front", np.full((200, 200, 3), 230, dtype=np.uint8))


def test_the_full_profile_blocks_export_until_the_estimates_are_accepted(
    station: dict[str, object],
) -> None:
    """Brief 15.3: an estimated_pending_acceptance value blocks a pattern export."""
    session_id = _capture(station, ("front", "side_left"), profile="full")
    folder = session_dir(session_id)
    gate = export(session_id, measure_session(session_id))

    assert gate.ok is False
    assert set(gate.blocked) == {"bust_points", "bum_points"}
    assert not (folder / "body_measurements.yaml").exists(), (
        "a blocked export must not leave the previous run's numbers behind"
    )
    assert (folder / "measurement_report.yaml").exists(), "the report is always written"

    session_module.save_overrides(
        session_id,
        {
            "bust_points": {"value": 17.4, "note": "measured by tape"},
            "bum_points": {"value": 18.9, "note": "measured by tape"},
        },
    )
    gate = export(session_id, measure_session(session_id))

    assert gate.ok, gate.reasons
    body = yaml.safe_load((folder / "body_measurements.yaml").read_text(encoding="utf-8"))
    assert body["body"]["bust_points"] == 17.4
