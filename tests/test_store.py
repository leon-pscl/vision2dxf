"""The session index, consent record and audit trail (brief 10.8)."""

from datetime import date, timedelta
from pathlib import Path

import pytest
import yaml

from bodymeasure import store


@pytest.fixture(autouse=True)
def database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the store at a temporary data directory."""
    monkeypatch.setattr(store, "data_dir", lambda: tmp_path)
    monkeypatch.setattr(store, "sessions_root", lambda: tmp_path / "sessions")
    store.initialise()
    return tmp_path


def _session(code: str = "S014", height_cm: float = 164.5) -> str:
    return store.create_session(code, "F", "2026-10-11", height_cm)


def test_the_measured_height_is_stored_with_the_session() -> None:
    session_id = _session(height_cm=171.4)
    assert store.get_session(session_id)["height_cm"] == pytest.approx(171.4)


def test_a_database_from_before_height_scaling_still_opens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``CREATE TABLE IF NOT EXISTS`` leaves an old table alone; the column is added."""
    legacy = tmp_path / "legacy"  # the autouse fixture already made the current one
    monkeypatch.setattr(store, "data_dir", lambda: legacy)
    monkeypatch.setattr(store, "sessions_root", lambda: legacy / "sessions")
    with store.connect() as connection:
        connection.execute(
            "CREATE TABLE sessions (id TEXT PRIMARY KEY, subject_code TEXT NOT NULL,"
            " sex TEXT, consent_at TEXT NOT NULL, consent_version TEXT NOT NULL,"
            " created_at TEXT NOT NULL, status TEXT NOT NULL, views TEXT NOT NULL DEFAULT '[]',"
            " confirmed_at TEXT, images_purge_after TEXT)"
        )

    store.initialise()
    assert store.get_session(_session(height_cm=160.0))["height_cm"] == pytest.approx(160.0)


def _populate(session_id: str, body: dict[str, float] | None = None) -> Path:
    """Give a session the files a measured one would have."""
    folder = store.session_folder(session_id)
    folder.mkdir(parents=True, exist_ok=True)
    views = folder / "views" / "front"
    views.mkdir(parents=True, exist_ok=True)
    (views / "00_raw.jpg").write_bytes(b"jpeg")
    (views / "07_mask_clean.png").write_bytes(b"png")
    predictions = folder / "predictions"
    predictions.mkdir(exist_ok=True)
    (predictions / "front_keypoints.json").write_text("{}", encoding="utf-8")
    (folder / "landmarks.json").write_text("[]", encoding="utf-8")
    (folder / "manifest.json").write_text("[]", encoding="utf-8")
    (folder / "debug").mkdir(exist_ok=True)
    (folder / "debug" / "front_gate.png").write_bytes(b"png")
    (folder / "body_measurements.yaml").write_text(
        yaml.safe_dump({"body": body or {"bust": 99.8, "waist": 84.3}}), encoding="utf-8"
    )
    (folder / "measurement_report.yaml").write_text(
        yaml.safe_dump({"measurements": [{"name": "bust", "edited": True}]}), encoding="utf-8"
    )
    return folder


def test_a_new_session_records_consent_and_hides_the_subject_code() -> None:
    session_id = _session()
    row = store.get_session(session_id)

    assert row["subject_code"] == "S014"
    assert row["status"] == "capturing"
    assert row["consent_version"] == "2026-10-11"
    assert row["consent_at"] and row["created_at"]
    assert "S014" not in session_id
    assert len(session_id) == 32, "a UUID4 hex, so no personal data reaches a path or a log"


def test_an_unconfirmed_session_is_scheduled_for_an_early_purge() -> None:
    row = store.get_session(_session())
    expected = (date.today() + timedelta(days=7)).isoformat()
    assert row["images_purge_after"] == expected


def test_views_accumulate_without_duplicating() -> None:
    session_id = _session()
    store.add_view(session_id, "front")
    store.add_view(session_id, "side_left")
    store.add_view(session_id, "front")

    assert store.get_session(session_id)["views"] == ["front", "side_left"]


def test_adding_a_view_to_a_missing_session_is_an_error() -> None:
    with pytest.raises(FileNotFoundError, match="no such session"):
        store.add_view("0" * 32, "front")


def test_confirming_stamps_the_retention_date() -> None:
    session_id = _session()
    store.set_status(session_id, "measured")
    store.set_status(session_id, "confirmed")
    row = store.get_session(session_id)

    assert row["status"] == "confirmed"
    assert row["confirmed_at"] is not None
    assert row["images_purge_after"] == (date.today() + timedelta(days=30)).isoformat()


def test_an_unknown_status_is_refused() -> None:
    with pytest.raises(ValueError, match="status must be one of"):
        store.set_status(_session(), "finished")


def test_edits_are_kept_in_order_with_their_notes() -> None:
    session_id = _session()
    store.log_edit(session_id, "landmark", "front.waist_row", [10, 20], [10, 40])
    store.log_edit(session_id, "value", "bust_points", 16.9, 17.4, "measured by tape")
    trail = store.list_edits(session_id)

    assert [entry["kind"] for entry in trail] == ["landmark", "value"]
    assert trail[1]["note"] == "measured by tape"
    assert trail[1]["old_value"] == "16.9"


def test_sessions_can_be_listed_and_filtered() -> None:
    first = _session("S001")
    second = _session("S002")
    store.set_status(second, "confirmed")

    assert {row["id"] for row in store.list_sessions()} == {first, second}
    assert [row["id"] for row in store.list_sessions(status="confirmed")] == [second]
    assert store.list_sessions(status="exported") == []


def test_a_deleted_session_leaves_only_one_deletion_row() -> None:
    """Brief 10.8: no folder, no session row, no edit rows, one deletion row."""
    session_id = _session()
    folder = _populate(session_id)
    store.log_edit(session_id, "value", "bust", 99.8, 100.0, "tape")
    store.set_status(session_id, "confirmed")

    assert store.delete_session(session_id, reason="user_request") is True

    assert not folder.exists()
    assert store.get_session(session_id) is None
    assert store.list_edits(session_id) == [], "edits cascade with the session"
    deletions = store.list_deletions()
    assert len(deletions) == 1
    assert deletions[0]["scope"] == "session"
    assert deletions[0]["reason"] == "user_request"
    assert deletions[0]["session_id"] == session_id


def test_deleting_a_session_twice_records_one_deletion() -> None:
    session_id = _session()
    assert store.delete_session(session_id, reason="user_request") is True
    assert store.delete_session(session_id, reason="user_request") is False
    assert len(store.list_deletions()) == 1


def test_purge_images_keeps_the_measurements() -> None:
    """Brief 10.8: the photos go, body_measurements.yaml stays."""
    session_id = _session()
    folder = _populate(session_id)

    assert store.purge_images(today=date.today() + timedelta(days=8)) == 1

    assert not (folder / "views").exists()
    assert not (folder / "predictions").exists()
    assert not (folder / "debug").exists()
    assert (folder / "manifest.json").exists(), "the index of what was there stays"
    assert (folder / "body_measurements.yaml").exists()
    assert (folder / "measurement_report.yaml").exists()
    assert (folder / "landmarks.json").exists()
    assert store.list_deletions()[0]["scope"] == "images"
    assert store.list_deletions()[0]["reason"] == "retention"


def test_purge_images_leaves_sessions_inside_their_retention() -> None:
    session_id = _session()
    folder = _populate(session_id)

    assert store.purge_images(today=date.today()) == 0
    assert (folder / "views" / "front" / "00_raw.jpg").exists()


def test_a_crash_after_the_tmp_write_leaves_the_previous_file(tmp_path: Path) -> None:
    """Brief 10.8: an interrupted write must not truncate what was there."""
    target = tmp_path / "body_measurements.yaml"
    store.write_atomic(target, "body:\n  bust: 99.8\n")

    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text("half a fi", encoding="utf-8")  # the crash happens here

    assert target.read_text(encoding="utf-8") == "body:\n  bust: 99.8\n"
    assert temporary.exists(), "the partial file is left behind, not swapped in"


def test_write_atomic_replaces_cleanly_and_handles_bytes(tmp_path: Path) -> None:
    target = tmp_path / "front.jpg"
    store.write_atomic(target, b"first")
    store.write_atomic(target, b"second")

    assert target.read_bytes() == b"second"
    assert not target.with_name(target.name + ".tmp").exists()


def test_export_csv_skips_unconfirmed_sessions(tmp_path: Path) -> None:
    """Brief 10.8: only sessions a tailor has confirmed reach the analysis file."""
    confirmed = _session("S001")
    _populate(confirmed)
    store.set_status(confirmed, "confirmed")
    unconfirmed = _session("S002")
    _populate(unconfirmed)

    path = store.export_csv(tmp_path / "export.csv")
    text = path.read_text(encoding="utf-8")

    assert "S001" in text
    assert "S002" not in text
    assert confirmed in text
    assert unconfirmed not in text


def test_export_csv_carries_the_26_keys_and_their_edited_flags(tmp_path: Path) -> None:
    session_id = _session()
    _populate(session_id)
    store.set_status(session_id, "confirmed")

    path = store.export_csv(tmp_path / "export.csv")
    header, row = path.read_text(encoding="utf-8").splitlines()[:2]

    assert header.startswith("session_id,subject_code,sex,")
    assert header.count("_edited") == 26
    assert len(header.split(",")) == 3 + 26 + 26
    assert len(row.split(",")) == len(header.split(","))
    assert "True" in row, "the edited flag from the report is carried over"


def test_a_session_id_that_is_not_a_uuid_cannot_reach_the_filesystem() -> None:
    for bad in ("../secrets", "..", "S014-20260101", ""):
        with pytest.raises(ValueError, match="UUID"):
            store.session_folder(bad)
