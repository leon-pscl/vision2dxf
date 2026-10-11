"""The session index, consent record and audit trail (brief 10).

Files on disk hold the images and the YAML the pipeline consumes; this SQLite
database holds only the index, consent and audit trail, and never an image or a
YAML body. **Every** database access and every session-folder deletion goes
through this module: nothing else opens ``app.db`` or removes a session.

Connections are opened per call rather than shared. FastAPI runs synchronous
endpoints in a thread pool, and a SQLite connection may not cross threads; WAL
makes the concurrent readers cheap.
"""

from __future__ import annotations

import csv
import io
import json
import os
import shutil
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from bodymeasure.config import ROOT, load_app_config, load_schema

SCHEMA_VERSION = 1
STATUSES = ("capturing", "measured", "confirmed", "exported")
EXPORTABLE = ("confirmed", "exported")
IMAGE_SUFFIXES = (".jpg", "_mask.png", "_kp.json")

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);

CREATE TABLE IF NOT EXISTS sessions (
  id              TEXT PRIMARY KEY,
  subject_code    TEXT NOT NULL,
  sex             TEXT CHECK (sex IN ('M','F','other')),
  height_cm       REAL NOT NULL,
  consent_at      TEXT NOT NULL,
  consent_version TEXT NOT NULL,
  created_at      TEXT NOT NULL,
  status          TEXT NOT NULL CHECK (status IN
                    ('capturing','measured','confirmed','exported')),
  views           TEXT NOT NULL DEFAULT '[]',
  confirmed_at    TEXT,
  images_purge_after TEXT
);

CREATE TABLE IF NOT EXISTS edits (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  at          TEXT NOT NULL,
  kind        TEXT NOT NULL CHECK (kind IN ('landmark','value')),
  target      TEXT NOT NULL,
  old_value   TEXT,
  new_value   TEXT,
  note        TEXT
);

CREATE TABLE IF NOT EXISTS deletions (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id  TEXT NOT NULL,
  at          TEXT NOT NULL,
  scope       TEXT NOT NULL CHECK (scope IN ('session','images')),
  reason      TEXT NOT NULL CHECK (reason IN ('user_request','retention'))
);
"""


def data_dir() -> Path:
    """Return the folder holding the database and the session folders."""
    return ROOT / load_app_config().storage.data_dir


def db_path() -> Path:
    """Return the path of the SQLite database."""
    return data_dir() / "app.db"


def sessions_root() -> Path:
    """Return the folder holding every session folder."""
    return ROOT / load_app_config().paths.sessions


def session_folder(session_id: str) -> Path:
    """Return one session's folder.

    Raises:
        ValueError: if the id is not a UUID4 hex string, which keeps a crafted
            id from escaping the sessions folder.

    """
    try:
        uuid.UUID(hex=session_id)
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError("session id must be a UUID hex string") from exc
    return sessions_root() / session_id


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    """Open the database, creating it if needed, and commit on clean exit."""
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA foreign_keys = ON")
        with connection:
            yield connection


def initialise() -> None:
    """Create the schema if it is missing. Safe to call on every startup."""
    with connect() as connection:
        connection.executescript(SCHEMA)
        stored = connection.execute("SELECT version FROM schema_version").fetchone()
        if stored is None:
            connection.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
        _add_height_cm(connection)


def _add_height_cm(connection: sqlite3.Connection) -> None:
    """Give a database created before height-based scaling its missing column.

    ``CREATE TABLE IF NOT EXISTS`` leaves an existing table alone, so a station
    upgraded in place would otherwise fail on the first insert. Legacy rows get
    0.0: they predate the column and were never measured by height.
    """
    columns = {row["name"] for row in connection.execute("PRAGMA table_info(sessions)")}
    if "height_cm" not in columns:
        connection.execute("ALTER TABLE sessions ADD COLUMN height_cm REAL NOT NULL DEFAULT 0.0")


def write_atomic(path: Path, data: bytes | str) -> Path:
    """Write a file atomically: into ``.tmp``, then rename over the target.

    A crash mid-write leaves the previous file intact rather than a truncated
    one (brief 10.2).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    mode, encoding = ("wb", None) if isinstance(data, bytes) else ("w", "utf-8")
    with temporary.open(mode, encoding=encoding, newline="" if encoding else None) as handle:
        handle.write(data)
    os.replace(temporary, path)
    return path


def create_session(subject_code: str, sex: str, consent_version: str, height_cm: float) -> str:
    """Record consent and the measured height, and return a new session id.

    The id is a UUID4 hex string and is also the folder name, so no subject code
    ever appears in a path, a URL or an access log.
    """
    session_id = uuid.uuid4().hex
    now = _now()
    unconfirmed = load_app_config().storage.unconfirmed_days
    purge_after = (date.today() + timedelta(days=unconfirmed)).isoformat()
    with connect() as connection:
        connection.execute(
            "INSERT INTO sessions (id, subject_code, sex, height_cm, consent_at,"
            " consent_version, created_at, status, views, images_purge_after)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, 'capturing', '[]', ?)",
            (session_id, subject_code, sex, height_cm, now, consent_version, now, purge_after),
        )
    return session_id


def get_session(session_id: str) -> dict[str, Any] | None:
    """Return one session row, or ``None`` if there is no such session."""
    with connect() as connection:
        row = connection.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    return _row_to_dict(row) if row else None


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    entry = dict(row)
    entry["views"] = json.loads(entry["views"])
    return entry


def add_view(session_id: str, view: str) -> None:
    """Record that a view has been captured, without duplicating it."""
    with connect() as connection:
        row = connection.execute(
            "SELECT views FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if row is None:
            raise FileNotFoundError(f"no such session: {session_id}")
        views = json.loads(row["views"])
        if view not in views:
            views.append(view)
        connection.execute(
            "UPDATE sessions SET views = ? WHERE id = ?", (json.dumps(views), session_id)
        )


def set_status(session_id: str, status: str) -> None:
    """Move a session along its lifecycle.

    Confirming stamps ``confirmed_at`` and pushes the image purge date out to the
    retention period the client agreed to.

    Raises:
        ValueError: for an unknown status.
        FileNotFoundError: if the session does not exist.

    """
    if status not in STATUSES:
        raise ValueError(f"status must be one of {', '.join(STATUSES)}")
    storage = load_app_config().storage
    with connect() as connection:
        exists = connection.execute(
            "SELECT 1 FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if exists is None:
            raise FileNotFoundError(f"no such session: {session_id}")
        if status == "confirmed":
            purge_after = (
                date.today() + timedelta(days=storage.retention_days_images)
            ).isoformat()
            connection.execute(
                "UPDATE sessions SET status = ?, confirmed_at = ?, images_purge_after = ?"
                " WHERE id = ?",
                (status, _now(), purge_after, session_id),
            )
        else:
            connection.execute(
                "UPDATE sessions SET status = ? WHERE id = ?", (status, session_id)
            )


def log_edit(
    session_id: str,
    kind: str,
    target: str,
    old: object,
    new: object,
    note: str = "",
) -> None:
    """Append one tailor edit to the audit trail."""
    with connect() as connection:
        connection.execute(
            "INSERT INTO edits (session_id, at, kind, target, old_value, new_value, note)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (session_id, _now(), kind, target, json.dumps(old), json.dumps(new), note),
        )


def list_edits(session_id: str) -> list[dict[str, Any]]:
    """Return the audit trail for one session, oldest first."""
    with connect() as connection:
        rows = connection.execute(
            "SELECT * FROM edits WHERE session_id = ? ORDER BY id", (session_id,)
        ).fetchall()
    return [dict(row) for row in rows]


def list_sessions(status: str | None = None) -> list[dict[str, Any]]:
    """Return sessions, newest first, optionally filtered by status."""
    query = "SELECT * FROM sessions"
    parameters: tuple[str, ...] = ()
    if status is not None:
        query += " WHERE status = ?"
        parameters = (status,)
    query += " ORDER BY created_at DESC"
    with connect() as connection:
        rows = connection.execute(query, parameters).fetchall()
    return [_row_to_dict(row) for row in rows]


def delete_session(session_id: str, reason: str) -> bool:
    """Delete the folder, then the row, and record the deletion.

    The folder goes first: if it cannot be removed, nothing is committed and the
    index still points at data that exists. Returns whether anything was deleted.
    """
    folder = session_folder(session_id)
    if folder.exists():
        shutil.rmtree(folder)
    with connect() as connection:
        cursor = connection.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        existed = cursor.rowcount > 0
        if existed:
            connection.execute(
                "INSERT INTO deletions (session_id, at, scope, reason) VALUES (?, ?, 'session', ?)",
                (session_id, _now(), reason),
            )
    return existed


def list_deletions() -> list[dict[str, Any]]:
    """Return the deletion log, which holds no personal data by design."""
    with connect() as connection:
        rows = connection.execute("SELECT * FROM deletions ORDER BY id").fetchall()
    return [dict(row) for row in rows]


def _delete_images(folder: Path, keep_debug: bool) -> bool:
    """Remove photos, masks, keypoints and debug overlays; keep the results."""
    removed = False
    for path in folder.glob("*"):
        if path.is_dir():
            if path.name == "debug" and not keep_debug:
                shutil.rmtree(path)
                removed = True
            continue
        if any(path.name.endswith(suffix) for suffix in IMAGE_SUFFIXES):
            path.unlink()
            removed = True
    return removed


def purge_images(today: date | None = None) -> int:
    """Delete photos past their retention date, keeping the measurements.

    ``landmarks.json`` and both YAML files survive, so a purged session is still
    a usable record; only the images of the person are gone.
    """
    today = today or date.today()
    purged = 0
    for session in list_sessions():
        purge_after = session["images_purge_after"]
        if not purge_after or purge_after > today.isoformat():
            continue
        folder = session_folder(session["id"])
        if not folder.exists():
            continue
        if _delete_images(folder, keep_debug=False):
            purged += 1
            with connect() as connection:
                connection.execute(
                    "INSERT INTO deletions (session_id, at, scope, reason)"
                    " VALUES (?, ?, 'images', 'retention')",
                    (session["id"], _now()),
                )
    return purged


def drop_debug(session_id: str) -> None:
    """Remove a confirmed session's overlays when the configuration says to."""
    if load_app_config().storage.keep_debug_after_confirm:
        return
    debug = session_folder(session_id) / "debug"
    if debug.exists():
        shutil.rmtree(debug)


def export_csv(path: Path) -> Path:
    """Write one row per confirmed session for Objective 2 analysis.

    Unconfirmed sessions are left out: they have not been checked by a tailor.
    """
    keys = sorted(load_schema())
    header = ["session_id", "subject_code", "sex", *keys, *[f"{key}_edited" for key in keys]]
    rows = []
    for session in list_sessions():
        if session["status"] not in EXPORTABLE:
            continue
        folder = session_folder(session["id"])
        body_file = folder / "body_measurements.yaml"
        if not body_file.exists():
            continue
        body = yaml.safe_load(body_file.read_text(encoding="utf-8"))["body"]
        report_file = folder / "measurement_report.yaml"
        report = (
            yaml.safe_load(report_file.read_text(encoding="utf-8"))
            if report_file.exists()
            else {"values": {}}
        )
        edited = {key: report["values"].get(key, {}).get("edited", False) for key in keys}
        rows.append(
            [session["id"], session["subject_code"], session["sex"]]
            + [body.get(key, "") for key in keys]
            + [edited[key] for key in keys]
        )

    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return write_atomic(path, buffer.getvalue())


__all__ = [
    "add_view",
    "connect",
    "create_session",
    "db_path",
    "delete_session",
    "export_csv",
    "get_session",
    "initialise",
    "list_deletions",
    "list_edits",
    "list_sessions",
    "log_edit",
    "purge_images",
    "session_folder",
    "set_status",
    "write_atomic",
]