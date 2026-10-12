"""The per-view stage images and the manifest that indexes them (brief 7, 8).

Every image the pipeline produces is written once, under its own stage name, and
listed in ``manifest.json`` with the algorithm and parameters that made it and
the earlier files it came from. Nothing is overwritten by a later stage: the raw
capture in particular is kept exactly as it arrived, because a measurement the
operator cannot trace back to a photograph is not reviewable.

The web app reads :data:`STAGES` through the API, so the filmstrip and this
module cannot drift apart.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from bodymeasure.store import session_folder, write_atomic

MANIFEST = "manifest.json"
GREYSCALE_DIMS = 2  # a mask or a contour arrives without a colour channel


@dataclass(frozen=True)
class Stage:
    """One stage of the visual pipeline: its id, its label and its file suffix."""

    stage_id: str
    label: str
    description: str
    suffix: str = ".png"


# The order is the order the operator steps through in the UI (brief 8).
STAGES: tuple[Stage, ...] = (
    Stage("00_raw", "Raw capture", "Exactly what the camera or the upload gave us.", ".jpg"),
    Stage("01_oriented", "Orientation", "Rotated upright; a side view mirrored to face +x."),
    Stage("02_undistorted", "Lens correction", "Undistorted, or marked not applied."),
    Stage("03_enhanced", "Enhancement", "The image the vision models actually saw."),
    Stage("04_detection", "Human detection", "Box, class and confidence of the chosen person."),
    Stage("05_roi", "Extracted ROI", "The person region, cropped from the enhanced image."),
    Stage("06_mask_raw", "Raw mask", "Segmentation output before any cleanup."),
    Stage("07_mask_clean", "Cleaned mask", "After opening, closing and hole treatment."),
    Stage("08_body_isolated", "Separated body", "Foreground body on a neutral background."),
    Stage("09_edges", "Contour", "The silhouette contour the geometry functions use."),
    Stage("10_keypoints", "Pose keypoints", "Named COCO-17 keypoints with confidence."),
    Stage("11_landmarks", "Landmarks", "Anatomical points and measurement rows."),
    Stage("12_measurement_overlay", "Measurements", "Breadths, depths, paths and values."),
)
STAGE_IDS: tuple[str, ...] = tuple(stage.stage_id for stage in STAGES)
_BY_ID = {stage.stage_id: stage for stage in STAGES}


def stage(stage_id: str) -> Stage:
    """Return one stage's definition.

    Raises:
        KeyError: if ``stage_id`` is not a stage of the pipeline.

    """
    return _BY_ID[stage_id]


def view_dir(session_id: str, view: str) -> Path:
    """Return the folder holding one view's stage images."""
    return session_folder(session_id) / "views" / view


def stage_path(session_id: str, view: str, stage_id: str) -> Path:
    """Return where one stage image of one view lives.

    Raises:
        KeyError: if ``stage_id`` is not a stage of the pipeline.

    """
    return view_dir(session_id, view) / f"{stage_id}{stage(stage_id).suffix}"


def _manifest_path(session_id: str) -> Path:
    return session_folder(session_id) / MANIFEST


def manifest(session_id: str) -> list[dict[str, object]]:
    """Return every manifest entry of a session, in pipeline order per view."""
    path = _manifest_path(session_id)
    if not path.exists():
        return []
    entries: list[dict[str, object]] = json.loads(path.read_text(encoding="utf-8"))
    return sorted(entries, key=lambda e: (str(e["view"]), str(e["stage_id"])))


def record(
    session_id: str,
    view: str,
    stage_id: str,
    image: np.ndarray,
    algorithm: str,
    parameters: dict[str, object] | None = None,
    source_files: list[str] | None = None,
    runtime_ms: float | None = None,
    note: str = "",
) -> Path:
    """Write one stage image and its manifest entry. Returns the path written.

    Re-capturing a view replaces that view's entry for the stage rather than
    appending a second one, so the manifest always describes the files on disk.

    Raises:
        KeyError: if ``stage_id`` is not a stage of the pipeline.
        OSError: if the image cannot be encoded or written.

    """
    path = stage_path(session_id, view, stage_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(path.suffix, _as_bgr(image))
    if not ok:
        raise OSError(f"could not encode {path}")
    write_atomic(path, buffer.tobytes())

    entry = {
        "stage_id": stage_id,
        "view": view,
        "path": path.relative_to(session_folder(session_id)).as_posix(),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "algorithm": algorithm,
        "parameters": parameters or {},
        "source_files": source_files or [],
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "label": stage(stage_id).label,
        "description": stage(stage_id).description,
    }
    if runtime_ms is not None:
        entry["runtime_ms"] = round(float(runtime_ms), 1)
    if note:
        entry["note"] = note

    kept = [e for e in manifest(session_id) if (e["view"], e["stage_id"]) != (view, stage_id)]
    write_atomic(_manifest_path(session_id), json.dumps([*kept, entry], indent=2))
    return path


def drop_view(session_id: str, view: str) -> None:
    """Forget one view's stages, so a retake cannot leave stale entries behind."""
    path = _manifest_path(session_id)
    if not path.exists():
        return
    folder = session_folder(session_id)
    kept = []
    for entry in manifest(session_id):
        if entry["view"] == view:
            (folder / str(entry["path"])).unlink(missing_ok=True)
            continue
        kept.append(entry)
    write_atomic(path, json.dumps(kept, indent=2))


def drop_stages(session_id: str, stage_ids: tuple[str, ...]) -> None:
    """Forget the named stages across every view, and delete their files.

    Landmarks are found for all views together, so losing them invalidates every
    view's landmark and measurement overlay at once: a stale overlay on a view
    that was not retaken would show the operator rows that no longer exist.
    """
    folder = session_folder(session_id)
    kept = []
    for entry in manifest(session_id):
        if entry["stage_id"] in stage_ids:
            (folder / str(entry["path"])).unlink(missing_ok=True)
            continue
        kept.append(entry)
    if folder.exists():
        write_atomic(_manifest_path(session_id), json.dumps(kept, indent=2))


def _as_bgr(image: np.ndarray) -> np.ndarray:
    """Return any stage image as a writable 8-bit BGR or BGRA frame.

    Masks arrive boolean and contours single-channel; both are saved as pictures
    because the operator has to be able to look at them.
    """
    array = np.asarray(image)
    if array.dtype == bool:
        array = array.astype(np.uint8) * 255
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == GREYSCALE_DIMS:
        return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    return array
