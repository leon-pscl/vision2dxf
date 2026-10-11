"""Session folders, state and export (brief 4).

One session is one subject's capture: the photos, what the models saw, the
landmarks, and the two YAML files Objective 1.2 consumes. Everything lives under
``data/sessions/<id>/`` so that deleting that folder deletes the subject's data,
which is what the "Delete session" button does.

The subject's measured height is the session's scale: ``height_cm`` divides into
each view's head-to-floor pixel span to give that view's millimetres per pixel.
There is no floor mark and no station calibration to stand on.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from bodymeasure import store
from bodymeasure.config import load_app_config, load_calibration
from bodymeasure.landmarks import (
    Landmark,
    as_dict,
    find_front_landmarks,
    find_side_landmarks,
    load_landmarks,
    save_landmarks,
    view_scale,
)
from bodymeasure.measure import MeasurementSet, SideView, apply_overrides, measure
from bodymeasure.overlay import landmark_overlay, save, view_overlay
from bodymeasure.posegate import GateResult, faces_right, gate_front, gate_side
from bodymeasure.preprocess import prepare
from bodymeasure.store import session_folder, write_atomic
from bodymeasure.vision import analyse

VIEWS = ("front", "side_left", "side_right")
SIDE_VIEWS = ("side_left", "side_right")
OVERRIDES_FILE = "value_overrides.json"
SUBJECT_CODE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
MASK_MIDPOINT = 127  # a saved mask PNG is 0 or 255
HEIGHT_RANGE_CM = (100.0, 220.0)
HEIGHT_HINT = "Measured by tailor, shoes off. Do not use the client's own estimate."
NO_DISTORTION_WARNING = "lens distortion not corrected"


@dataclass(frozen=True)
class Subject:
    """Who is being measured, their height, and their consent.

    ``code`` is the tailor's client code, never a full name: it is recorded in
    the index, not in any path (brief 10.2). ``height_cm`` is what every
    measurement on every view is scaled by, so it is measured, not estimated.
    """

    code: str
    sex: str
    consent: bool
    height_cm: float
    notes: str = ""


def session_dir(session_id: str) -> Path:
    """Return one session's folder.

    Raises:
        ValueError: if the id is not a UUID hex string.

    """
    return session_folder(session_id)


def create_session(subject: Subject) -> str:
    """Record consent and the measured height, create the folder, return its id.

    Raises:
        ValueError: if consent is not given, the subject code is unusable, or
            the height is outside the range a standing adult can occupy.

    """
    if not subject.consent:
        raise ValueError("consent is required before any capture")
    if not SUBJECT_CODE.match(subject.code):
        raise ValueError("subject code must be letters, digits, dashes or underscores")
    low, high = HEIGHT_RANGE_CM
    if not low <= subject.height_cm <= high:
        raise ValueError(f"height must be {low:.0f}-{high:.0f} cm, measured to 0.1 cm")

    consent_version = load_app_config().storage.consent_version
    session_id = store.create_session(
        subject.code, subject.sex, consent_version, round(subject.height_cm, 1)
    )
    folder = session_dir(session_id)
    folder.mkdir(parents=True, exist_ok=True)
    write_atomic(folder / "subject.yaml", yaml.safe_dump(asdict(subject), sort_keys=False))
    return session_id


def load_subject(session_id: str) -> Subject:
    """Read ``subject.yaml``.

    Raises:
        FileNotFoundError: if the session does not exist.

    """
    path = session_dir(session_id) / "subject.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no such session: {session_id}")
    return Subject(**yaml.safe_load(path.read_text(encoding="utf-8")))


def _mirror(
    image: np.ndarray, mask: np.ndarray, keypoints: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flip a side view so the body faces +x, as brief 5.4 requires."""
    flipped = keypoints.copy()
    flipped[:, 0] = image.shape[1] - 1 - flipped[:, 0]
    return cv2.flip(image, 1), np.fliplr(mask), flipped


def save_view(session_id: str, view: str, image: np.ndarray) -> GateResult:
    """Prepare, segment and gate one captured view, storing what passed.

    A rejected view is still written, so the operator can see the overlay that
    explains the rejection, but it carries no mask or keypoints.

    Raises:
        ValueError: for an unknown view, an uncalibrated station, or a frame
            with no person in it.

    """
    if view not in VIEWS:
        raise ValueError(f"view must be one of {', '.join(VIEWS)}")
    folder = session_dir(session_id)
    if not folder.exists():
        raise FileNotFoundError(f"no such session: {session_id}")

    config = load_app_config()
    prepared = prepare(image, load_calibration())
    analysis = analyse(prepared)
    mask, keypoints = analysis.mask, analysis.keypoints

    if view in SIDE_VIEWS and not faces_right(keypoints):
        prepared, mask, keypoints = _mirror(prepared, mask, keypoints)

    if view == "front":
        result = gate_front(mask, keypoints, config.thresholds)
    else:
        result = gate_side(mask, keypoints, config.thresholds, _front_shoulder_px(folder))

    # A new capture invalidates any landmarks found from the old one.
    (folder / "landmarks.json").unlink(missing_ok=True)
    write_atomic(folder / f"{view}.jpg", cv2.imencode(".jpg", prepared)[1].tobytes())
    write_atomic(
        folder / f"{view}_mask.png",
        cv2.imencode(".png", mask.astype(np.uint8) * 255)[1].tobytes(),
    )
    write_atomic(folder / f"{view}_kp.json", json.dumps(keypoints.tolist()))
    store.add_view(session_id, view)
    save(view_overlay(prepared, mask, keypoints), folder / "debug" / f"{view}_gate.png")
    return result


def _front_shoulder_px(folder: Path) -> float:
    """Return the front view's shoulder span, or 0 when there is no front view yet."""
    path = folder / "front_kp.json"
    if not path.exists():
        return 0.0
    keypoints = np.array(json.loads(path.read_text(encoding="utf-8")))
    return abs(float(keypoints[5, 0] - keypoints[6, 0]))


def _load_view(folder: Path, view: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return the stored image, mask and keypoints of a view, if it was captured."""
    image_path = folder / f"{view}.jpg"
    mask_path = folder / f"{view}_mask.png"
    kp_path = folder / f"{view}_kp.json"
    if not (image_path.exists() and mask_path.exists() and kp_path.exists()):
        return None
    image = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > MASK_MIDPOINT
    keypoints = np.array(json.loads(kp_path.read_text(encoding="utf-8")))
    return image, mask, keypoints


def captured_views(session_id: str) -> list[str]:
    """Return which views the session holds."""
    folder = session_dir(session_id)
    return [view for view in VIEWS if (folder / f"{view}_mask.png").exists()]


def find_landmarks(session_id: str) -> list[Landmark]:
    """Find every landmark on every captured view and write ``landmarks.json``.

    Each view's scale comes from its own floor and head rows, so front and side
    views do not need the same size: the side rows are found from the heights,
    not the row numbers, the front view produced.

    Raises:
        ValueError: if the front view or every side view is missing.

    """
    folder = session_dir(session_id)
    height_cm = load_subject(session_id).height_cm
    front = _load_view(folder, "front")
    if front is None:
        raise ValueError("the front view is required")

    front_image, front_mask, front_kp = front
    marks = find_front_landmarks(front_mask, front_kp, height_cm)
    front_rows = as_dict(marks)
    front_scale = view_scale(marks, height_cm)
    front_heights = {
        name: front_scale.height_mm(front_rows[f"{name}_row"].y)
        for name in ("armpit", "waist", "crotch")
    }
    save(landmark_overlay(front_image, front_mask, marks), folder / "debug" / "front_marks.png")

    sides = [view for view in SIDE_VIEWS if _load_view(folder, view) is not None]
    if not sides:
        raise ValueError("at least one side view is required")
    for view in sides:
        image, mask, keypoints = _load_view(folder, view)
        side_marks = find_side_landmarks(mask, keypoints, view, front_heights, height_cm)
        marks.extend(side_marks)
        save(
            landmark_overlay(image, mask, side_marks), folder / "debug" / f"{view}_marks.png"
        )

    save_landmarks(marks, folder / "landmarks.json", height_cm)
    return marks


def load_overrides(session_id: str) -> dict[str, dict[str, object]]:
    """Return the values the tailor typed in, keyed by measurement name."""
    path = session_dir(session_id) / OVERRIDES_FILE
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_overrides(session_id: str, overrides: dict[str, dict[str, object]]) -> Path:
    """Merge new tailor edits into the stored ones. Returns the path written."""
    path = session_dir(session_id) / OVERRIDES_FILE
    previous = load_overrides(session_id)
    for name, edit in overrides.items():
        store.log_edit(
            session_id,
            "value",
            name,
            previous.get(name, {}).get("value"),
            edit.get("value"),
            str(edit.get("note", "")),
        )
    merged = {**previous, **overrides}
    write_atomic(path, json.dumps(merged, indent=2))
    return path


def _log_landmark_moves(
    session_id: str, before: list[Landmark], after: list[Landmark]
) -> None:
    """Record every landmark the tailor actually moved, in the audit trail."""
    previous = {mark.name + mark.view: mark for mark in before}
    for mark in after:
        old = previous.get(mark.name + mark.view)
        if old is None or (old.x, old.y) == (mark.x, mark.y):
            continue
        store.log_edit(
            session_id,
            "landmark",
            f"{mark.view}.{mark.name}",
            [old.x, old.y],
            [mark.x, mark.y],
        )


def measure_session(session_id: str, landmarks: list[Landmark] | None = None) -> MeasurementSet:
    """Measure a session, optionally with landmarks the tailor edited.

    Raises:
        ValueError: if a view the measurement needs is missing.

    """
    folder = session_dir(session_id)
    path = folder / "landmarks.json"
    if landmarks is None:
        landmarks = load_landmarks(path) if path.exists() else find_landmarks(session_id)
    else:
        _log_landmark_moves(session_id, load_landmarks(path) if path.exists() else [], landmarks)
        save_landmarks(landmarks, path, load_subject(session_id).height_cm)

    front = _load_view(folder, "front")
    if front is None:
        raise ValueError("the front view is required")
    _, front_mask, front_kp = front
    height_cm = load_subject(session_id).height_cm

    sides = []
    for view in SIDE_VIEWS:
        loaded = _load_view(folder, view)
        if loaded is None:
            continue
        marks = {mark.name: mark for mark in landmarks if mark.view == view}
        if not marks:
            continue
        sides.append(SideView(name=view, mask=loaded[1], landmarks=marks, height_cm=height_cm))
    if not sides:
        raise ValueError("at least one side view is required")

    front_marks = [mark for mark in landmarks if mark.view == "front"]
    measurements = measure(front_mask, front_marks, sides, front_kp, height_cm)
    if (store.get_session(session_id) or {}).get("status") == "capturing":
        store.set_status(session_id, "measured")
    return apply_overrides(measurements, load_overrides(session_id))


def export(session_id: str, measurements: MeasurementSet) -> tuple[Path, Path]:
    """Write ``body_measurements.yaml`` and ``measurement_report.yaml``.

    The body file carries only the 26 primary keys: pygarment recomputes the
    underscore keys itself every time it loads the file.
    """
    folder = session_dir(session_id)
    body_path = write_atomic(
        folder / "body_measurements.yaml",
        yaml.safe_dump({"body": measurements.body()}, sort_keys=True),
    )

    subject = load_subject(session_id)
    calibration = load_calibration()
    warnings = list(measurements.warnings)
    if calibration is None or not calibration.has_lens_model:
        warnings.append(NO_DISTORTION_WARNING)
    report = {
        "subject_id": subject.code,
        "side_views_used": [
            view.removeprefix("side_") for view in captured_views(session_id) if view != "front"
        ],
        "values": {
            name: {
                "source": entry.source,
                "confidence": entry.confidence,
                "edited": entry.edited,
                "views": entry.views,
                **({"note": entry.note} if entry.note else {}),
            }
            for name, entry in sorted(measurements.values.items())
        },
        "warnings": warnings,
    }
    report_path = write_atomic(
        folder / "measurement_report.yaml", yaml.safe_dump(report, sort_keys=False)
    )
    return body_path, report_path
