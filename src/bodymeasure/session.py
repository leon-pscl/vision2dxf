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
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import cv2
import numpy as np
import yaml

from bodymeasure import artifacts, store
from bodymeasure.config import load_app_config, load_calibration
from bodymeasure.export import Gate, write_all
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
from bodymeasure.overlay import (
    contour_image,
    crop_roi,
    draw_detection,
    draw_keypoints,
    isolate_body,
    landmark_overlay,
    measurement_overlay,
    save,
    view_overlay,
)
from bodymeasure.posegate import GateResult, faces_right, gate_front, gate_side
from bodymeasure.preprocess import NOT_APPLIED, Prepared, prepare_stages
from bodymeasure.store import session_folder, write_atomic
from bodymeasure.vision import ViewAnalysis, analyse

VIEWS = ("front", "side_left", "side_right")
SIDE_VIEWS = ("side_left", "side_right")
OVERRIDES_FILE = "value_overrides.json"
PROFILES = ("core_upper_body", "top_shirt", "skirt", "trousers", "full")
SCALE_MODES = ("stature",)
# What each measurement row on the overlay feeds, so a number in the table can
# be traced to the span drawn on the picture (brief 22).
ROW_LABELS = {
    "armpit_row": "chest breadth",
    "waist_row": "waist breadth",
    "crotch_row": "crotch breadth",
    "bust_row": "bust depth",
    "underbust_row": "underbust depth",
    "hip_row": "hip depth",
}
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
    profile: str = "full"
    scale_mode: str = "stature"


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
    if subject.profile not in PROFILES:
        raise ValueError(f"garment profile must be one of {', '.join(PROFILES)}")
    if subject.scale_mode not in SCALE_MODES:
        raise ValueError(f"scale mode must be one of {', '.join(SCALE_MODES)}")

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


def _record_preprocessing(
    session_id: str, view: str, prepared: Prepared, mirrored: bool
) -> None:
    """Write stages 00 to 03: what the photograph was, and what we did to it."""
    artifacts.record(
        session_id, view, "00_raw", prepared.raw, "camera_or_upload",
        {"note": "never overwritten"},
    )
    artifacts.record(
        session_id, view, "01_oriented", prepared.oriented, "rotate",
        {"rotation_deg": prepared.rotation_deg, "mirrored_to_face_plus_x": mirrored},
        [f"views/{view}/00_raw.jpg"],
    )
    artifacts.record(
        session_id, view, "02_undistorted", prepared.undistorted, "cv2.undistort",
        {"applied": prepared.lens_corrected} if prepared.lens_corrected
        else {"applied": False, "status": NOT_APPLIED},
        [f"views/{view}/01_oriented.png"],
    )
    artifacts.record(
        session_id, view, "03_enhanced", prepared.enhanced, "enhance",
        {"pad_frac": prepared.pad_frac, **prepared.enhance_parameters},
        [f"views/{view}/02_undistorted.png"],
        note="the image the vision models were given",
    )


def _record_vision(
    session_id: str, view: str, image: np.ndarray, analysis: ViewAnalysis
) -> None:
    """Write stages 04 to 10: the detection, the masks, the body and the pose."""
    config = load_app_config()
    source = [f"views/{view}/03_enhanced.png"]
    artifacts.record(
        session_id, view, "04_detection",
        draw_detection(image, analysis.box, analysis.confidence, analysis.instances),
        "yolo26_seg",
        {
            "class": "person",
            "confidence": round(analysis.confidence, 3),
            "instances_over_threshold": analysis.instances,
            "box_xyxy": list(analysis.box),
            "imgsz": config.models.imgsz,
            "runtime_ms": round(analysis.seg_ms, 1),
        },
        source,
        runtime_ms=analysis.seg_ms,
    )
    artifacts.record(
        session_id, view, "05_roi", crop_roi(image, analysis.box), "crop_to_detection",
        {"box_xyxy_in_enhanced": list(analysis.box), "margin_px": 8}, source,
    )
    raw_mask = analysis.raw_mask if analysis.raw_mask is not None else analysis.mask
    artifacts.record(
        session_id, view, "06_mask_raw", raw_mask, "yolo26_seg",
        {"retina_masks": config.models.retina_masks}, source,
    )
    artifacts.record(
        session_id, view, "07_mask_clean", analysis.mask, "clean_mask",
        {
            "kernel_px": int(config.thresholds["morph_kernel_px"]),
            "max_hole_area_px": int(config.thresholds["hole_fill_max_area_px"]),
        },
        [f"views/{view}/06_mask_raw.png"],
    )
    artifacts.record(
        session_id, view, "08_body_isolated", isolate_body(image, analysis.mask),
        "isolate_body", {"background": "neutral grey"},
        [*source, f"views/{view}/07_mask_clean.png"],
    )
    artifacts.record(
        session_id, view, "09_edges", contour_image(analysis.mask), "findContours",
        {"mode": "RETR_LIST", "approximation": "CHAIN_APPROX_NONE"},
        [f"views/{view}/07_mask_clean.png"],
    )
    artifacts.record(
        session_id, view, "10_keypoints",
        draw_keypoints(image, analysis.keypoints, config.thresholds["keypoint_conf"], True),
        "yolo26_pose",
        {
            "keypoint_conf": config.thresholds["keypoint_conf"],
            "runtime_ms": round(analysis.pose_ms, 1),
        },
        source,
        runtime_ms=analysis.pose_ms,
    )


def save_view(
    session_id: str, view: str, image: np.ndarray, upright: bool = False
) -> GateResult:
    """Prepare, segment and gate one captured view, keeping every stage.

    A rejected view is still written, so the operator can see the stages that
    explain the rejection; it simply carries no accepted mask, and the gate
    result says what to do about it. ``upright`` marks a frame that is already
    the right way up, as any upload or stored sample is.

    Raises:
        ValueError: for an unknown view, or a frame with no person in it.
        FileNotFoundError: if the session does not exist.

    """
    if view not in VIEWS:
        raise ValueError(f"view must be one of {', '.join(VIEWS)}")
    folder = session_dir(session_id)
    if not folder.exists():
        raise FileNotFoundError(f"no such session: {session_id}")

    config = load_app_config()
    # A retake replaces this view entirely: no stale stage may survive it.
    artifacts.drop_view(session_id, view)
    prepared = prepare_stages(image, load_calibration(), upright)
    analysis = analyse(prepared.enhanced)
    image_out, mask, keypoints = prepared.enhanced, analysis.mask, analysis.keypoints

    mirrored = view in SIDE_VIEWS and not faces_right(keypoints)
    if mirrored:
        image_out, mask, keypoints = _mirror(image_out, mask, keypoints)
        analysis = replace(
            analysis, mask=mask, keypoints=keypoints, raw_mask=np.fliplr(analysis.raw_mask)
            if analysis.raw_mask is not None else None,
        )
        prepared = replace(prepared, enhanced=image_out)

    if view == "front":
        result = gate_front(mask, keypoints, config.thresholds)
    else:
        result = gate_side(mask, keypoints, config.thresholds, _front_shoulder_px(session_id))

    # A new capture invalidates any landmarks found from the old one, on every
    # view: they are found together, so none of them survives alone.
    (folder / "landmarks.json").unlink(missing_ok=True)
    artifacts.drop_stages(session_id, ("11_landmarks", "12_measurement_overlay"))
    _record_preprocessing(session_id, view, prepared, mirrored)
    _record_vision(session_id, view, image_out, analysis)
    predictions = folder / "predictions"
    predictions.mkdir(parents=True, exist_ok=True)
    write_atomic(
        predictions / f"{view}_detections.json",
        json.dumps(
            {
                "model": config.models.seg,
                "class": "person",
                "confidence": analysis.confidence,
                "box_xyxy": list(analysis.box),
                "instances_over_threshold": analysis.instances,
                "mirrored_to_face_plus_x": mirrored,
                "seg_ms": round(analysis.seg_ms, 1),
            },
            indent=2,
        ),
    )
    write_atomic(
        predictions / f"{view}_keypoints.json",
        json.dumps({"model": config.models.pose, "coco17": keypoints.tolist()}, indent=2),
    )
    write_atomic(
        predictions / f"{view}_gate.json", json.dumps(result.as_dict(), indent=2)
    )
    store.add_view(session_id, view)
    save(view_overlay(image_out, mask, keypoints), folder / "debug" / f"{view}_gate.png")
    return result


def gate_of(session_id: str, view: str) -> dict[str, object]:
    """Return the stored gate result of a view, or an explicit "not captured".

    Raises:
        ValueError: for an unknown view.

    """
    if view not in VIEWS:
        raise ValueError(f"view must be one of {', '.join(VIEWS)}")
    path = session_dir(session_id) / "predictions" / f"{view}_gate.json"
    if not path.exists():
        return {
            "ok": False,
            "captured": False,
            "reasons": [{"code": "NOT_CAPTURED", "message": "This view has not been captured."}],
        }
    return {**json.loads(path.read_text(encoding="utf-8")), "captured": True}


def _front_shoulder_px(session_id: str) -> float:
    """Return the front view's shoulder span, or 0 when there is no front view yet."""
    keypoints = _load_keypoints(session_id, "front")
    if keypoints is None:
        return 0.0
    return abs(float(keypoints[5, 0] - keypoints[6, 0]))


def _load_keypoints(session_id: str, view: str) -> np.ndarray | None:
    """Return one view's stored COCO-17 keypoints, or ``None`` if it has none."""
    path = session_dir(session_id) / "predictions" / f"{view}_keypoints.json"
    if not path.exists():
        return None
    return np.array(json.loads(path.read_text(encoding="utf-8"))["coco17"], dtype=float)


def _load_view(session_id: str, view: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return a view's enhanced image, cleaned mask and keypoints, if it was captured.

    These are the stage files themselves (brief 7), not copies of them: the
    picture the operator reviews is the picture the measurement came from.
    """
    image_path = artifacts.stage_path(session_id, view, "03_enhanced")
    mask_path = artifacts.stage_path(session_id, view, "07_mask_clean")
    keypoints = _load_keypoints(session_id, view)
    if keypoints is None or not (image_path.exists() and mask_path.exists()):
        return None
    image = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > MASK_MIDPOINT
    return image, mask, keypoints


def captured_views(session_id: str) -> list[str]:
    """Return which views the session holds."""
    return [
        view
        for view in VIEWS
        if artifacts.stage_path(session_id, view, "07_mask_clean").exists()
    ]


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
    front = _load_view(session_id, "front")
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
    _record_landmarks(session_id, "front", front_image, front_mask, marks)

    sides = [view for view in SIDE_VIEWS if _load_view(session_id, view) is not None]
    if not sides:
        raise ValueError("at least one side view is required")
    for view in sides:
        image, mask, keypoints = _load_view(session_id, view)
        side_marks = find_side_landmarks(mask, keypoints, view, front_heights, height_cm)
        marks.extend(side_marks)
        _record_landmarks(session_id, view, image, mask, side_marks)

    save_landmarks(marks, folder / "landmarks.json", height_cm)
    return marks


def _record_landmarks(
    session_id: str, view: str, image: np.ndarray, mask: np.ndarray, marks: list[Landmark]
) -> None:
    """Write stage 11: every landmark and measurement row, named, on the photograph."""
    artifacts.record(
        session_id, view, "11_landmarks", landmark_overlay(image, mask, marks),
        "find_landmarks",
        {
            "points": len(marks),
            "edited": sum(1 for mark in marks if not mark.auto),
            "rows": [mark.name for mark in marks if mark.name.endswith("_row")],
        },
        [f"views/{view}/07_mask_clean.png", f"views/{view}/10_keypoints.png"],
    )
    save(landmark_overlay(image, mask, marks), session_dir(session_id) / "debug"
         / f"{view}_marks.png")


def _record_measurements(
    session_id: str, landmarks: list[Landmark], measurements: MeasurementSet, height_cm: float
) -> None:
    """Write stage 12 for every captured view: the spans that were measured."""
    for view in captured_views(session_id):
        loaded = _load_view(session_id, view)
        marks = [mark for mark in landmarks if mark.view == view]
        if loaded is None or not marks:
            continue
        image, mask, _ = loaded
        scale = view_scale(marks, height_cm)
        rows = as_dict(marks)
        reference = rows["waist_row"].x if "waist_row" in rows else mask.shape[1] / 2
        artifacts.record(
            session_id, view, "12_measurement_overlay",
            measurement_overlay(image, mask, marks, ROW_LABELS, reference, scale.mm_per_px),
            "measurement_overlay",
            {
                "mm_per_px": round(scale.mm_per_px, 6),
                "floor_row": scale.floor_row,
                "head_row": scale.head_row,
                "formulae": sorted(
                    {entry.formula for entry in measurements.values.values() if entry.formula}
                ),
            },
            [f"views/{view}/11_landmarks.png"],
        )


def process(session_id: str) -> MeasurementSet:
    """Run everything after capture: landmarks, measurements, overlays, exports.

    Vision already ran once per capture, so this is the cheap half of the
    pipeline and is safe to call again whenever the operator wants the numbers
    recomputed. Stored landmark edits survive, because only a capture clears
    them.

    Raises:
        FileNotFoundError: if the session does not exist.
        ValueError: if a view the measurement needs is missing.

    """
    measurements = measure_session(session_id)
    export(session_id, measurements)
    return measurements


def process_status(session_id: str) -> dict[str, object]:
    """Return which stage each view has reached, and what is still missing.

    Built from the manifest rather than from a job record, so it describes the
    files that exist rather than what some earlier run intended to write
    (brief 16.2).
    """
    done: dict[str, list[str]] = {}
    for entry in artifacts.manifest(session_id):
        done.setdefault(str(entry["view"]), []).append(str(entry["stage_id"]))
    captured = captured_views(session_id)
    return {
        "views": {
            view: {
                "stages_done": sorted(done.get(view, [])),
                "stages_missing": [
                    stage_id
                    for stage_id in artifacts.STAGE_IDS
                    if stage_id not in done.get(view, [])
                ],
                "gate": gate_of(session_id, view),
            }
            for view in VIEWS
            if view in done or view in captured
        },
        "measured": (session_dir(session_id) / "landmarks.json").exists(),
        "status": (store.get_session(session_id) or {}).get("status"),
    }


def reset_landmarks(session_id: str) -> list[Landmark]:
    """Throw away every operator edit and find the landmarks again (brief 11.4).

    Raises:
        ValueError: if a view the search needs is missing.

    """
    (session_dir(session_id) / "landmarks.json").unlink(missing_ok=True)
    (session_dir(session_id) / OVERRIDES_FILE).unlink(missing_ok=True)
    artifacts.drop_stages(session_id, ("11_landmarks", "12_measurement_overlay"))
    return find_landmarks(session_id)


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

    front = _load_view(session_id, "front")
    if front is None:
        raise ValueError("the front view is required")
    _, front_mask, front_kp = front
    height_cm = load_subject(session_id).height_cm

    sides = []
    for view in SIDE_VIEWS:
        loaded = _load_view(session_id, view)
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
    measurements = apply_overrides(measurements, load_overrides(session_id))
    _record_measurements(session_id, landmarks, measurements, height_cm)
    if (store.get_session(session_id) or {}).get("status") == "capturing":
        store.set_status(session_id, "measured")
    return measurements


def export(session_id: str, measurements: MeasurementSet) -> Gate:
    """Write the report, the extended file and, if the gate allows it, the body file.

    The report is always written, including for a partial or failed session; the
    pattern-ready file is written only when every field the chosen garment
    profile needs was actually acquired (brief 15.3). Returns that gate, so the
    review page can say exactly what is blocking an export.

    Raises:
        FileNotFoundError: if the session does not exist.

    """
    subject = load_subject(session_id)
    return write_all(
        session_dir(session_id),
        session_id,
        measurements,
        subject.profile,
        load_calibration(),
    )
