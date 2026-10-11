"""Find the rows and points the measurements are taken from (brief 5.5).

Definitions follow the GarmentCode measurement sheet. A "torso run" at row ``y``
is the contiguous foreground run containing ``torso_x``, the midpoint of the two
hip keypoints. Rows are stored as landmarks too, with ``x`` on the body centre,
so the review UI can drag a row like any other point.

Each view carries its own :class:`ViewScale`: the subject's measured height
divided by the pixels from the top of the head to the floor turns pixels into
millimetres, with no physical calibration on the station.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from bodymeasure.geometry import gap_containing, run_containing, runs_in_row
from bodymeasure.posegate import (
    L_EAR,
    L_ELBOW,
    L_HIP,
    L_SHOULDER,
    L_WRIST,
    R_EAR,
    R_ELBOW,
    R_HIP,
    R_SHOULDER,
    R_WRIST,
    visible_side,
)

CURVATURE_SPAN = 15  # contour points either side, per brief 5.5
NECK_WIDEN_FRACTION = 0.25  # the neck base is where width exceeds the minimum by 25%
SHOULDER_SEARCH_FRACTION = 0.10  # of shoulder-to-hip distance, below the shoulder keypoint
WAIST_BAND = (0.30, 0.90)  # of the armpit-to-hip span
WRIST_SEARCH_MM = 40.0  # +/- 4 cm around the wrist keypoint
THIGH_DEPTH_MM = 100.0  # thigh rows run 10 cm below the crotch


@dataclass(frozen=True)
class Landmark:
    """One named point, in image pixels, on one view."""

    name: str
    view: str
    x: float
    y: float
    auto: bool = True


@dataclass(frozen=True)
class ViewScale:
    """How many millimetres one pixel is on one view, from the subject's height.

    ``floor_row`` and ``head_row`` are landmarks, so the tailor can drag the head
    row down over hair or a cap and every measurement from that view follows.
    """

    floor_row: int
    head_row: int
    mm_per_px: float

    def height_mm(self, row: float) -> float:
        """Return the height above the floor, in mm, of image row ``row``."""
        return (self.floor_row - row) * self.mm_per_px

    def row_of(self, h_mm: float) -> int:
        """Return the image row standing ``h_mm`` above the floor."""
        return int(round(self.floor_row - h_mm / self.mm_per_px))


def view_scale(landmarks: list[Landmark], height_cm: float) -> ViewScale:
    """Return one view's scale from its floor and head rows and the entered height.

    Raises:
        ValueError: if the scale landmarks are missing, or the body has no
            height in pixels at all.

    """
    rows = as_dict(landmarks)
    missing = {"floor_row", "head_row"} - set(rows)
    if missing:
        raise ValueError(f"missing scale landmarks: {', '.join(sorted(missing))}")
    floor, head = rows["floor_row"].y, rows["head_row"].y
    span = floor - head
    if span <= 0:
        raise ValueError("the floor row must be below the head row: check the mask")
    return ViewScale(int(floor), int(head), height_cm * 10.0 / span)


def as_dict(landmarks: list[Landmark]) -> dict[str, Landmark]:
    """Index landmarks by name."""
    return {landmark.name: landmark for landmark in landmarks}


def scale_landmarks(mask: np.ndarray, view: str, centre_x: float) -> list[Landmark]:
    """Return the floor and head rows that set this view's scale.

    The floor row is the lowest foreground row and the head row the highest, so
    the span between them is the subject as seen in this view, shoes included.

    Raises:
        ValueError: if the mask is empty.

    """
    rows = np.flatnonzero(np.asarray(mask).any(axis=1))
    if rows.size == 0:
        raise ValueError("empty mask: nothing to measure")
    return [
        Landmark("floor_row", view, centre_x, float(rows.max())),
        Landmark("head_row", view, centre_x, float(rows.min())),
    ]


def save_landmarks(landmarks: list[Landmark], path: Path, height_cm: float) -> Path:
    """Write ``landmarks.json``: the points plus each view's resulting scale.

    The points are the single source of truth; the scale beside them is what they
    imply, recomputed here so the two cannot drift apart. Returns the path.
    """
    views = {}
    for view in dict.fromkeys(mark.view for mark in landmarks):
        scale = view_scale([mark for mark in landmarks if mark.view == view], height_cm)
        views[view] = {
            "floor_row": scale.floor_row,
            "head_row": scale.head_row,
            "mm_per_px": round(scale.mm_per_px, 6),
        }
    payload = {
        "views": views,
        "landmarks": [asdict(mark) for mark in landmarks],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_landmarks(path: Path) -> list[Landmark]:
    """Read ``landmarks.json``, including any edits the tailor made."""
    return [
        Landmark(**entry) for entry in json.loads(path.read_text(encoding="utf-8"))["landmarks"]
    ]


def _contour(mask: np.ndarray) -> np.ndarray:
    """Return the largest contour as an ``(N, 2)`` array of pixels.

    Raises:
        ValueError: if the mask is empty.

    """
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("empty mask: nothing to landmark")
    return max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float)


def turning_angles(points: np.ndarray, span: int = CURVATURE_SPAN) -> np.ndarray:
    """Return the turning angle in degrees at every contour point.

    The angle is measured between the chords reaching ``span`` points back and
    ``span`` points forward, so a sharp corner scores high and a smooth arc low.
    """
    before = points - np.roll(points, span, axis=0)
    after = np.roll(points, -span, axis=0) - points
    norms = np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = np.where(norms > 0, (before * after).sum(axis=1) / norms, 1.0)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _sharpest(
    contour: np.ndarray, keep: np.ndarray, span: int = CURVATURE_SPAN
) -> np.ndarray | None:
    """Return the contour point of maximum turning angle among ``keep``."""
    if not keep.any():
        return None
    angles = np.where(keep, turning_angles(contour, span), -1.0)
    return contour[int(np.argmax(angles))]


def neck_base_row(mask: np.ndarray, ear_row: int, shoulder_row: int, torso_x: int) -> int:
    """Return the row where the neck widens into the shoulders.

    The minimum width between the ears and the shoulders is the neck; the base is
    the first row below it that is wider by ``NECK_WIDEN_FRACTION``.

    Raises:
        ValueError: if no neck is found in that band.

    """
    widths = {}
    for row in range(int(ear_row), int(shoulder_row) + 1):
        run = run_containing(mask, row, torso_x)
        if run is not None:
            widths[row] = run[1] - run[0] + 1
    if not widths:
        raise ValueError("no neck found between the ears and the shoulders")

    narrowest = min(widths, key=lambda row: widths[row])
    limit = widths[narrowest] * (1 + NECK_WIDEN_FRACTION)
    for row in range(narrowest, int(shoulder_row) + 1):
        if widths.get(row, 0) > limit:
            return row
    return int(shoulder_row)


def armpit_row(mask: np.ndarray, shoulder_row: int, torso_x: int, limit_row: int) -> int:
    """Return the first row below the shoulders with background on both sides of the torso.

    Raises:
        ValueError: if the arms touch the torso all the way down, which the pose
            gate should already have rejected.

    """
    for row in range(int(shoulder_row), int(limit_row)):
        run = run_containing(mask, row, torso_x)
        if run is None:
            continue
        runs = runs_in_row(mask, row)
        if any(other[1] < run[0] for other in runs) and any(other[0] > run[1] for other in runs):
            return row
    raise ValueError("No gap between the arms and the torso: hold the arms further out")


def crotch_row(mask: np.ndarray, hip_row: int, torso_x: int) -> int:
    """Return the first row below the hips with background between the legs.

    Raises:
        ValueError: if the legs never separate.

    """
    for row in range(int(hip_row), mask.shape[0]):
        if gap_containing(mask, row, torso_x) is not None:
            return row
    raise ValueError("No gap between the legs: stand with the feet apart")


def waist_row(mask: np.ndarray, armpit: int, hip_keypoint_row: int, torso_x: int) -> int:
    """Return the narrowest torso row in the 30-90% band from armpit to hips.

    Raises:
        ValueError: if no torso is found in that band.

    """
    span = int(hip_keypoint_row) - int(armpit)
    low = int(armpit + WAIST_BAND[0] * span)
    high = int(armpit + WAIST_BAND[1] * span)
    widths = {}
    for row in range(low, high + 1):
        run = run_containing(mask, row, torso_x)
        if run is not None:
            widths[row] = run[1] - run[0] + 1
    if not widths:
        raise ValueError("no torso found between the armpits and the hips")
    return min(widths, key=lambda row: widths[row])


def narrowest_cross_section(
    mask: np.ndarray,
    centre: np.ndarray,
    direction: np.ndarray,
    search_px: float,
    half_width_px: float = 120.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the ends of the thinnest cut perpendicular to ``direction``.

    Used for the wrist: the arm is sampled along the elbow-wrist line within
    ``search_px`` of the keypoint, and the thinnest cut across it wins.

    Raises:
        ValueError: if the arm is not found anywhere in the search range.

    """
    unit = direction / (float(np.linalg.norm(direction)) or 1.0)
    normal = np.array([-unit[1], unit[0]])
    offsets = np.arange(-half_width_px, half_width_px + 1)
    middle = len(offsets) // 2
    best: tuple[float, np.ndarray, np.ndarray] | None = None

    for step in np.arange(-search_px, search_px + 1):
        point = centre + unit * step
        samples = point + normal[None, :] * offsets[:, None]
        columns = np.round(samples[:, 0]).astype(int)
        rows = np.round(samples[:, 1]).astype(int)
        inside = (rows >= 0) & (rows < mask.shape[0]) & (columns >= 0) & (columns < mask.shape[1])
        hit = np.zeros(len(offsets), dtype=bool)
        hit[inside] = np.asarray(mask)[rows[inside], columns[inside]] > 0
        run = next(
            (item for item in runs_in_row(hit[None, :], 0) if item[0] <= middle <= item[1]), None
        )
        if run is None:
            continue  # the sample centre has left the arm
        width = run[1] - run[0] + 1
        if best is None or width < best[0]:
            best = (width, point + normal * offsets[run[0]], point + normal * offsets[run[1]])

    if best is None:
        raise ValueError("arm not found near the wrist keypoint")
    return best[1], best[2]


def find_front_landmarks(
    mask: np.ndarray, keypoints: np.ndarray, height_cm: float
) -> list[Landmark]:
    """Find every front-view landmark of brief 5.5.

    Raises:
        ValueError: if the silhouette does not show a landmark the engine needs.

    """
    torso_x = int(round((keypoints[L_HIP, 0] + keypoints[R_HIP, 0]) / 2))
    shoulder_row = int(round((keypoints[L_SHOULDER, 1] + keypoints[R_SHOULDER, 1]) / 2))
    hip_row = int(round((keypoints[L_HIP, 1] + keypoints[R_HIP, 1]) / 2))
    ear_row = int(round((keypoints[L_EAR, 1] + keypoints[R_EAR, 1]) / 2))

    scale = scale_landmarks(mask, "front", torso_x)
    neck_row = neck_base_row(mask, ear_row, shoulder_row, torso_x)
    neck_run = run_containing(mask, neck_row, torso_x)
    if neck_run is None:
        raise ValueError("no neck found at the neck-base row")
    armpit = armpit_row(mask, shoulder_row, torso_x, hip_row)
    waist = waist_row(mask, armpit, hip_row, torso_x)
    crotch = crotch_row(mask, hip_row, torso_x)

    # COCO numbers sides by the subject's own body, so someone facing the camera
    # has their left on the image right. Landmarks follow the same convention:
    # otherwise measure.py pairs a shoulder with the opposite arm.
    left_outward = 1.0 if keypoints[L_SHOULDER, 0] > torso_x else -1.0
    outward = {"left": left_outward, "right": -left_outward}
    neck_x = {
        side: float(neck_run[1] if outward[side] > 0 else neck_run[0])
        for side in ("left", "right")
    }

    marks = [
        *scale,
        Landmark("neck_base_left", "front", neck_x["left"], neck_row),
        Landmark("neck_base_right", "front", neck_x["right"], neck_row),
        Landmark("armpit_row", "front", torso_x, armpit),
        Landmark("waist_row", "front", torso_x, waist),
        Landmark("crotch_row", "front", torso_x, crotch),
    ]

    contour = _contour(mask)
    search_depth = SHOULDER_SEARCH_FRACTION * (hip_row - shoulder_row)
    in_band = (contour[:, 1] >= neck_row) & (contour[:, 1] <= shoulder_row + search_depth)
    for side in ("left", "right"):
        # "Outer contour" means lateral to the neck: the neck-shoulder junction is
        # itself a sharp corner and would otherwise win the curvature search.
        lateral = (contour[:, 0] - torso_x) * outward[side] > abs(neck_x[side] - torso_x)
        point = _sharpest(contour, in_band & lateral)
        if point is None:
            raise ValueError(f"could not find the {side} shoulder point")
        marks.append(Landmark(f"shoulder_{side}", "front", float(point[0]), float(point[1])))

    search_px = WRIST_SEARCH_MM / view_scale(scale, height_cm).mm_per_px
    for name, elbow_index, wrist_index in (
        ("wrist_left", L_ELBOW, L_WRIST),
        ("wrist_right", R_ELBOW, R_WRIST),
    ):
        elbow = keypoints[elbow_index, :2]
        wrist = keypoints[wrist_index, :2]
        start, end = narrowest_cross_section(mask, wrist, wrist - elbow, search_px)
        marks.append(Landmark(f"{name}_a", "front", float(start[0]), float(start[1])))
        marks.append(Landmark(f"{name}_b", "front", float(end[0]), float(end[1])))

    return marks


def balance_x_at(row: float, top: Landmark, bottom: Landmark) -> float:
    """Return the x of the balance line at ``row``, extrapolating beyond its ends."""
    if bottom.y == top.y:
        return top.x
    ratio = (row - top.y) / (bottom.y - top.y)
    return top.x + ratio * (bottom.x - top.x)


def find_side_landmarks(
    mask: np.ndarray,
    keypoints: np.ndarray,
    view: str,
    front_heights: Mapping[str, float],
    height_cm: float,
) -> list[Landmark]:
    """Find every side-view landmark of brief 5.5.

    ``front_heights`` carries the armpit, waist and crotch heights in mm found on
    the front view. Heights, not rows, because each view has its own scale: the
    same height lands on a different row in a differently sized frame.

    Raises:
        ValueError: if the silhouette does not show a landmark the engine needs.

    """
    side = visible_side(keypoints)
    shoulder_index = L_SHOULDER if side == "left" else R_SHOULDER
    hip_index = L_HIP if side == "left" else R_HIP
    ear_index = L_EAR if side == "left" else R_EAR

    top = Landmark(
        "balance_top",
        view,
        float(keypoints[shoulder_index, 0]),
        float(keypoints[shoulder_index, 1]),
    )
    bottom = Landmark(
        "balance_bottom", view, float(keypoints[hip_index, 0]), float(keypoints[hip_index, 1])
    )
    floor, head = scale_landmarks(mask, view, 0.0)
    scale = view_scale([floor, head], height_cm)
    marks = [
        top,
        bottom,
        Landmark("floor_row", view, balance_x_at(floor.y, top, bottom), floor.y),
        Landmark("head_row", view, balance_x_at(head.y, top, bottom), head.y),
    ]

    def depths(row: int) -> tuple[float, float]:
        """Return the (front, back) depth in pixels at ``row``."""
        centre = int(round(balance_x_at(row, top, bottom)))
        run = run_containing(mask, row, centre)
        if run is None:
            return 0.0, 0.0
        return float(run[1] - centre + 1), float(centre - run[0])

    armpit = scale.row_of(front_heights["armpit"])
    waist = scale.row_of(front_heights["waist"])
    crotch = scale.row_of(front_heights["crotch"])
    bust = max(range(armpit, waist + 1), key=lambda row: depths(row)[0])
    underbust = min(range(bust, waist + 1), key=lambda row: depths(row)[0])
    hip = max(range(waist, crotch + 1), key=lambda row: depths(row)[1])

    for name, row in (("bust_row", bust), ("underbust_row", underbust), ("hip_row", hip)):
        marks.append(Landmark(name, view, balance_x_at(row, top, bottom), row))

    contour = _contour(mask)
    ear_row = float(keypoints[ear_index, 1])
    shoulder_row = float(keypoints[shoulder_index, 1])
    centre_x = balance_x_at((ear_row + shoulder_row) / 2, top, bottom)
    keep = (
        (contour[:, 1] >= ear_row)
        & (contour[:, 1] <= shoulder_row)
        & (contour[:, 0] < centre_x)  # the back, since the body faces +x
    )
    nape = _sharpest(contour, keep)
    if nape is None:
        raise ValueError("could not find the nape point")
    marks.append(Landmark("nape", view, float(nape[0]), float(nape[1])))

    # The brief puts the HPS where the top of the shoulder crosses the balance
    # line, but that line runs through the neck and head in a side view, so it
    # never leaves the body there. The neck-shoulder junction is the same point,
    # and it is found by the rule the front view already uses for the neck base.
    hps_row = neck_base_row(mask, int(ear_row), int(shoulder_row), int(round(centre_x)))
    marks.append(Landmark("hps", view, balance_x_at(hps_row, top, bottom), hps_row))

    return marks
