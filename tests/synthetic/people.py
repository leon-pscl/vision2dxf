"""A drawn silhouette with matching COCO-17 keypoints.

Enough of a body to exercise the pose gate and the landmark search: head, neck,
torso, two arms at a settable angle, and two legs with a real gap between them.
Everything is drawn at 1 mm/px so a pixel is a millimetre.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

FRAME_MARGIN = 0.09  # the pose gate wants 8% of the frame clear top and bottom
_DRAWN_FLOOR = 1750  # rows below, in the drawing as originally laid out
_DRAWN_HEAD_TOP = 105
FRAME_HEIGHT = round((_DRAWN_FLOOR - _DRAWN_HEAD_TOP + 1) / (1 - 2 * FRAME_MARGIN))
FRAME_PAD = round(FRAME_MARGIN * FRAME_HEIGHT) - _DRAWN_HEAD_TOP

# Every row below is a row in the finished frame, not in the drawing: the figure is
# framed with FRAME_MARGIN clear above and below, so a good capture really does
# pass the gate rather than failing it for want of room.
SHAPE = (FRAME_HEIGHT, 900)  # rows, columns
FLOOR_ROW = _DRAWN_FLOOR + FRAME_PAD
CENTRE_X = 450

HEAD_CENTRE = (CENTRE_X, 220 + FRAME_PAD)
HEAD_AXES = (80, 115)
SHOULDER_Y = 380 + FRAME_PAD
SHOULDER_DX = 95
WAIST_Y = 760 + FRAME_PAD
WAIST_DX = 62
HIP_Y = 880 + FRAME_PAD
HIP_DX = 78
KNEE_Y = 1320 + FRAME_PAD
ANKLE_Y = 1700 + FRAME_PAD
CROTCH_Y = 1000 + FRAME_PAD  # the legs meet until here, so the crotch is below the hip
LEG_DX = 85
ARM_UPPER_MM = 250.0
ARM_LOWER_MM = 230.0
ARM_THICKNESS = 55
HAND_MM = 90.0  # the hand continues past the wrist, so the wrist is a real narrowing
NECK_TOP_DX = 45
NECK_BOTTOM_DX = 75
NECK_TOP_Y = 295 + FRAME_PAD
LEG_GAP = 18


@dataclass(frozen=True)
class SyntheticPerson:
    """A silhouette, its keypoints, and the rows the landmark search should find."""

    mask: np.ndarray
    keypoints: np.ndarray
    floor_row: int
    torso_x: int
    shoulder_span_px: float
    rows: dict[str, int]


def _keypoints(arm_angle_deg: float, side_on: bool, wrist_over_torso: bool) -> np.ndarray:
    """Build COCO-17 keypoints consistent with what :func:`make_person` draws."""
    angle = math.radians(arm_angle_deg)
    shoulder_dx = 10 if side_on else SHOULDER_DX
    points = np.zeros((17, 3), dtype=float)
    points[:, 2] = 0.9

    points[0] = (CENTRE_X, HEAD_CENTRE[1] + 20, 0.9)  # nose
    points[1] = (CENTRE_X - 30, HEAD_CENTRE[1], 0.9)
    points[2] = (CENTRE_X + 30, HEAD_CENTRE[1], 0.9)
    points[3] = (CENTRE_X - 70, HEAD_CENTRE[1] + 10, 0.9)
    points[4] = (CENTRE_X + 70, HEAD_CENTRE[1] + 10, 0.9)

    for shoulder, elbow, wrist, sign in ((5, 7, 9, -1), (6, 8, 10, 1)):
        shoulder_x = CENTRE_X + sign * shoulder_dx
        points[shoulder] = (shoulder_x, SHOULDER_Y, 0.9)
        elbow_x = shoulder_x + sign * ARM_UPPER_MM * math.sin(angle)
        elbow_y = SHOULDER_Y + ARM_UPPER_MM * math.cos(angle)
        points[elbow] = (elbow_x, elbow_y, 0.9)
        if wrist_over_torso:
            points[wrist] = (CENTRE_X, WAIST_Y, 0.9)
        else:
            points[wrist] = (
                elbow_x + sign * ARM_LOWER_MM * math.sin(angle),
                elbow_y + ARM_LOWER_MM * math.cos(angle),
                0.9,
            )

    points[11] = (CENTRE_X - HIP_DX, HIP_Y, 0.9)
    points[12] = (CENTRE_X + HIP_DX, HIP_Y, 0.9)
    points[13] = (CENTRE_X - LEG_DX, KNEE_Y, 0.9)
    points[14] = (CENTRE_X + LEG_DX, KNEE_Y, 0.9)
    points[15] = (CENTRE_X - LEG_DX, ANKLE_Y, 0.9)
    points[16] = (CENTRE_X + LEG_DX, ANKLE_Y, 0.9)

    if side_on:
        # The far side of the body is hidden, so its keypoints are unreliable.
        for index in (2, 4, 6, 8, 10, 12, 14, 16):
            points[index, 2] = 0.15
    return points


def make_person(
    arm_angle_deg: float = 30.0,
    side_on: bool = False,
    wrist_over_torso: bool = False,
    floor_offset_px: int = 0,
) -> SyntheticPerson:
    """Draw a silhouette and return it with its keypoints and known rows.

    ``floor_offset_px`` shifts the whole body down, to test the floor-mark check.
    """
    mask = np.zeros(SHAPE, dtype=np.uint8)
    points = _keypoints(arm_angle_deg, side_on, wrist_over_torso)

    cv2.ellipse(mask, HEAD_CENTRE, HEAD_AXES, 0, 0, 360, 1, -1)
    # The neck flares into the shoulders rather than meeting them in a step, so
    # the "width exceeds the minimum by 25%" rule has a row to land on.
    neck = np.array(
        [
            [CENTRE_X - NECK_TOP_DX, NECK_TOP_Y],
            [CENTRE_X + NECK_TOP_DX, NECK_TOP_Y],
            [CENTRE_X + NECK_BOTTOM_DX, SHOULDER_Y],
            [CENTRE_X - NECK_BOTTOM_DX, SHOULDER_Y],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, [neck], 1)
    torso = np.array(
        [
            [CENTRE_X - SHOULDER_DX, SHOULDER_Y],
            [CENTRE_X + SHOULDER_DX, SHOULDER_Y],
            [CENTRE_X + WAIST_DX, WAIST_Y],
            [CENTRE_X + HIP_DX, HIP_Y],
            [CENTRE_X - HIP_DX, HIP_Y],
            [CENTRE_X - WAIST_DX, WAIST_Y],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, [torso], 1)

    for shoulder, elbow, wrist in ((5, 7, 9), (6, 8, 10)):
        forearm = points[wrist, :2] - points[elbow, :2]
        hand = points[wrist, :2] + forearm / (np.linalg.norm(forearm) or 1.0) * HAND_MM
        chain = np.vstack([points[[shoulder, elbow, wrist], :2], hand]).round().astype(np.int32)
        cv2.polylines(mask, [chain], False, 1, ARM_THICKNESS)
        cv2.circle(mask, tuple(chain[-1]), ARM_THICKNESS // 2, 1, -1)

    for sign in (-1, 1):
        # The leg starts exactly as wide as the hip and widens towards the knee, so
        # the silhouette has no step at the hip row and the hip width is stable.
        outer = CENTRE_X + sign * HIP_DX
        # The inner edges meet at the centre down to CROTCH_Y and step apart there,
        # rather than parting gradually: a gap that opens by a fraction of a pixel
        # per row would put the crotch wherever rounding lands. make_side draws the
        # same body, so the crotch row found here is the one needed there.
        leg = np.array(
            [
                [CENTRE_X, HIP_Y - 1],
                [outer, HIP_Y - 1],
                [CENTRE_X + sign * (LEG_DX + 25), FLOOR_ROW],
                [CENTRE_X + sign * LEG_GAP, FLOOR_ROW],
                [CENTRE_X + sign * LEG_GAP, CROTCH_Y + 1],
                [CENTRE_X, CROTCH_Y],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [leg], 1)

    if floor_offset_px:
        mask = np.roll(mask, floor_offset_px, axis=0)
        points[:, 1] += floor_offset_px

    return SyntheticPerson(
        mask=mask.astype(bool),
        keypoints=points,
        floor_row=FLOOR_ROW + floor_offset_px,
        torso_x=CENTRE_X,
        shoulder_span_px=2.0 * SHOULDER_DX,
        rows={
            "shoulder": SHOULDER_Y + floor_offset_px,
            "waist": WAIST_Y + floor_offset_px,
            "hip": HIP_Y + floor_offset_px,
            "knee": KNEE_Y + floor_offset_px,
            "floor": FLOOR_ROW + floor_offset_px,
        },
    )


# Side profile: (row, back offset, front offset) from the balance line, in px.
# The maxima and minima are the answers the landmark search must find. Rows are in
# frame coordinates like every other row here, so the profile lines up with the
# front view's armpit, waist and crotch.
_DRAWN_SIDE_PROFILE = (
    (105, 55, 70),     # head top
    (200, 85, 95),     # head
    (300, 45, 55),     # neck
    (380, 85, 90),     # shoulder
    (520, 95, 150),    # bust: deepest front
    (600, 92, 100),    # underbust: shallowest front below the bust
    (760, 88, 118),    # waist
    (880, 150, 90),    # hip: deepest back
    (1000, 110, 95),   # crotch
    (1320, 70, 75),    # knee
    (1700, 60, 120),   # ankle and foot
    (1750, 60, 130),   # floor
)
SIDE_PROFILE = tuple((row + FRAME_PAD, back, front) for row, back, front in _DRAWN_SIDE_PROFILE)
SIDE_BALANCE_X = 450
_DRAWN_SIDE_ROWS = {"bust": 520, "underbust": 600, "waist": 760, "hip": 880, "crotch": 1000}
SIDE_ROWS = {name: row + FRAME_PAD for name, row in _DRAWN_SIDE_ROWS.items()}


@dataclass(frozen=True)
class SyntheticSide:
    """A side silhouette facing +x, its keypoints, and the rows that define it."""

    mask: np.ndarray
    keypoints: np.ndarray
    floor_row: int
    balance_x: int
    rows: dict[str, int]
    depths: dict[str, tuple[float, float]]


@dataclass(frozen=True)
class RescaledFigure:
    """A synthetic figure redrawn at another size, with the height it implies.

    The engine's scale is the subject's height over the head-to-floor pixel span,
    so a figure drawn at any size measures the same as long as the reported
    ``height_cm`` matches what was drawn.
    """

    mask: np.ndarray
    keypoints: np.ndarray
    mm_per_px: float

    @property
    def height_cm(self) -> float:
        """Return the height the drawn figure implies, to the 0.1 cm the app takes."""
        rows = np.flatnonzero(self.mask.any(axis=1))
        return round(float(rows.max() - rows.min()) * self.mm_per_px / 10.0, 1)


def rescale(
    mask: np.ndarray,
    keypoints: np.ndarray,
    factor: float,
    offset_px: tuple[int, int] = (0, 0),
) -> RescaledFigure:
    """Return a figure redrawn ``factor`` times larger in pixels, then shifted.

    ``offset_px`` is ``(columns, rows)``, the image x/y order. Area averaging
    keeps the drawn widths honest at a fraction of a pixel, which is what a
    different camera distance actually does to a silhouette.
    """
    height = max(1, int(round(mask.shape[0] * factor)))
    width = max(1, int(round(mask.shape[1] * factor)))
    drawn = (cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_AREA)
             > 0.5)
    # The figure lands exactly on the offset and the canvas grows to hold it, so a
    # negative offset shifts the body up rather than being dropped. The origin is
    # subtracted because mask and keypoints both index the canvas, not the frame.
    column, row = offset_px
    origin_row, origin_column = min(0, row), min(0, column)
    shift_row, shift_column = row - origin_row, column - origin_column
    canvas = np.zeros(
        (
            max(mask.shape[0], row + height) - origin_row,
            max(mask.shape[1], column + width) - origin_column,
        ),
        dtype=bool,
    )
    canvas[
        shift_row : shift_row + height,
        shift_column : shift_column + width,
    ] = drawn
    points = keypoints.copy()
    # cv2.resize maps pixel centres, so a source pixel lands at
    # (pixel + 0.5) * factor - 0.5 in the destination.
    points[:, 0] = (points[:, 0] + 0.5) * factor - 0.5 + shift_column
    points[:, 1] = (points[:, 1] + 0.5) * factor - 0.5 + shift_row
    return RescaledFigure(mask=canvas, keypoints=points, mm_per_px=1.0 / factor)


def _side_edges() -> tuple[np.ndarray, np.ndarray]:
    """Interpolate the back and front edges for every row of the figure."""
    rows = np.array([entry[0] for entry in SIDE_PROFILE], dtype=float)
    back = np.array([entry[1] for entry in SIDE_PROFILE], dtype=float)
    front = np.array([entry[2] for entry in SIDE_PROFILE], dtype=float)
    every_row = np.arange(SHAPE[0], dtype=float)
    return (
        SIDE_BALANCE_X - np.interp(every_row, rows, back, left=0.0, right=back[-1]),
        SIDE_BALANCE_X + np.interp(every_row, rows, front, left=0.0, right=front[-1]),
    )


def make_side(view: str = "side_left") -> SyntheticSide:
    """Draw a side silhouette whose bust, underbust and hip rows are known."""
    back, front = _side_edges()
    mask = np.zeros(SHAPE, dtype=bool)
    for row in range(SIDE_PROFILE[0][0], FLOOR_ROW + 1):
        mask[row, int(round(back[row])) : int(round(front[row])) + 1] = True

    points = np.zeros((17, 3), dtype=float)
    points[:, 2] = 0.15  # the far side of the body is hidden
    for index, (x, y) in (
        (0, (SIDE_BALANCE_X + 90, 220)),  # nose, in front of the ear
        (3, (SIDE_BALANCE_X + 20, 215)),  # left ear
        (5, (SIDE_BALANCE_X, SHOULDER_Y)),
        (11, (SIDE_BALANCE_X, HIP_Y)),
        (13, (SIDE_BALANCE_X + 5, KNEE_Y)),
        (15, (SIDE_BALANCE_X, ANKLE_Y)),
    ):
        points[index] = (x, y, 0.9)

    depths = {
        name: (
            front[row] - SIDE_BALANCE_X + 1,
            SIDE_BALANCE_X - back[row],
        )
        for name, row in SIDE_ROWS.items()
    }
    return SyntheticSide(
        mask=mask,
        keypoints=points,
        floor_row=FLOOR_ROW,
        balance_x=SIDE_BALANCE_X,
        rows=dict(SIDE_ROWS),
        depths=depths,
    )
