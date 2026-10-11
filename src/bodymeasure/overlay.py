"""Debug and review images: what the engine saw, drawn on the photo."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from bodymeasure.posegate import KEYPOINT_NAMES

OUTLINE = (0, 220, 0)
KEYPOINT = (0, 160, 255)
FLOOR = (255, 80, 80)
LANDMARK = (255, 200, 0)
EDITED = (255, 0, 255)  # a landmark the tailor moved


def draw_mask_outline(
    image: np.ndarray, mask: np.ndarray, colour: tuple[int, int, int] = OUTLINE
) -> np.ndarray:
    """Return a copy of ``image`` with the mask boundary traced on it."""
    canvas = image.copy()
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(canvas, contours, -1, colour, 2)
    return canvas


def draw_keypoints(
    image: np.ndarray,
    keypoints: np.ndarray,
    min_conf: float = 0.5,
    labels: bool = False,
) -> np.ndarray:
    """Return a copy of ``image`` with the keypoints marked.

    Points below ``min_conf`` are drawn hollow, so a weak detection is visible
    rather than silently absent.
    """
    canvas = image.copy()
    for index, (x, y, confidence) in enumerate(np.asarray(keypoints, dtype=float)):
        centre = (int(round(x)), int(round(y)))
        cv2.circle(canvas, centre, 7, KEYPOINT, -1 if confidence >= min_conf else 2)
        if labels:
            cv2.putText(
                canvas,
                KEYPOINT_NAMES[index],
                (centre[0] + 10, centre[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                KEYPOINT,
                1,
                cv2.LINE_AA,
            )
    return canvas


def draw_row(
    image: np.ndarray, row: int, label: str = "", colour: tuple[int, int, int] = FLOOR
) -> np.ndarray:
    """Return a copy of ``image`` with a labelled horizontal line at ``row``."""
    canvas = image.copy()
    cv2.line(canvas, (0, row), (canvas.shape[1], row), colour, 2)
    if label:
        cv2.putText(
            canvas, label, (8, max(row - 8, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 1,
            cv2.LINE_AA,
        )
    return canvas


def view_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    keypoints: np.ndarray,
    floor_row: int | None = None,
    min_conf: float = 0.5,
) -> np.ndarray:
    """Return the standard per-view debug image: outline, keypoints, floor line."""
    canvas = draw_mask_outline(image, mask)
    canvas = draw_keypoints(canvas, keypoints, min_conf)
    if floor_row is not None:
        canvas = draw_row(canvas, floor_row, "floor")
    return canvas


def save(image: np.ndarray, path: Path) -> Path:
    """Write ``image`` as a PNG, creating the folder. Returns the path written.

    Raises:
        OSError: if the file cannot be written.

    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"could not write {path}")
    return path


def landmark_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    landmarks: list,
    rows_as_lines: bool = True,
) -> np.ndarray:
    """Return a debug image showing every landmark, named.

    Landmarks whose name ends in ``_row`` are drawn as full-width lines, because
    that is what they mean: a height at which the body is measured.
    """
    canvas = draw_mask_outline(image, mask)
    for mark in landmarks:
        point = (int(round(mark.x)), int(round(mark.y)))
        if rows_as_lines and mark.name.endswith("_row"):
            canvas = draw_row(canvas, point[1], mark.name, LANDMARK)
            continue
        colour = LANDMARK if mark.auto else EDITED
        cv2.drawMarker(canvas, point, colour, cv2.MARKER_CROSS, 16, 2)
        cv2.putText(
            canvas,
            mark.name,
            (point[0] + 10, point[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            colour,
            1,
            cv2.LINE_AA,
        )
    return canvas
