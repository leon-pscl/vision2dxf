"""Mask clean-up (brief 5.3).

Opening then closing removes speckle and seals pinholes, the largest connected
component drops anything that is not the body, and only *small* holes are
filled: the gaps between arm and torso and between the legs are large and carry
the measurements, so they must stay background.
"""

from __future__ import annotations

import cv2
import numpy as np


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Return only the largest foreground component, as uint8 0/1."""
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return binary
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest).astype(np.uint8)


def fill_small_holes(mask: np.ndarray, max_area_px: int) -> np.ndarray:
    """Fill enclosed background regions smaller than ``max_area_px``.

    Regions touching the frame edge are never filled: they are outside the body,
    not holes in it.
    """
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    height, width = binary.shape
    count, labels, stats, _ = cv2.connectedComponentsWithStats(1 - binary, connectivity=4)
    filled = binary.copy()
    for label in range(1, count):
        x, y, w, h, area = (int(value) for value in stats[label, :5])
        touches_edge = x == 0 or y == 0 or x + w == width or y + h == height
        if area < max_area_px and not touches_edge:
            filled[labels == label] = 1
    return filled


def clean_mask(
    mask: np.ndarray, kernel_px: int = 5, hole_fill_max_area_px: int = 500
) -> np.ndarray:
    """Run the full clean-up and return a boolean mask."""
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_px, kernel_px))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = keep_largest_component(binary)
    return fill_small_holes(binary, hole_fill_max_area_px).astype(bool)
