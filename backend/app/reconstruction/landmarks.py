"""Landmarks on the reconstructed surface.

A landmark is a (column, level) pair on the ring grid, so every landmark is a real
surface point and every seam that ends on one ends on a mesh path. Columns are indexed
from centre front: column 0 is centre front, M/4 the right side, M/2 centre back.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .perimeter_solver import point_at


@dataclass(frozen=True)
class Landmark3D:
    name: str
    column: int
    y_mm: float
    position_mm: tuple[float, float, float]
    source_field: str | None
    note: str

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "column": self.column,
            "yMm": round(self.y_mm, 4),
            "positionMm": [round(v, 4) for v in self.position_mm],
            "sourceField": self.source_field,
            "note": " ".join(self.note.split()),
        }


def column_angle(column: int, columns: int) -> float:
    return 2.0 * math.pi * column / columns


def key_columns(columns: int) -> dict[str, int]:
    if columns % 4:
        raise ValueError("the ring resolution must be a multiple of four")
    return {
        "center_front": 0,
        "right_side": columns // 4,
        "center_back": columns // 2,
        "left_side": 3 * columns // 4,
    }


def column_for_half_width(
    a: float, b: float, n: float, columns: int, target_x: float, quarter: str = "front"
) -> int:
    """The column whose x is closest to `target_x`, searching one quarter of the ring.

    Returns an index in [0, M/4] for the front quarter and [M/4, M/2] for the back one,
    which is what the seam network needs: a landmark that lands on a column, never
    between two.
    """
    lo, hi = (0, columns // 4) if quarter == "front" else (columns // 4, columns // 2)
    best, best_err = lo, float("inf")
    for j in range(lo, hi + 1):
        x = point_at(a, b, n, column_angle(j, columns))[0]
        err = abs(abs(x) - target_x)
        if err < best_err:
            best, best_err = j, err
    return best


def ring_x(a: float, b: float, n: float, columns: int) -> np.ndarray:
    """x of every column on one ring, for the landmark search and the diagnostics."""
    t = np.array([column_angle(j, columns) for j in range(columns)])
    p = 2.0 / n
    return a * np.sign(np.sin(t)) * np.abs(np.sin(t)) ** p


def build_landmarks(sections: dict, m, columns: int) -> dict[str, Landmark3D]:
    """The landmarks the seam network and the pattern labels refer to."""
    keys = key_columns(columns)
    out: dict[str, Landmark3D] = {}

    def place(name: str, section, column: int, source: str | None, note: str) -> None:
        x, z = point_at(section.a_mm, section.b_mm, section.exponent,
                        column_angle(column, columns))
        out[name] = Landmark3D(name, column, section.y_mm, (x, section.y_mm, z), source, note)

    chest = sections["upper_chest"]
    bust = sections["bust"]
    waist = sections["waist"]

    place("center_front_neck", chest, keys["center_front"], None,
          "centre front on the upper-chest section, before the neckline is cut")
    place("center_back_neck", chest, keys["center_back"], None,
          "centre back on the upper-chest section, before the neckline is cut")
    place("center_front_waist", waist, keys["center_front"], "waist_line",
          "centre front at the waist level")
    place("center_back_waist", waist, keys["center_back"], "waist_line",
          "centre back at the waist level")
    place("side_waist", waist, keys["right_side"], "waist_line",
          "right side seam at the waist level")

    apex_column = column_for_half_width(
        bust.a_mm, bust.b_mm, bust.exponent, columns, m.bust_apex_spacing_full / 2.0,
    )
    place("bust_apex", bust, apex_column, "bust_points",
          "half the confirmed apex spacing, snapped to the nearest ring column")
    return out
