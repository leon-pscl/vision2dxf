"""Generate front and side silhouettes whose widths and depths are known exactly.

The body is modelled as a stack of elliptical cross-sections: at a given height
the front view shows the ellipse width and the side view shows its front and
back depth about the balance line. Rasterising row by row keeps the drawn sizes
exact, so a test can assert that the geometry recovers them to within a pixel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Height above the floor (mm) -> (width, front depth, back depth) in mm.
# Loosely human, but the numbers only have to be known, not realistic.
SLICES: dict[str, tuple[float, float, float]] = {
    "bust": (300.0, 130.0, 110.0),
    "underbust": (270.0, 100.0, 105.0),
    "waist": (250.0, 95.0, 100.0),
    "hips": (330.0, 105.0, 145.0),
}
HEIGHTS_MM: dict[str, float] = {
    "bust": 1250.0,
    "underbust": 1150.0,
    "waist": 1050.0,
    "hips": 900.0,
}


@dataclass(frozen=True)
class SyntheticFigure:
    """A front and side silhouette pair with the sizes that were drawn into it."""

    front: np.ndarray
    side: np.ndarray
    mm_per_px_x: float
    mm_per_px_y: float
    floor_row: int
    torso_x: int
    balance_x: int

    def row_of(self, name: str) -> int:
        """Return the image row holding the named slice."""
        return int(round(self.floor_row - HEIGHTS_MM[name] / self.mm_per_px_y))


def make_figure(mm_per_px: float = 1.04, shape: tuple[int, int] = (1600, 900)) -> SyntheticFigure:
    """Draw the figure of :data:`SLICES` at ``mm_per_px`` and return it with its truth."""
    height, width = shape
    floor_row = height - 40
    torso_x = width // 2
    balance_x = width // 2
    front = np.zeros(shape, dtype=bool)
    side = np.zeros(shape, dtype=bool)

    for name, (w_mm, d_front_mm, d_back_mm) in SLICES.items():
        row = int(round(floor_row - HEIGHTS_MM[name] / mm_per_px))
        half = int(round(w_mm / mm_per_px / 2))
        front[row, torso_x - half : torso_x + half] = True
        # split_depth_at counts the balance column itself as front, so the front
        # run is one pixel shorter than the requested depth.
        front_px = int(round(d_front_mm / mm_per_px)) - 1
        back_px = int(round(d_back_mm / mm_per_px))
        side[row, balance_x - back_px : balance_x + front_px + 1] = True

    return SyntheticFigure(
        front=front,
        side=side,
        mm_per_px_x=mm_per_px,
        mm_per_px_y=mm_per_px,
        floor_row=floor_row,
        torso_x=torso_x,
        balance_x=balance_x,
    )
