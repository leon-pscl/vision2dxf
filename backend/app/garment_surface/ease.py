"""Ease, and the conversion from a circumference to a surface displacement.

Ease arrives as a circumference: "80 mm at the bust" means the garment section is 80 mm
longer around than the body section. The surface, though, moves normal to itself. The two
are not interchangeable, so the conversion is written down here once.

For a section scaled uniformly by s about its own centre, the perimeter is homogeneous of
degree one, so

    P(s*a, s*b) = s * P(a, b)   =>   s = (P + dP) / P

and a point p on the section moves to s*p, a normal displacement of

    d(t) = (s - 1) * |p(t)|

which varies around the section: it is largest at the side, where the radius is largest,
and smallest at centre front. A circumference ease is therefore *not* a constant normal
offset, and the achieved normal distance is reported per level rather than assumed.

Between two levels the ease is blended with a smoothstep, so the garment surface has no
crease where two ease values meet, and above the topmost eased level it is held constant.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import reconstruction_config


@dataclass(frozen=True)
class Ease:
    bust_mm: float
    waist_mm: float
    hip_mm: float
    armhole_mm: float
    preset: str

    def as_dict(self) -> dict:
        return {
            "preset": self.preset,
            "bustMm": self.bust_mm,
            "waistMm": self.waist_mm,
            "hipMm": self.hip_mm,
            "armholeMm": self.armhole_mm,
        }


def resolve_ease(settings) -> Ease:
    cfg = reconstruction_config()["garment"]
    presets = cfg["ease_presets"]
    name = settings.ease_preset if settings.ease_preset in presets else cfg["default_ease_preset"]
    preset = presets[name]

    def pick(override, key):
        return float(override) if override is not None else float(preset[key])

    return Ease(
        bust_mm=pick(settings.bust_ease_mm, "bust_mm"),
        waist_mm=pick(settings.waist_ease_mm, "waist_mm"),
        hip_mm=pick(settings.hip_ease_mm, "hip_mm"),
        armhole_mm=pick(settings.armhole_ease_mm, "armhole_mm"),
        preset=name,
    )


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def ease_profile(ys: np.ndarray, levels: dict, ease: Ease) -> np.ndarray:
    """Circumference ease in mm at every row of the ring stack."""
    control: list[tuple[float, float]] = [(levels["bust"].y_mm, ease.bust_mm),
                                          (levels["waist"].y_mm, ease.waist_mm)]
    if "hip" in levels:
        control.append((levels["hip"].y_mm, ease.hip_mm))
    control.sort()

    out = np.full(len(ys), control[0][1], dtype=float)
    out[ys >= control[-1][0]] = control[-1][1]
    for (y0, e0), (y1, e1) in zip(control, control[1:]):
        span = y1 - y0
        if span <= 0:
            continue
        mask = (ys > y0) & (ys < y1)
        out[mask] = e0 + (e1 - e0) * _smoothstep((ys[mask] - y0) / span)
    return out


def underarm_drop(ease: Ease) -> float:
    """How far the armhole ease lowers the underarm level."""
    fraction = float(reconstruction_config()["garment"]["armhole_ease_drop_fraction"])
    return fraction * ease.armhole_mm
