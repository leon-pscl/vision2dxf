"""Superellipse perimeter and the inverse problems the reconstruction needs.

A section is

    abs(x / a) ** n + abs(z / b) ** n = 1

sampled as x = a * sgn(sin t) * abs(sin t) ** (2/n), z = b * sgn(cos t) * abs(cos t) ** (2/n),
so t = 0 is centre front, t = pi/2 the right side and t = pi centre back.

The perimeter is always computed numerically, by chord summation. No closed-form
approximation is used anywhere, so there is never an unreported approximation error; the
only error is the discretisation, which is reported with every solved section.
"""
from __future__ import annotations

import math

import numpy as np

from ..config import reconstruction_config


class SectionError(Exception):
    """A width and circumference that no section can satisfy."""

    def __init__(self, errors: list[dict]) -> None:
        self.errors = errors
        super().__init__("; ".join(e["message"] for e in errors))


def _samples() -> int:
    return int(reconstruction_config()["cross_section"]["perimeter_samples"])


def ring(a: float, b: float, n: float, count: int, offset: float = 0.0) -> np.ndarray:
    """`count` points on the section, uniform in t, as an (count, 2) array of (x, z)."""
    t = offset + np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
    p = 2.0 / n
    x = a * np.sign(np.sin(t)) * np.abs(np.sin(t)) ** p
    z = b * np.sign(np.cos(t)) * np.abs(np.cos(t)) ** p
    return np.column_stack([x, z])


def point_at(a: float, b: float, n: float, t: float) -> tuple[float, float]:
    p = 2.0 / n
    return (
        a * math.copysign(abs(math.sin(t)) ** p, math.sin(t)),
        b * math.copysign(abs(math.cos(t)) ** p, math.cos(t)),
    )


def perimeter(a: float, b: float, n: float, samples: int | None = None) -> float:
    """Numerical perimeter by chord summation."""
    pts = ring(a, b, n, samples or _samples())
    closed = np.vstack([pts, pts[:1]])
    return float(np.sum(np.linalg.norm(np.diff(closed, axis=0), axis=1)))


def unit_perimeter(ratio: float, n: float) -> float:
    """Perimeter of the section with a = 1 and b = ratio.

    The perimeter is homogeneous of degree one in (a, b), so a section of a fixed
    depth-to-width ratio scales linearly: this is what makes solve_scale exact.
    """
    return perimeter(1.0, ratio, n)


def solve_scale(target_perimeter: float, ratio: float, n: float) -> tuple[float, float]:
    """Half width and half depth for a target circumference at a fixed b/a ratio.

    Exact, because perimeter(k*a, k*b) == k * perimeter(a, b).
    """
    if target_perimeter <= 0.0 or ratio <= 0.0:
        raise SectionError([_err("circumference", "SECTION_NOT_POSITIVE",
                                 "A section needs a positive circumference and ratio.")])
    a = target_perimeter / unit_perimeter(ratio, n)
    return a, a * ratio


def solve_depth(target_perimeter: float, half_width: float, n: float) -> tuple[float, int]:
    """Half depth for a known half width and target circumference.

    Returns (half_depth, iterations). The perimeter rises monotonically with the depth and
    tends to 4 * half_width as the depth vanishes, so a circumference at or below that is
    geometrically impossible and is rejected rather than clamped.
    """
    cfg = reconstruction_config()["cross_section"]["solver"]
    if half_width <= 0.0:
        raise SectionError([_err("width", "SECTION_WIDTH_NOT_POSITIVE",
                                 "A section needs a positive width.")])
    floor = 4.0 * half_width
    if target_perimeter <= floor:
        raise SectionError([_err(
            "width", "SECTION_IMPOSSIBLE",
            f"A section {2 * half_width:.1f} mm wide has a perimeter of at least "
            f"{floor:.1f} mm, but the target circumference is {target_perimeter:.1f} mm. "
            f"The width and the circumference cannot both be right.",
        )])

    lo, hi = 1e-9, max(half_width, 1.0)
    for _ in range(64):
        if perimeter(half_width, hi, n) >= target_perimeter:
            break
        hi *= 2.0
        if hi > 1e6:
            raise SectionError([_err(
                "circumference", "SECTION_UNSOLVABLE",
                f"No depth reaches a perimeter of {target_perimeter:.1f} mm for a width of "
                f"{2 * half_width:.1f} mm.",
            )])

    tol = float(cfg["tolerance_mm"])
    iterations = 0
    for iterations in range(1, int(cfg["max_iterations"]) + 1):
        mid = 0.5 * (lo + hi)
        value = perimeter(half_width, mid, n)
        if abs(value - target_perimeter) <= tol:
            return mid, iterations
        if value < target_perimeter:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), iterations


def _err(field: str, code: str, message: str) -> dict:
    return {
        "field": field, "code": code, "message": " ".join(message.split()),
        "suggestedAction": "Correct the width or the circumference, or change the section model.",
        "severity": "blocking",
    }
