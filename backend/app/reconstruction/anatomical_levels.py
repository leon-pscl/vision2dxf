"""Anatomical levels: the normalised vertical coordinate system.

y = 0 is the neck reference point and +y runs down, which is the frame the measurement
semantics layer already resolves into. A level is never rearranged to make an ordering
work: an inconsistent profile stops the generation and names the fields that disagree.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..assumptions import AssumptionRegistry
from ..config import reconstruction_config


class LevelError(Exception):
    """Inconsistent or implausible anatomical levels; the operator must confirm."""

    def __init__(self, errors: list[dict]) -> None:
        self.errors = errors
        super().__init__("; ".join(e["message"] for e in errors))


@dataclass(frozen=True)
class AnatomicalLevel:
    name: str
    y_mm: float
    source_field: str | None
    origin_definition: str
    confidence: float | None

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "yMm": round(self.y_mm, 4),
            "sourceField": self.source_field,
            "originDefinition": self.origin_definition,
            "confidence": self.confidence,
        }


# The order every profile must satisfy. neck and upper_chest share y = 0: the neck section
# is nested inside the upper-chest section at the neck reference level, see cross_sections.
ORDER = ("neck", "upper_chest", "underarm", "bust", "underbust", "waist", "hip")
AT_ORIGIN = ("neck", "upper_chest")
# The underarm carries no cross-section: it is where the armhole meets the side seam. It
# is checked against the shoulder tip and the waist by the seam network instead of being
# forced above the bust, which would be an anatomical assumption the input does not make.
OUTSIDE_CHAIN = ("underarm",)

NECK_ORIGIN = "neck reference point, the base of the neck at the shoulder line; y = 0"


def build_levels(m, registry: AssumptionRegistry) -> dict[str, AnatomicalLevel]:
    """Levels in millimetres, from the confirmed drafting measurements."""
    cfg = reconstruction_config()["levels"]
    levels: dict[str, AnatomicalLevel] = {
        "neck": AnatomicalLevel("neck", 0.0, "neck_w", NECK_ORIGIN, 1.0),
        "upper_chest": AnatomicalLevel("upper_chest", 0.0, "back_width", NECK_ORIGIN, 0.5),
        "underarm": AnatomicalLevel(
            "underarm", m.armscye_depth_vertical, "armscye_depth",
            "confirmed armscye depth below the neck reference point", 0.8,
        ),
        "bust": AnatomicalLevel(
            "bust", m.neck_to_bust_level, "bust_line",
            "confirmed bust level below the neck reference point", 1.0,
        ),
        "waist": AnatomicalLevel(
            "waist", m.neck_to_waist_level, "waist_line",
            "confirmed waist level below the neck reference point", 1.0,
        ),
    }

    fraction = float(cfg["underbust_fraction_of_bust_to_waist"])
    underbust_y = m.neck_to_bust_level + fraction * m.bust_to_waist_vertical
    levels["underbust"] = AnatomicalLevel(
        "underbust", underbust_y, None,
        f"assumed at {fraction:.0%} of the bust-to-waist distance; the input carries an "
        f"underbust circumference but no underbust level",
        0.3,
    )
    registry.add(
        "level.underbust", "Underbust level", round(underbust_y, 2), "mm",
        f"""The underbust circumference has no level of its own in the input, so the
        section is placed at {fraction:.0%} of the bust-to-waist distance. Nothing
        measures the fraction.""",
        ["bust_line", "waist_line"],
    )

    if cfg["include_hip_section"]:
        drop = float(cfg["waist_to_hip_mm"])
        levels["hip"] = AnatomicalLevel(
            "hip", m.neck_to_waist_level + drop, None,
            "assumed fixed drop below the waist; hips_line is measured from the crotch "
            "point and cannot be placed in the neck-referenced frame",
            0.2,
        )
        registry.add(
            "level.hip", "Hip level", drop, "mm",
            """hips_line declares the crotch point as its origin, so it yields no level in
            the neck-referenced pattern frame. The hip section is placed at a fixed drop
            below the waist instead. Off by default.""",
            ["hips_line", "crotch_hip_diff"],
        )

    validate_levels(levels)
    return levels


def validate_levels(levels: dict[str, AnatomicalLevel]) -> None:
    """Strict ordering and separation. Raises rather than reordering."""
    gap = float(reconstruction_config()["levels"]["minimum_separation_mm"])
    present = [name for name in ORDER if name in levels]
    errors: list[dict] = []

    for name in present:
        level = levels[name]
        if name in AT_ORIGIN:
            continue
        if not math.isfinite(level.y_mm):
            errors.append(_err(name, "LEVEL_NOT_FINITE",
                               f"The {name} level is not a finite number."))
        elif level.y_mm <= 0.0:
            errors.append(_err(
                name, "LEVEL_ABOVE_ORIGIN",
                f"The {name} level sits at {level.y_mm:.1f} mm, on or above the neck "
                f"reference point. Confirm the origin chosen for its source field.",
            ))

    ordered = [name for name in present if name not in AT_ORIGIN + OUTSIDE_CHAIN]
    for a, b in zip(ordered, ordered[1:]):
        ya, yb = levels[a].y_mm, levels[b].y_mm
        if not (math.isfinite(ya) and math.isfinite(yb)):
            continue
        if yb < ya:
            errors.append(_err(
                b, "LEVEL_OUT_OF_ORDER",
                f"The {b} level ({yb:.1f} mm) sits above the {a} level ({ya:.1f} mm). "
                f"Confirm the vertical origins before the surface is built; levels are "
                f"never reordered automatically.",
            ))
        elif yb - ya < gap:
            errors.append(_err(
                b, "LEVELS_TOO_CLOSE",
                f"The {a} and {b} levels are {yb - ya:.1f} mm apart; at least {gap:.0f} mm "
                f"is required for a section of its own.",
            ))
    if errors:
        raise LevelError(errors)


def _err(field: str, code: str, message: str) -> dict:
    return {
        "field": field, "code": code, "message": " ".join(message.split()),
        "suggestedAction": "Confirm the vertical field definitions, or correct the value.",
        "severity": "blocking",
    }
