"""A0 placement and nesting of flattened pattern panels.

Pattern geometry is never altered by placement. This module is responsible only for
choosing a rotation and (x, y) origin for each panel so that panels fit on paper
with the configured margin and do not overlap. It also reports the consumed area and
efficiency so placement can be evaluated against the objective.

Rotations are limited to multiples of 90 degrees, which preserves axis-aligned bounding
boxes and keeps cutters from having to rotate pieces on the sheet.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .config import drafting_config
from .drafting.geometry import PatternDocument, PatternPanel


A0_MM = (1189.0, 841.0)


@dataclass
class PlacementRect:
    panel: PatternPanel
    rotation_deg: float
    x: float
    y: float
    width: float
    height: float


@dataclass
class PlacementReport:
    paper_width: float
    paper_height: float
    margin: float
    rects: list[PlacementRect] = field(default_factory=list)
    overflow: list[str] = field(default_factory=list)

    @property
    def panel_count(self) -> int:
        return len(self.rects)

    @property
    def used_area(self) -> float:
        return float(sum(r.width * r.height for r in self.rects))

    @property
    def paper_area(self) -> float:
        return self.paper_width * self.paper_height

    @property
    def usable_area(self) -> float:
        return (self.paper_width - 2 * self.margin) * (self.paper_height - 2 * self.margin)

    @property
    def efficiency(self) -> float:
        return self.used_area / self.paper_area if self.paper_area else 0.0


def _paper_size(name: str, orientation: str | None) -> tuple[float, float]:
    cfg = drafting_config()
    if name not in cfg["paper"]["sizes_mm"]:
        raise ValueError(f"unknown paper size: {name}")
    w, h = cfg["paper"]["sizes_mm"][name]
    if orientation == "landscape":
        return (max(w, h), min(w, h))
    if orientation == "portrait":
        return (min(w, h), max(w, h))
    return (float(w), float(h))


def place_document(
    doc: PatternDocument,
    paper_size: str = "A0",
    orientation: str | None = None,
    margin_mm: float | None = None,
    rotations: tuple[float, ...] = (0.0, 90.0),
) -> PlacementReport:
    """Bottom-left-ish shelf placement, evaluating every allowed rotation."""
    cfg = drafting_config()["paper"]
    paper_w, paper_h = _paper_size(paper_size, orientation)
    margin = float(margin_mm if margin_mm is not None else cfg["margin_mm"])

    items = []
    for panel in doc.panels:
        x0, y0, x1, y1 = panel.bbox()
        w = x1 - x0
        h = y1 - y0
        rotations_to_try = []
        for rot in rotations:
            if rot % 180 == 0:
                rotations_to_try.append((rot, (w, h)))
            else:
                rotations_to_try.append((rot, (h, w)))
        items.append((panel, rotations_to_try))

    # widest-first improves shelf packing in a cheap, deterministic way.
    items.sort(key=lambda it: -max(r[1][0] * r[1][1] for r in it[1]))

    bins: list[tuple[float, float, float, float]] = []  # x, y, w, h
    rects: list[PlacementRect] = []
    overflow: list[str] = []
    shelf_y = margin
    shelf_height = 0.0
    cursor_x = margin
    available_w = paper_w - 2 * margin

    for panel, options in items:
        chosen_rot, (w, h) = min(options, key=lambda item: max(item[1][0], item[1][1]))
        if cursor_x + w + margin > paper_w - margin and shelf_height > 0:
            shelf_y += shelf_height + margin
            cursor_x = margin
            shelf_height = 0.0

        x0, y0, x1, y1 = panel.bbox()
        x = cursor_x
        y = shelf_y
        if x + w + margin > paper_w - margin or y + h + margin > paper_h - margin:
            overflow.append(panel.key or panel.label or "<panel>")
            continue

        # translate from panel bbox origin to the chosen origin.
        origin_x = x - x0
        origin_y = y - y0
        placed = PlacementRect(panel=panel, rotation_deg=chosen_rot,
                               x=origin_x, y=origin_y, width=w, height=h)
        rects.append(placed)
        bins.append((x, y, w, h))
        cursor_x += w + margin
        shelf_height = max(shelf_height, h)

    report = PlacementReport(
        paper_width=paper_w, paper_height=paper_h, margin=margin,
        rects=rects, overflow=overflow,
    )
    return report


def validate_placement(report: PlacementReport, tolerance: float = 1e-6) -> list[str]:
    """Return a list of overlap/out-of-bounds problems."""
    errors: list[str] = []
    for i, a in enumerate(report.rects):
        for j, b in enumerate(report.rects):
            if j <= i:
                continue
            ax, ay, aw, ah = a.x, a.y, a.width, a.height
            bx, by, bw, bh = b.x, b.y, b.width, b.height
            if ax < bx + bw and ax + aw > bx and ay < by + bh and ay + ah > by:
                errors.append(f"overlap: {a.panel.key} vs {b.panel.key}")
        if a.x < 0 or a.y < 0 or a.x + a.width - report.paper_width > tolerance:
            errors.append(f"out of bounds: {a.panel.key}")
        if a.y + a.height - report.paper_height > tolerance:
            errors.append(f"out of bounds: {a.panel.key}")
    if report.overflow:
        errors.append("panels did not fit: " + ", ".join(report.overflow))
    return errors
