"""Stitch (seam) relationships between panels.

The reference fixture stores a seam as a pair of records; one runs forward along its
edge and the other runs backward, with a matching isCounterClockWise flag. We keep
that exact shape so the output stays fixture-compatible.
"""
from __future__ import annotations

from .geometry import PatternDocument, PatternPanel, StitchPair, StitchRecord

# Semantic seams, expressed as (front role, back role, label).
SEAMS = (
    ("side_seam", "side_seam", "Side seam"),
    ("shoulder", "shoulder", "Shoulder seam"),
)


def build_stitches(doc: PatternDocument, panels: list[PatternPanel]) -> list[StitchPair]:
    front = next(p for p in panels if p.key == "front")
    back = next(p for p in panels if p.key == "back")
    pairs: list[StitchPair] = []
    for front_role, back_role, label in SEAMS:
        fe = front.edge_by_role(front_role)
        be = back.edge_by_role(back_role)
        if fe is None or be is None:
            continue
        fe.paired_edge_id = be.ensure_id()
        be.paired_edge_id = fe.ensure_id()
        pairs.append(
            StitchPair(
                records=[
                    StitchRecord(front.ensure_id(), fe.ensure_id(), 0.0, 1.0, False),
                    StitchRecord(back.ensure_id(), be.ensure_id(), 1.0, 0.0, True),
                ]
            )
        )
    return pairs
