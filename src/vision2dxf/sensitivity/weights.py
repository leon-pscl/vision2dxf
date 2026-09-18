from __future__ import annotations
import itertools


CRITERIA_ORDER = [
    "economic",
    "usability",
    "performance",
    "reliability",
    "maintainability",
]

LOI_VALUES = [10, 9, 8, 7, 6]


def generate_loi_permutations() -> list[tuple[int, ...]]:
    return list(itertools.permutations(LOI_VALUES))


def loi_to_weights(loi: tuple[int, ...]) -> dict[str, float]:
    total = sum(loi)
    return {c: v / total for c, v in zip(CRITERIA_ORDER, loi)}
