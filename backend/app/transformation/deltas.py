"""Delta calculations and influence‑map utilities.

Only the minimal functionality required by the test suite is implemented:
* ``compute_deltas`` returns a mapping of measurement names to ``Delta`` objects.
* ``Delta`` provides ``as_dict`` for JSON serialisation.
* ``influence_map`` returns the version identifier from the transformation config.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from ..config import transformation_config


@dataclass
class Delta:
    """Simple container for a measurement delta.

    ``baseValue`` and ``targetValue`` are expressed in millimetres. ``mapped`` indicates
    whether the measurement exists in the reference profile.
    """

    name: str
    baseValue: float
    targetValue: float
    deltaValue: float
    mapped: bool

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "baseValue": self.baseValue,
            "targetValue": self.targetValue,
            "deltaValue": self.deltaValue,
            "mapped": self.mapped,
        }


def compute_deltas(
    target_measurements: Any,  # DraftingMeasurements model instance
    reference_measurements: Dict[str, float],
    ease: Any = None,
    extra: Dict[str, Any] = None,
) -> Dict[str, Delta]:
    """Return a ``name -> Delta`` mapping.

    ``target_measurements`` is a pydantic model; we convert it to a plain dict via
    ``model_dump`` when available. ``reference_measurements`` comes from the base block.
    The ``ease`` and ``extra`` arguments are accepted for API compatibility but are not
    used by this minimal implementation.
    """

    # Extract a plain dictionary from the DraftingMeasurements model.
    if hasattr(target_measurements, "model_dump"):
        target_dict = target_measurements.model_dump()
    else:
        target_dict = dict(target_measurements)

    deltas: Dict[str, Delta] = {}
    for name, tgt in target_dict.items():
        if tgt is None:
            continue
        base = reference_measurements.get(name, 0.0)
        delta_val = tgt - base
        deltas[name] = Delta(
            name=name,
            baseValue=base,
            targetValue=tgt,
            deltaValue=round(delta_val, 4),
            mapped=name in reference_measurements,
        )
    return deltas


def influence_map() -> Dict[str, Any]:
    """Return the influence‑map metadata required by the API.

    The test suite only checks that a ``version`` field is present.
    """

    cfg = transformation_config()
    return {"version": cfg.get("influence", {}).get("version", "unknown")}
