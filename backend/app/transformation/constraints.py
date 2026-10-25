"""Envelope evaluation utilities.

Provides ``evaluate_envelope`` which checks whether any measurement delta exceeds the
configured limits. The function returns an ``EnvelopeResult`` containing a ``status``
(``ok``, ``warn`` or ``stop``) and a ``confirmed`` flag indicating whether the
operator has explicitly confirmed an out‑of‑envelope transformation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

from ..config import transformation_config


@dataclass
class EnvelopeResult:
    status: str
    confirmed: bool

    def as_dict(self) -> Dict[str, Any]:
        return {"status": self.status, "confirmed": self.confirmed}


def evaluate_envelope(deltas: Dict[str, Any], confirmed: bool = False) -> EnvelopeResult:
    """Evaluate measurement deltas against the transformation envelope.

    ``deltas`` is a mapping of measurement name to a ``Delta`` (or compatible object)
    providing ``baseValue`` and ``deltaValue`` attributes. The configuration lives in
    ``transformation.yaml``. For each measurement we compare the absolute delta to the
    absolute millimetre limit (``max_abs_mm``) and to the percentage limit relative to
    the base value (``max_pct``). If a delta exceeds the limit multiplied by the
    ``stop_factor`` the transformation is refused (``stop``). If it exceeds the limit
    but not the stop factor we return ``warn`` when the operator has confirmed the
    envelope, otherwise ``exceeded`` (handled upstream as a blocked transformation).
    """

    cfg = transformation_config()
    default_cfg = cfg["envelope"]["default"]
    specific_cfg = cfg["envelope"].get("measurements", {})
    stop_factor = cfg["envelope"].get("stop_factor", 2.0)

    exceeded = False
    for name, delta in deltas.items():
        # ``delta`` may be a ``Delta`` dataclass or a simple dict‑like object.
        base = getattr(delta, "baseValue", getattr(delta, "base_value", 0.0))
        delta_val = getattr(delta, "deltaValue", getattr(delta, "delta_value", 0.0))
        # Skip measurements that have no reference value – they are unmapped and not
        # subject to envelope limits.
        if not getattr(delta, "mapped", False):
            continue
        cfg_entry = specific_cfg.get(name, default_cfg)
        max_abs = cfg_entry.get("max_abs_mm")
        max_pct = cfg_entry.get("max_pct")
        # Absolute‑millimetre check.
        if max_abs is not None:
            if abs(delta_val) > max_abs * stop_factor:
                return EnvelopeResult(status="stop", confirmed=confirmed)
            if abs(delta_val) > max_abs:
                exceeded = True
        # Percentage check – avoid division by zero.
        if max_pct is not None and base:
            pct = abs(delta_val) / base * 100.0
            if pct > max_pct:
                exceeded = True
    if exceeded:
        return EnvelopeResult(status="warn" if confirmed else "exceeded", confirmed=confirmed)
    return EnvelopeResult(status="ok", confirmed=confirmed)
