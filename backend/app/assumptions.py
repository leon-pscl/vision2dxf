"""Assumption registry.

The input is a handful of scalar measurements, not a scan, so most of the reconstructed
surface rests on something nobody measured. Every such value is recorded here and written
to generation-manifest.json. Nothing in this registry ever reaches the compatibility
pattern JSON.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SurfaceAssumption:
    id: str
    name: str
    value: float | str
    unit: str | None
    reason: str
    source_fields: list[str]
    configurable: bool

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "reason": " ".join(self.reason.split()),
            "sourceFields": self.source_fields,
            "configurable": self.configurable,
        }


@dataclass
class AssumptionRegistry:
    entries: list[SurfaceAssumption] = field(default_factory=list)

    def add(
        self,
        id: str,
        name: str,
        value: float | str,
        unit: str | None,
        reason: str,
        source_fields: list[str] | None = None,
        configurable: bool = True,
    ) -> None:
        self.entries.append(
            SurfaceAssumption(id, name, value, unit, reason, source_fields or [], configurable)
        )

    def as_list(self) -> list[dict]:
        return [a.as_dict() for a in self.entries]

    def __len__(self) -> int:
        return len(self.entries)
