"""Step 6b mock: parametric body model fit."""

from __future__ import annotations

from contracts import MeasurementBackend, MeasurementResult

from . import measure_common


class MockBodyModel:
    def run(self, height_cm: float, weight_kg: float | None) -> MeasurementResult:
        return measure_common.measure(MeasurementBackend.BODY_MODEL, height_cm, weight_kg)
