"""Step 6a mock: direct regression from the silhouette."""

from __future__ import annotations

from contracts import MeasurementBackend, MeasurementResult

from . import measure_common


class MockRegression:
    def run(self, height_cm: float, weight_kg: float | None) -> MeasurementResult:
        return measure_common.measure(MeasurementBackend.REGRESSION, height_cm, weight_kg)
