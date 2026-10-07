"""One Protocol per pipeline step.

An adapter is any object with a matching ``run``. Nothing else is required:
no inheritance, no registration decorator. ``adapters.yaml`` names the concrete
class by dotted path and ``registry`` instantiates it.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from contracts import (
    CalibrationResult,
    CaptureInput,
    CaptureResult,
    DrapeResult,
    DraftResult,
    GarmentSpec,
    GarmentType,
    LandmarkResult,
    MeasurementResult,
    ProductionResult,
    SegmentationInput,
    SegmentationResult,
    ValidationResult,
)


@runtime_checkable
class GarmentSpecifier(Protocol):
    def run(self, garment_type: GarmentType) -> GarmentSpec: ...


@runtime_checkable
class Capturer(Protocol):
    def run(self, inp: CaptureInput) -> CaptureResult: ...


@runtime_checkable
class Segmenter(Protocol):
    def run(self, inp: SegmentationInput) -> SegmentationResult: ...


@runtime_checkable
class Calibrator(Protocol):
    def run(self, image_sizes: dict[str, int], height_cm: float) -> CalibrationResult: ...


@runtime_checkable
class Landmarker(Protocol):
    def run(self, image: str, size: dict[str, int]) -> LandmarkResult: ...


@runtime_checkable
class Measurer(Protocol):
    """Step 6. Two implementations, one identical result type."""

    def run(self, height_cm: float, weight_kg: float | None) -> MeasurementResult: ...


@runtime_checkable
class Validator(Protocol):
    def run(self, spec: GarmentSpec, measurements: MeasurementResult) -> ValidationResult: ...


@runtime_checkable
class Drafter(Protocol):
    def run(self, spec: GarmentSpec, measurements: MeasurementResult, route: str) -> DraftResult: ...


@runtime_checkable
class Producer(Protocol):
    def run(self, spec: GarmentSpec, measurements: MeasurementResult) -> ProductionResult: ...


@runtime_checkable
class Draper(Protocol):
    def run(self, spec: GarmentSpec) -> DrapeResult: ...
