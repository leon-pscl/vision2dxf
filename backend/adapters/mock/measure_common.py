"""Step 6 mocks: measurement, backends 6a (regression) and 6b (body model).

Both return the identical MeasurementResult type, intervals included. They
share one estimator so the two backends agree on the value and differ only in
the spread, which is what a real swap would look like.
"""

from __future__ import annotations

from contracts import (
    GarmentType,
    Measurement,
    MeasurementBackend,
    MeasurementResult,
)

from ._common import cfg, seeded

# name -> (cm at BMI 22 / 170 cm, cm per kg over that, cm per 10 cm of height)
_BASE: dict[str, tuple[float, float, float]] = {
    "chest": (96.0, 0.55, 1.6),
    "waist": (82.0, 0.72, 1.2),
    "hip": (98.0, 0.60, 1.4),
    "shoulder_width": (45.0, 0.18, 0.9),
    "back_length": (42.0, 0.14, 0.7),
    "armhole_depth": (21.0, 0.06, 0.25),
    "sleeve_length": (60.0, 0.05, 0.5),
    "outseam": (103.0, 0.10, 1.1),
    "inseam": (78.0, 0.08, 0.8),
    "rise": (27.0, 0.10, 0.35),
    "thigh": (58.0, 0.28, 0.6),
}

# 6b is a body model fit, so it is assumed tighter than 6a. Same shape, one knob.
_SPREAD = {
    MeasurementBackend.REGRESSION: 1.0,
    MeasurementBackend.BODY_MODEL: 0.62,
}


def measure(
    backend: MeasurementBackend,
    height_cm: float,
    weight_kg: float | None,
    required: list[str] | None = None,
) -> MeasurementResult:
    conf = cfg()["measurement"]
    if required is None:
        required = list(_BASE)

    weight = weight_kg if weight_kg else _implied_weight(height_cm)
    # distance from the reference body drives the wobble, so identical inputs
    # always give identical outputs
    # seeded on the body, not the backend: 6a and 6b estimate the same person,
    # so the point values agree and only the interval width differs.
    rnd = seeded("measure", round(height_cm, 2), round(weight, 2))

    out: dict[str, Measurement] = {}
    for name in required:
        base, per_kg, per_10cm = _BASE.get(name, (30.0, 0.1, 0.4))
        value = base + per_kg * (weight - 70.0) + per_10cm * ((height_cm - 170.0) / 10.0)
        value += rnd.uniform(-1.0, 1.0)

        frac = float(conf["default_interval_fraction"]) * _SPREAD[backend]
        half = max(float(conf["min_interval_cm"]) * frac, abs(value) * frac)
        out[name] = Measurement(
            value_cm=round(value, 1),
            lower_cm=round(value - half, 1),
            upper_cm=round(value + half, 1),
            method=f"mock:{backend.value}",
        )

    return MeasurementResult(
        is_mock=True,
        backend=backend,
        measurements=out,
        warnings=[
            f"mock {backend.value} backend",
            "uncalibrated: no conformal intervals, spread is a fixed fraction",
        ],
    )


def _implied_weight(height_cm: float) -> float:
    """No weight given: assume the reference BMI of 22."""
    return 22.0 * (height_cm / 100) ** 2


def required_for(garment_type: GarmentType) -> list[str]:
    from .garment_spec import CATALOGUE

    return list(CATALOGUE[garment_type]["required_measurements"])
