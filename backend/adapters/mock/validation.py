"""Step 7 mock: validation flags plus the YAML payload.

The flags are real checks against the real measurements, so a nonsense ease
value actually trips a rule. The YAML is what the frontend downloads.
"""

from __future__ import annotations

import yaml as pyyaml

from contracts import (
    GarmentSpec,
    MeasurementResult,
    ValidationFlag,
    ValidationResult,
)

from ._common import cfg


class MockValidation:
    def run(self, spec: GarmentSpec, measurements: MeasurementResult) -> ValidationResult:
        conf = cfg()["validation"]
        flags: list[ValidationFlag] = []

        for name, ease in spec.ease_cm.items():
            m = measurements.measurements.get(name)
            if m is None:
                continue

            if ease > float(conf["max_ease_cm"]):
                flags.append(
                    ValidationFlag(
                        measurement=name,
                        rule="ease_max",
                        severity="warning",
                        detail=f"{ease:.1f} cm ease exceeds the {conf['max_ease_cm']:.0f} cm ceiling",
                    )
                )
            if ease < float(conf["min_ease_cm"]):
                flags.append(
                    ValidationFlag(
                        measurement=name,
                        rule="ease_min",
                        severity="warning",
                        detail=f"{ease:.1f} cm ease is negative, it will pull the garment",
                    )
                )

            if m.lower_cm is None or m.upper_cm is None:
                flags.append(
                    ValidationFlag(
                        measurement=name,
                        rule="interval_missing",
                        severity="error",
                        detail="no interval, measurement cannot be trusted for cutting",
                    )
                )
                continue

            width = m.upper_cm - m.lower_cm
            if width > abs(m.value_cm) * 0.15:
                flags.append(
                    ValidationFlag(
                        measurement=name,
                        rule="interval_wide",
                        severity="warning",
                        detail=f"interval spans {width:.1f} cm, over 15% of the value",
                    )
                )

        # a top needs a length that reaches the hip; without one the draft is open
        if "slacks" not in spec.garment_type.value and not any(
            f.measurement in ("back_length", "outseam") for f in flags
        ):
            flags.append(
                ValidationFlag(
                    measurement="back_length",
                    rule="length_missing",
                    severity="info",
                    detail="no body length available, drafting used the height default",
                )
            )

        if not flags:
            flags.append(
                ValidationFlag(
                    measurement="-",
                    rule="all_clear",
                    severity="info",
                    detail="every rule passed",
                )
            )

        payload = _build_yaml(spec, measurements)
        return ValidationResult(
            is_mock=True,
            flags=flags,
            yaml=payload,
            warnings=["mock validation, measurements are mock too"],
        )


def _build_yaml(spec: GarmentSpec, measurements: MeasurementResult) -> str:
    doc = {
        "session": {
            "garment_type": spec.garment_type.value,
            "backend": measurements.backend.value if measurements.backend else None,
            "is_mock": measurements.is_mock,
        },
        "ease_cm": spec.ease_cm,
        "design_rules": spec.design_rules,
        "measurements": {
            name: {
                "value_cm": m.value_cm,
                "lower_cm": m.lower_cm,
                "upper_cm": m.upper_cm,
                "method": m.method,
            }
            for name, m in measurements.measurements.items()
        },
        "warnings": measurements.warnings,
    }
    return pyyaml.safe_dump(doc, sort_keys=False, default_flow_style=False)
