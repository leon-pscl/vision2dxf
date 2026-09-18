from __future__ import annotations
from ..core.models import MeasurementProfile, PatternGenerator
from ..core.validation import validate_pattern


def measure_reliability(
    gen: PatternGenerator,
    profiles: list[MeasurementProfile],
    runs_per_profile: int = 5,
) -> list[dict]:
    results = []
    for profile in profiles:
        failures = 0
        total = runs_per_profile
        run_details = []
        for i in range(total):
            try:
                result = gen.generate(profile)
                result = validate_pattern(result)
                if not result.valid:
                    failures += 1
                    run_details.append({"run": i + 1, "failed": True, "errors": result.validation_errors})
                else:
                    run_details.append({"run": i + 1, "failed": False})
            except Exception as e:
                failures += 1
                run_details.append({"run": i + 1, "failed": True, "errors": [str(e)]})

        failure_rate = (failures / total) * 100
        results.append({
            "design_id": gen.design_id,
            "profile_id": profile.profile_id,
            "total_runs": total,
            "failures": failures,
            "failure_rate_percent": failure_rate,
            "runs": run_details,
        })
    return results
