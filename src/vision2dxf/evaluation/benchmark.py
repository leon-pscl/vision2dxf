from __future__ import annotations
import time
import statistics
from ..core.models import MeasurementProfile, PatternGenerator


def benchmark_generator(
    gen: PatternGenerator,
    profiles: list[MeasurementProfile],
    warmup: int = 1,
    repetitions: int = 30,
) -> list[dict]:
    results = []
    for profile in profiles:
        for _ in range(warmup):
            gen.generate(profile)

        observations = []
        for _ in range(repetitions):
            t0 = time.perf_counter_ns()
            gen.generate(profile)
            t1 = time.perf_counter_ns()
            observations.append(t1 - t0)

        median_ns = statistics.median(observations)
        results.append({
            "design_id": gen.design_id,
            "profile_id": profile.profile_id,
            "median_ns": median_ns,
            "median_ms": median_ns / 1e6,
            "min_ns": min(observations),
            "max_ns": max(observations),
            "observations": observations,
        })
    return results
