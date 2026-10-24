"""Small benchmark for the design-1 formula-based drafting engine.

Run with:
    python backend/benchmark_design1.py

It measures, after a measurement profile has already been resolved into drafting
measurements, the wall-clock time and peak Python memory consumed by one
`draft()` call. Profiling is done with the standard-library `time` and
`tracemalloc` modules so the script runs without extra dependencies.
"""
from __future__ import annotations

import statistics
import time
import tracemalloc
from pathlib import Path

from app.config import drafting_config
from app.drafting.measurements import parse_source
from app.drafting.semantics import registry, resolve
from app.drafting.ease import Settings
from app.drafting.pipeline import draft

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
REPEATS = 5
FIXTURE_NAMES = ("body_measurements_small.yaml", "body_measurements_medium.yaml",
                 "body_measurements_large.yaml")


def _confirmations():
    reg = registry()
    return {name: f.candidates[0].id for name, f in reg.fields.items()
            if f.status == "ambiguous"}


def benchmark_one(name: str) -> dict[str, float]:
    profile = parse_source((FIXTURES / name).read_text(encoding="utf-8"), name)
    resolution = resolve(profile.values_cm, _confirmations())
    assert resolution.can_generate

    times: list[float] = []
    peaks: list[int] = []
    for _ in range(REPEATS):
        tracemalloc.start()
        start = time.perf_counter()
        result = draft(resolution, Settings())
        elapsed = time.perf_counter() - start
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        times.append(elapsed)
        peaks.append(peak)
        assert result.report.status == "pass"

    return {
        "fixture": name,
        "repeats": REPEATS,
        "avg_time_seconds": statistics.mean(times),
        "min_time_seconds": min(times),
        "avg_peak_memory_bytes": statistics.mean(peaks),
        "max_peak_memory_bytes": max(peaks),
    }


def main() -> None:
    print("design-1 drafting benchmark")
    print(f"repeats per fixture: {REPEATS}")
    print(f"drafting.yaml version: {drafting_config()['version']}")
    print()
    for name in FIXTURE_NAMES:
        stats = benchmark_one(name)
        print(f"{stats['fixture']}:")
        print(f"  avg time   : {stats['avg_time_seconds'] * 1000:.3f} ms")
        print(f"  min time   : {stats['min_time_seconds'] * 1000:.3f} ms")
        print(f"  avg peak   : {stats['avg_peak_memory_bytes'] / 1024:.1f} KiB")
        print(f"  max peak   : {stats['max_peak_memory_bytes'] / 1024:.1f} KiB")
        print()


if __name__ == "__main__":
    main()