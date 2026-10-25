"""Small benchmark for the design-3 anthropometric surface-reconstruction and
mesh-flattening pipeline.

Run with:
    cd backend && ../.venv-dev/Scripts/python.exe tests/benchmark_design1.py
  (or any other interpreter that has the project dependencies)

It measures, after a measurement profile has already been resolved into drafting
measurements, the wall-clock time and peak Python memory consumed by one
`generate()` call. Profiling is done with the standard-library `time` and
`tracemalloc` modules so the script runs without extra dependencies.
"""
from __future__ import annotations

import statistics
import sys
import time
import tracemalloc
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.config import reconstruction_config
from app.drafting.ease import Settings
from app.drafting.semantics import default_confirmations
from app.pipeline import generate, resolve_semantics

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
REPEATS = 5
FIXTURE_NAMES = ("body_measurements.yaml", "body_measurements_target_a.yaml",
                 "body_measurements_target_b.yaml")


def _confirmations() -> dict[str, str]:
    return default_confirmations()


def benchmark_one(name: str) -> dict[str, float]:
    profile, resolution = resolve_semantics(
        (FIXTURES / name).read_text(encoding="utf-8"), name, _confirmations()
    )
    assert resolution.can_generate, f"{name}: semantics did not resolve: {resolution.blocking}"

    times: list[float] = []
    peaks: list[int] = []
    for _ in range(REPEATS):
        tracemalloc.start()
        start = time.perf_counter()
        result = generate(resolution, profile, Settings(confirm_units=True))
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
    print("design-3 reconstruction and flattening benchmark")
    print(f"repeats per fixture: {REPEATS}")
    print(f"reconstruction.yaml version: {reconstruction_config()['version']}")
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
