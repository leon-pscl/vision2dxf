"""Small benchmark for the design-2 base-block transformation engine.

Run with:
    python backend/tests/benchmark_design2.py

It measures, after a measurement profile has already been resolved into
transformation measurements, the wall-clock time and peak Python memory consumed
by one ``transform()`` call. Profiling is done with the standard-library ``time``
and ``tracemalloc`` modules so the script runs without extra dependencies.
"""
from __future__ import annotations

import statistics
import sys
import time
import tracemalloc
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parents[0]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.base_block.measurements import default_confirmations
from app.config import drafting_config, transformation_config
from app.drafting.measurements import parse_source
from app.drafting.semantics import resolve
from app.drafting.ease import Settings
from app.transformation.pipeline import resolve_semantics, transform

FIXTURES = BACKEND.parents[0] / "fixtures"
REPEATS = 5
# Profiles that resolve cleanly with the default confirmations and transform
# successfully under design 2.
FIXTURE_NAMES = (
    "body_measurements.yaml",
    "body_measurements_target_a.yaml",
    "body_measurements_target_b.yaml",
)


def benchmark_one(name: str) -> dict[str, float]:
    profile, resolution = resolve_semantics(
        (FIXTURES / name).read_text(encoding="utf-8"), name, default_confirmations()
    )
    assert resolution.can_generate

    times: list[float] = []
    peaks: list[int] = []
    for _ in range(REPEATS):
        tracemalloc.start()
        start = time.perf_counter()
        result = transform(resolution, Settings())
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
    print("design-2 transformation benchmark")
    print(f"repeats per fixture: {REPEATS}")
    print(f"drafting.yaml version: {drafting_config()['version']}")
    print(f"transformation.yaml version: {transformation_config().get('version')}")
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
