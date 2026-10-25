"""Small benchmark for the design-3 anthropometric surface-reconstruction and
mesh-flattening pipeline.

Run with:
    cd backend && ../.venv-dev/Scripts/python.exe tests/benchmark_design3.py
  (or any other interpreter that has the project dependencies)

It measures, after a measurement profile has already been resolved into drafting
measurements, the wall-clock time, storage consumption, pattern-complexity
growth and memory consumption of one ``generate()`` call plus PDF export.
"""
from __future__ import annotations

import builtins
import csv
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

_OPENED_FILES: set[str] = set()


def _record_path(path) -> None:
    try:
        if path is None:
            return
        s = os.fspath(path)
    except TypeError:
        try:
            s = str(path)
        except Exception:
            return
    try:
        p = Path(s)
        if p.is_file():
            _OPENED_FILES.add(str(p.resolve()))
    except (OSError, RuntimeError, ValueError):
        pass


_ORIGINAL_OPEN = builtins.open


def _tracked_open(file, *args, **kwargs):
    _record_path(file)
    return _ORIGINAL_OPEN(file, *args, **kwargs)


builtins.open = _tracked_open

_ORIGINAL_READ_TEXT = Path.read_text
_ORIGINAL_READ_BYTES = Path.read_bytes


def _tracked_read_text(self, *args, **kwargs):
    _record_path(self)
    return _ORIGINAL_READ_TEXT(self, *args, **kwargs)


def _tracked_read_bytes(self, *args, **kwargs):
    _record_path(self)
    return _ORIGINAL_READ_BYTES(self, *args, **kwargs)


Path.read_text = _tracked_read_text
Path.read_bytes = _tracked_read_bytes

import psutil

from app.config import reconstruction_config
from app.drafting.ease import Settings
from app.drafting.semantics import default_confirmations
from app.export.pdf import export_pdf
from app.pipeline import generate, resolve_semantics

FIXTURES = REPO_ROOT / "fixtures"
RESULTS_DIR = BACKEND / "results"
PDF_PATH = RESULTS_DIR / "benchmark_design3_output.pdf"

FIXTURE_NAMES = (
    "body_measurements.yaml",
)
PERF_REPEATS = 30
ECON_REPEATS = 30
SCALABILITY_REPEATS = 15
SCALABILITY_LOW = "body_measurements.yaml"
SCALABILITY_HIGH = "body_measurements.yaml"
SAMPLER_INTERVAL = 0.01


def git_branch() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _confirmations() -> dict[str, str]:
    return default_confirmations()


def _load_resolution(name: str):
    text = (FIXTURES / name).read_text(encoding="utf-8")
    profile, resolution = resolve_semantics(text, name, _confirmations())
    assert resolution.can_generate, f"{name}: semantics did not resolve: {resolution.blocking}"
    return profile, resolution


def _one_generation(name: str):
    profile, resolution = _load_resolution(name)
    settings = Settings(confirm_units=True)
    start = time.perf_counter()
    result = generate(resolution, profile, settings)
    pdf_bytes = export_pdf(
        result.document, result.cutting, result.settings.paper_size,
        result.settings.paper_orientation,
    )
    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    PDF_PATH.write_bytes(pdf_bytes)
    elapsed = time.perf_counter() - start
    assert result.report.status == "pass"
    return elapsed, result


def warm_up() -> None:
    print("  running warm-up trial...")
    start = time.perf_counter()
    _one_generation(FIXTURE_NAMES[0])
    print(f"  warm-up finished in {time.perf_counter() - start:.3f} s")


def performance() -> dict:
    out: dict = {"trials": {}, "summary": {}}
    for name in FIXTURE_NAMES:
        times: list[float] = []
        print(f"  performance: {name} ({PERF_REPEATS} trials)")
        for i in range(PERF_REPEATS):
            elapsed, _ = _one_generation(name)
            times.append(elapsed)
            print(f"    trial {i + 1:>2}/{PERF_REPEATS}: {elapsed:.6f} s", flush=True)
        out["trials"][name] = times
        out["summary"][name] = {
            "n": len(times),
            "median_seconds": statistics.median(times),
            "min_seconds": min(times),
            "max_seconds": max(times),
        }
    return out


def efficiency() -> dict:
    # Keep the files recorded during module import (algorithm source files)
    # and add anything opened by this run.
    _one_generation(FIXTURE_NAMES[0])

    files: list[dict] = []
    for path in sorted(_OPENED_FILES):
        if os.path.isfile(path):
            size = os.path.getsize(path)
            files.append({"path": path, "bytes": size, "mb": size / 1024 ** 2})

    if os.path.isfile(str(PDF_PATH)):
        size = os.path.getsize(str(PDF_PATH))
        files.append({"path": str(PDF_PATH), "bytes": size, "mb": size / 1024 ** 2})

    total_bytes = sum(f["bytes"] for f in files)
    result = {
        "files": files,
        "total_bytes": total_bytes,
        "total_mb": total_bytes / 1024 ** 2,
    }

    print("Efficiency - files consumed:")
    for f in files:
        print(f"  {f['bytes']:>12}  {f['mb']:.6f} MB  {f['path']}")
    print(f"  TOTAL {total_bytes} bytes = {result['total_mb']:.6f} MB")
    return result


def scalability() -> dict:
    low_times: list[float] = []
    high_times: list[float] = []
    print(f"  scalability low  (2 panels, {SCALABILITY_REPEATS} trials)")
    for i in range(SCALABILITY_REPEATS):
        elapsed, _ = _one_generation(SCALABILITY_LOW)
        low_times.append(elapsed)
        print(f"    trial {i + 1:>2}/{SCALABILITY_REPEATS}: {elapsed:.6f} s", flush=True)
    print(f"  scalability high (19 panels, {SCALABILITY_REPEATS} trials)")
    for i in range(SCALABILITY_REPEATS):
        elapsed, _ = _one_generation(SCALABILITY_HIGH)
        high_times.append(elapsed)
        print(f"    trial {i + 1:>2}/{SCALABILITY_REPEATS}: {elapsed:.6f} s", flush=True)

    low_median = statistics.median(low_times)
    high_median = statistics.median(high_times)
    gc = high_median / low_median if low_median else float("inf")

    return {
        "low": {"panels": 2, "n": len(low_times),
                "median_seconds": low_median,
                "min_seconds": min(low_times),
                "max_seconds": max(low_times),
                "trials": low_times},
        "high": {"panels": 19, "n": len(high_times),
                 "median_seconds": high_median,
                 "min_seconds": min(high_times),
                 "max_seconds": max(high_times),
                 "trials": high_times},
        "gc": gc,
    }


def _memory_sampler(proc, samples, stop_event):
    while not stop_event.is_set():
        try:
            samples.append(proc.memory_info().rss)
        except Exception:
            pass
        stop_event.wait(SAMPLER_INTERVAL)


def economic() -> dict:
    proc = psutil.Process()
    trials: list[dict] = []
    for i in range(ECON_REPEATS):
        print(f"  economic trial {i + 1:>2}/{ECON_REPEATS}", flush=True)
        baseline_rss = proc.memory_info().rss
        samples: list[int] = []
        stop_event = threading.Event()
        sampler = threading.Thread(
            target=_memory_sampler, args=(proc, samples, stop_event), daemon=True
        )
        sampler.start()
        _one_generation(FIXTURE_NAMES[0])
        stop_event.set()
        sampler.join()

        mean_rss = statistics.mean(samples) if samples else baseline_rss
        trials.append({
            "baseline_rss_bytes": baseline_rss,
            "baseline_mb": baseline_rss / 1024 ** 2,
            "samples": len(samples),
            "mean_rss_bytes": mean_rss,
            "mean_mb": mean_rss / 1024 ** 2,
        })

    avg_mean_mb = statistics.mean(t["mean_mb"] for t in trials)
    avg_baseline_mb = statistics.mean(t["baseline_mb"] for t in trials)
    return {
        "trials": trials,
        "average_mb": avg_mean_mb,
        "average_baseline_mb": avg_baseline_mb,
        "n": len(trials),
    }


def _write_results(branch, perf, eff, scal, econ) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for name, times in perf["trials"].items():
        for i, v in enumerate(times):
            rows.append({
                "branch": branch, "record_type": "raw", "metric": "performance",
                "fixture": name, "level": "", "trial": i, "value": v,
                "unit": "seconds", "statistic": "",
            })
    for i, t in enumerate(scal["low"]["trials"]):
        rows.append({"branch": branch, "record_type": "raw", "metric": "scalability",
                     "fixture": SCALABILITY_LOW, "level": "low_2_panels", "trial": i,
                     "value": t, "unit": "seconds", "statistic": ""})
    for i, t in enumerate(scal["high"]["trials"]):
        rows.append({"branch": branch, "record_type": "raw", "metric": "scalability",
                     "fixture": SCALABILITY_HIGH, "level": "high_19_panels", "trial": i,
                     "value": t, "unit": "seconds", "statistic": ""})
    for i, t in enumerate(econ["trials"]):
        rows.append({"branch": branch, "record_type": "raw", "metric": "economic",
                     "fixture": FIXTURE_NAMES[0], "level": "", "trial": i,
                     "value": t["mean_mb"], "unit": "MB", "statistic": ""})

    for name, s in perf["summary"].items():
        rows.append({"branch": branch, "record_type": "summary",
                     "metric": "performance", "fixture": name, "level": "",
                     "trial": "", "value": s["median_seconds"], "unit": "seconds",
                     "statistic": "median"})
        rows.append({"branch": branch, "record_type": "summary",
                     "metric": "performance", "fixture": name, "level": "",
                     "trial": "", "value": s["min_seconds"], "unit": "seconds",
                     "statistic": "min"})
        rows.append({"branch": branch, "record_type": "summary",
                     "metric": "performance", "fixture": name, "level": "",
                     "trial": "", "value": s["max_seconds"], "unit": "seconds",
                     "statistic": "max"})
    rows.append({"branch": branch, "record_type": "summary", "metric": "efficiency",
                 "fixture": "", "level": "", "trial": "", "value": eff["total_mb"],
                 "unit": "MB", "statistic": "total"})
    rows.append({"branch": branch, "record_type": "summary", "metric": "scalability",
                 "fixture": SCALABILITY_LOW, "level": "low_2_panels", "trial": "",
                 "value": scal["low"]["median_seconds"], "unit": "seconds",
                 "statistic": "median"})
    rows.append({"branch": branch, "record_type": "summary", "metric": "scalability",
                 "fixture": SCALABILITY_HIGH, "level": "high_19_panels", "trial": "",
                 "value": scal["high"]["median_seconds"], "unit": "seconds",
                 "statistic": "median"})
    rows.append({"branch": branch, "record_type": "summary", "metric": "scalability",
                 "fixture": "", "level": "", "trial": "", "value": scal["gc"],
                 "unit": "ratio", "statistic": "gc"})
    rows.append({"branch": branch, "record_type": "summary", "metric": "economic",
                 "fixture": FIXTURE_NAMES[0], "level": "", "trial": "",
                 "value": econ["average_mb"], "unit": "MB", "statistic": "average"})
    rows.append({"branch": branch, "record_type": "summary", "metric": "economic",
                 "fixture": FIXTURE_NAMES[0], "level": "", "trial": "",
                 "value": econ["average_baseline_mb"], "unit": "MB",
                 "statistic": "baseline"})

    csv_path = RESULTS_DIR / "benchmark_design3.csv"
    fieldnames = ["branch", "record_type", "metric", "fixture", "level", "trial",
                  "value", "unit", "statistic"]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    json_path = RESULTS_DIR / "benchmark_design3.json"
    json_path.write_text(json.dumps({
        "branch": branch,
        "performance": perf,
        "efficiency": eff,
        "scalability": scal,
        "economic": econ,
    }, indent=2), encoding="utf-8")

    print(f"\nResults written to {csv_path} and {json_path}")


def main() -> None:
    print("design-3 reconstruction and flattening benchmark")
    print(f"branch: {git_branch()}")
    print(f"reconstruction.yaml version: {reconstruction_config()['version']}")
    print("warming up...")
    warm_up()
    print("warm-up done\n")

    branch = git_branch()
    perf = performance()
    print("\nPerformance (median inference time, s):")
    for name, s in perf["summary"].items():
        print(f"  {name}: median={s['median_seconds']:.6f} min={s['min_seconds']:.6f} max={s['max_seconds']:.6f}")

    eff = efficiency()

    scal = scalability()
    print("\nScalability:")
    print(f"  low  (2 panels):  median={scal['low']['median_seconds']:.6f} s")
    print(f"  high (19 panels): median={scal['high']['median_seconds']:.6f} s")
    print(f"  GC = {scal['gc']:.6f}")

    econ = economic()
    print("\nEconomic (average memory, MB):")
    print(f"  average={econ['average_mb']:.6f}  baseline={econ['average_baseline_mb']:.6f}")

    _write_results(branch, perf, eff, scal, econ)

    print("\nSummary table:")
    print(f"{'metric':<16}{'value':>14}")
    print("-" * 30)
    medians = [s["median_seconds"] for s in perf["summary"].values()]
    print(f"{'perf median (s)':<16}{statistics.median(medians):>14.6f}")
    print(f"{'efficiency (MB)':<16}{eff['total_mb']:>14.6f}")
    print(f"{'scalability GC':<16}{scal['gc']:>14.6f}")
    print(f"{'economic (MB)':<16}{econ['average_mb']:>14.6f}")


if __name__ == "__main__":
    main()
