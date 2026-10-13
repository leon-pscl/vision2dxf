"""One Design 3 trial, in one fresh child process.

The benchmark protocol requires every trial to run in a fresh process, and requires the
memory pass and the time pass to be separate passes with tracemalloc off during timing.
This module is that child: it runs exactly one trial and prints one JSON object to stdout.
`run_design_3.py` spawns it and aggregates.

    python benchmark/_child.py --case d3-case-001-central --pass memory
    python benchmark/_child.py --case d3-case-001-central --pass time

Thread counts are set from the frozen configuration before NumPy is imported, so no trial
is timed under a different BLAS thread count than the next one.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _set_threads() -> None:
    threading = yaml.safe_load(
        (ROOT / "benchmark" / "config-design-3.yaml").read_text(encoding="utf-8")
    )["threading"]
    for name, value in threading.items():
        os.environ[name] = str(value)


_set_threads()

# Common libraries are imported before tracing starts: they are shared with the other
# designs, so their allocations are not Design 3's to account for. The Design 3 program
# itself is imported after tracing starts, which is what "the initialized configuration
# retained in Python" means.
import numpy  # noqa: E402,F401
import shapely  # noqa: E402,F401
import ezdxf  # noqa: E402,F401

sys.path.insert(0, str(ROOT / "backend"))


def build_engine() -> dict:
    """Initialise the Design 3 program and load its static configuration.

    This is the analogue of the protocol's `build_design_3_engine`: it imports the
    production modules and touches every frozen configuration, so those allocations are
    retained and counted as current rather than as a one-off peak.
    """
    import app.config
    import app.design3_interface
    import app.export.dxf
    import app.pipeline
    import app.schemas.garmage

    return {
        "reconstruction": app.config.reconstruction_config()["version"],
        "drafting": app.config.drafting_config()["version"],
        "software": app.pipeline.SOFTWARE_VERSION,
        "schema": app.design3_interface.SCHEMA_VERSION,
    }


def load_inputs(case_id: str) -> tuple[dict, dict]:
    config = yaml.safe_load((ROOT / "benchmark" / "config-design-3.yaml")
                            .read_text(encoding="utf-8"))
    entry = next(c for c in config["cases"] if c["case_id"] == case_id)
    measurements = yaml.safe_load((ROOT / entry["measurements"]).read_text(encoding="utf-8"))
    garment = yaml.safe_load((ROOT / config["garment"]).read_text(encoding="utf-8"))
    # The fixture carries provenance alongside `body`; the engine reads only the body.
    return measurements["body"], garment["settings"]


def summarise(run) -> dict:
    """The fields the trial CSV needs from a successful run."""
    panels = run.diagnostics["panels"]
    mismatches = [v for k, v in run.report["metrics"].items() if k.startswith("seamMismatch:")]
    return {
        "valid_output": run.report["status"] == "pass",
        "error_code": "",
        "total_vertices": run.diagnostics["weightTotals"]["totalVertices"],
        "total_triangles": run.diagnostics["weightTotals"]["totalTriangles"],
        "solver": "+".join(sorted({p["solver"] for p in panels})),
        "converged": all(p["converged"] for p in panels),
        "iterations": max((p["iterations"] for p in panels), default=0),
        "flipped_triangles": sum(p["flippedTriangles"] for p in panels),
        "max_seam_mismatch_mm": round(max(mismatches, default=0.0), 6),
        "mesh_diagnostics": run.diagnostics,
        "validation_status": run.report["status"],
    }


def memory_pass(measurements: dict, garment: dict) -> dict:
    """Metric 1: Python-tracked peak memory, traced around the whole generation."""
    import tracemalloc

    from app.design3_interface import Design3Failure, run_design_3

    gc.collect()
    tracemalloc.start()
    build_engine()
    # Reset the peak after initialisation: the retained configuration stays counted as
    # current, but the import spike that built it does not.
    tracemalloc.reset_peak()

    try:
        with tempfile.TemporaryDirectory() as tmp:
            run = run_design_3(measurements, garment, dxf_output=Path(tmp) / "pattern.dxf")
        out = summarise(run)
    except Design3Failure as exc:
        out = {"valid_output": False, "error_code": exc.code, "mesh_diagnostics": None,
               "message": exc.message}
    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    out.update({
        "time_s": None,
        "peak_memory_mb": peak_bytes / (1024 ** 2),
        "current_memory_mb": current_bytes / (1024 ** 2),
    })
    return out


def time_pass(measurements: dict, garment: dict) -> dict:
    """Metric 3: generation time, with tracemalloc off so it costs nothing."""
    from time import perf_counter_ns

    from app.design3_interface import Design3Failure, run_design_3

    build_engine()

    try:
        with tempfile.TemporaryDirectory() as tmp:
            start_ns = perf_counter_ns()
            run = run_design_3(measurements, garment, dxf_output=Path(tmp) / "pattern.dxf")
            end_ns = perf_counter_ns()
        out = summarise(run)
    except Design3Failure as exc:
        end_ns = perf_counter_ns()
        out = {"valid_output": False, "error_code": exc.code, "mesh_diagnostics": None,
               "message": exc.message}

    out.update({
        "time_s": (end_ns - start_ns) / 1_000_000_000.0,
        "peak_memory_mb": None,
    })
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--pass", dest="pass_type", required=True, choices=("memory", "time"))
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--warmup", action="store_true")
    args = parser.parse_args(argv)

    record = {"case_id": args.case, "pass_type": args.pass_type, "trial": args.trial,
              "warmup": args.warmup, "python_version": sys.version.split()[0],
              "platform": sys.platform}

    try:
        measurements, garment = load_inputs(args.case)
        record.update((memory_pass if args.pass_type == "memory" else time_pass)
                      (measurements, garment))
    except Exception as exc:  # a trial that blows up is a trial, and it stays in the log
        record.update({
            "valid_output": False, "error_code": "UNCAUGHT_EXCEPTION",
            "message": str(exc), "traceback": traceback.format_exc()[-2000:],
            "time_s": None, "peak_memory_mb": None,
        })

    print(json.dumps(record, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())