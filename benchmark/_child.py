"""One Design 2 trial, in one fresh child process.

The benchmark protocol requires every trial to run in a fresh process, and requires the
memory pass and the time pass to be separate passes with tracemalloc off during timing.
This module is that child: it runs exactly one trial and prints one JSON object to stdout.
`run_design_2.py` spawns it and aggregates.

    python benchmark/_child.py --case d2-case-002-larger --pass memory
    python benchmark/_child.py --case d2-case-002-larger --pass time

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
        (ROOT / "benchmark" / "config-design-2.yaml").read_text(encoding="utf-8")
    )["threading"]
    for name, value in threading.items():
        os.environ[name] = str(value)


_set_threads()

# Common libraries are imported before tracing starts: they are shared with the other
# designs, so their allocations are not Design 2's to account for. The Design 2 program
# itself is imported after tracing starts, which is what "the initialized base block
# retained in Python memory" means.
import numpy  # noqa: E402,F401
import shapely  # noqa: E402,F401
import ezdxf  # noqa: E402,F401

sys.path.insert(0, str(ROOT / "backend"))


def build_engine() -> dict:
    """Initialise the Design 2 program and load the frozen base block.

    This is the analogue of the protocol's `build_design_2_engine`: it imports the
    production modules and loads the pinned block, so those allocations are retained and
    counted as current rather than as a one-off parsing peak.
    """
    import app.config
    import app.design2_interface
    import app.export.dxf
    import app.serialization.garmageset

    from app.base_block.loader import resolve_block

    block = resolve_block(_config()["base_block"]["id"])
    return {
        "transformation": app.config.transformation_config()["version"],
        "drafting": app.config.drafting_config()["version"],
        "software": app.design2_interface.__name__,
        "schema": app.design2_interface.SCHEMA_VERSION,
        "baseBlock": {"id": block.id, "version": block.version},
    }


def _config() -> dict:
    return yaml.safe_load(
        (ROOT / "benchmark" / "config-design-2.yaml").read_text(encoding="utf-8"))


def load_inputs(case_id: str) -> tuple[dict, dict]:
    config = _config()
    entry = next(c for c in config["cases"] if c["case_id"] == case_id)
    measurements = yaml.safe_load((ROOT / entry["measurements"]).read_text(encoding="utf-8"))
    garment = yaml.safe_load((ROOT / config["garment"]).read_text(encoding="utf-8"))
    # The fixture carries provenance alongside `body`; the engine reads only the body.
    return measurements["body"], garment["settings"]


def summarise(run) -> dict:
    """The fields the trial CSV needs from a successful run."""
    panels = run.diagnostics["panels"]
    violations = sum(len(p["constraintViolations"]) + len(p["mergedLandmarks"])
                     for p in panels)
    return {
        "valid_output": run.report["status"] == "pass",
        "error_code": "",
        "solver_status": "+".join(
            f"{p['method']}:{'converged' if p['converged'] else 'not_converged'}"
            for p in panels),
        "iterations": max((p["iterations"] for p in panels), default=0),
        # A bound violation is a declared limit the result reached or passed: a constraint
        # the solver could not satisfy, or the transformation envelope left behind.
        "bound_violation": violations > 0,
        "constraint_violations": violations,
        "landmark_count": sum(p["landmarks"] or 0 for p in panels),
        "max_landmark_residual_mm": round(
            max((p["maxResidualMm"] or 0.0 for p in panels), default=0.0), 6),
        "max_seam_mismatch_mm": round(
            max((m["mismatchMm"] for m in run.diagnostics["seamMismatchAfterMm"]), default=0.0),
            6) if run.diagnostics.get("seamMismatchAfterMm") else 0.0,
        "seam_balance_rounds": run.diagnostics.get("seamBalanceRounds"),
        "unmapped_deltas": len(run.diagnostics["unmappedDeltas"]),
        "panel_count": len(run.pattern["panels"]),
        "stitch_count": len(run.pattern["stitches"]),
        "validation_status": run.report["status"],
    }


def _failure(exc) -> dict:
    return {"valid_output": False, "error_code": exc.code, "message": exc.message,
            "details": exc.details, "solver_status": "", "iterations": None,
            "bound_violation": True, "panel_count": None, "stitch_count": None,
            "landmark_count": None, "max_landmark_residual_mm": None,
            "max_seam_mismatch_mm": None, "seam_balance_rounds": None,
            "unmapped_deltas": None, "validation_status": "fail"}


def memory_pass(measurements: dict, garment: dict) -> dict:
    """Metric 1: Python-tracked peak memory, traced around the whole generation."""
    import tracemalloc

    from app.design2_interface import Design2Failure, run_design_2

    gc.collect()
    tracemalloc.start()
    build_engine()
    # Reset the peak after initialisation: the retained base block stays counted as
    # current, but the import and parse spike that built it does not.
    tracemalloc.reset_peak()

    try:
        with tempfile.TemporaryDirectory() as tmp:
            run = run_design_2(measurements, garment, dxf_output=Path(tmp) / "pattern.dxf")
        out = summarise(run)
    except Design2Failure as exc:
        out = _failure(exc)
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

    from app.design2_interface import Design2Failure, run_design_2

    build_engine()

    try:
        with tempfile.TemporaryDirectory() as tmp:
            start_ns = perf_counter_ns()
            run = run_design_2(measurements, garment, dxf_output=Path(tmp) / "pattern.dxf")
            end_ns = perf_counter_ns()
        out = summarise(run)
    except Design2Failure as exc:
        end_ns = perf_counter_ns()
        out = _failure(exc)

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