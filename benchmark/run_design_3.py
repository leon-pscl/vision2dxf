"""The Design 3 benchmark runner.

Every trial runs in a fresh child process (`_child.py`), one process per trial, so no
trial inherits another's imports, caches or allocator state. The memory pass and the time
pass are separate passes, as the protocol requires.

    python benchmark/run_design_3.py --smoke     # 1 case, 3 + 3 trials, proves the harness
    python benchmark/run_design_3.py             # the compliant run

Nothing here filters, retries or replaces a trial. A case that fails stays in the trial log
with its error code, and every trial - warm-up included - is written out.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark"))

import yaml  # noqa: E402

from environment import config_fingerprint, configuration_version, environment_record, git_state  # noqa: E402

DESIGN = "Design 3 - Anthropometric Surface Reconstruction and Constrained Mesh Flattening"

TRIAL_FIELDS = [
    "design", "case_id", "pass_type", "trial", "warmup", "time_s", "peak_memory_mb",
    "valid_output", "error_code", "total_vertices", "total_triangles", "solver", "converged",
    "iterations", "flipped_triangles", "max_seam_mismatch_mm", "config_version", "config_hash",
    "commit_hash", "python_version", "platform", "timestamp_utc",
]

SUMMARY_FIELDS = [
    "case_id", "role", "valid_memory_trials", "valid_time_trials", "memory_median_mb",
    "memory_iqr_mb", "time_median_s", "time_iqr_s", "all_mesh_checks_passed", "failure_count",
]

DIAGNOSTIC_FIELDS = [
    "case_id", "trial", "panel_id", "vertices", "triangles", "components", "boundary_loops",
    "euler_characteristic", "disk_topology", "solver", "converged", "iterations",
    "final_energy", "flipped_triangles", "flipped_pct", "mean_edge_distortion",
    "max_edge_distortion", "mean_area_distortion", "max_area_distortion", "seam_mismatch_mm",
    "boundary_intersections", "valid",
]


def load_config() -> dict:
    return yaml.safe_load((ROOT / "benchmark" / "config-design-3.yaml").read_text(encoding="utf-8"))


def run_trial(case_id: str, pass_type: str, trial: int, warmup: bool) -> dict:
    """One child process, one trial. Never raises: a failed trial is a result."""
    argv = [
        sys.executable, str(ROOT / "benchmark" / "_child.py"),
        "--case", case_id, "--pass", pass_type, "--trial", str(trial),
    ]
    if warmup:
        argv.append("--warmup")
    try:
        completed = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        return {"case_id": case_id, "pass_type": pass_type, "trial": trial, "warmup": warmup,
                "valid_output": False, "error_code": "UNCAUGHT_EXCEPTION",
                "message": "the child process exceeded 900 s", "time_s": None,
                "peak_memory_mb": None}
    if completed.returncode != 0:
        return {"case_id": case_id, "pass_type": pass_type, "trial": trial, "warmup": warmup,
                "valid_output": False, "error_code": "UNCAUGHT_EXCEPTION",
                "message": (completed.stderr or "")[-500:], "time_s": None,
                "peak_memory_mb": None}

    try:
        return json.loads(completed.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"case_id": case_id, "pass_type": pass_type, "trial": trial, "warmup": warmup,
                "valid_output": False, "error_code": "UNCAUGHT_EXCEPTION",
                "message": f"unreadable child output: {completed.stdout[-500:]}",
                "time_s": None, "peak_memory_mb": None}


def trial_row(record: dict, fingerprint: str, commit: str, version: str) -> dict:
    """The trial CSV row. An unmeasured field stays blank; it is never filled with zero."""
    return {
        "design": DESIGN,
        "case_id": record["case_id"],
        "pass_type": record["pass_type"],
        "trial": record["trial"],
        "warmup": record["warmup"],
        "time_s": _round(record.get("time_s"), 9),
        "peak_memory_mb": _round(record.get("peak_memory_mb"), 6),
        "valid_output": record.get("valid_output", False),
        "error_code": record.get("error_code", ""),
        "total_vertices": record.get("total_vertices", ""),
        "total_triangles": record.get("total_triangles", ""),
        "solver": record.get("solver", ""),
        "converged": record.get("converged", ""),
        "iterations": record.get("iterations", ""),
        "flipped_triangles": record.get("flipped_triangles", ""),
        "max_seam_mismatch_mm": record.get("max_seam_mismatch_mm", ""),
        "config_version": version,
        "config_hash": fingerprint,
        "commit_hash": commit,
        "python_version": record.get("python_version", ""),
        "platform": record.get("platform", ""),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _round(value, places: int):
    return "" if value is None else round(value, places)


def euler_characteristic(vertices: int, triangles: int, boundary_loops: int) -> int:
    """V - E + F, with E recovered from the triangle count: 3F = 2E - B."""
    edges = (3 * triangles + boundary_loops) / 2
    return int(vertices - edges + triangles)


def diagnostic_rows(record: dict, mismatch_mm: float) -> list[dict]:
    diagnostics = record.get("mesh_diagnostics") or {}
    rows = []
    for panel in diagnostics.get("panels", []):
        loops = panel.get("boundary_loops", 0)
        rows.append({
            "case_id": record["case_id"],
            "trial": record["trial"],
            "panel_id": panel["panel"],
            "vertices": panel.get("vertices", ""),
            "triangles": panel.get("triangles", ""),
            "components": 1,
            "boundary_loops": loops,
            "euler_characteristic": euler_characteristic(
                panel.get("vertices", 0), panel.get("triangles", 0), loops),
            "disk_topology": panel.get("diskTopology", ""),
            "solver": panel.get("solver", ""),
            "converged": panel.get("converged", ""),
            "iterations": panel.get("iterations", ""),
            "final_energy": panel.get("finalEnergy", ""),
            "flipped_triangles": panel.get("flippedTriangles", ""),
            "flipped_pct": panel.get("flippedPct", ""),
            "mean_edge_distortion": panel.get("meanEdgeDistortion", ""),
            "max_edge_distortion": panel.get("maxEdgeDistortion", ""),
            "mean_area_distortion": panel.get("meanAreaDistortion", ""),
            "max_area_distortion": panel.get("maxAreaDistortion", ""),
            "seam_mismatch_mm": mismatch_mm,
            "boundary_intersections": panel.get("boundaryIntersections", ""),
            "valid": bool(panel.get("diskTopology")) and panel.get("converged") is True
            and panel.get("flippedTriangles") == 0
            and panel.get("boundaryIntersections") == 0,
        })
    return rows


def iqr(values: list[float]) -> float:
    """Interquartile range by the inclusive (linearly interpolated) quartile definition,
    which is what NumPy's default percentile method reports."""
    if len(values) < 2:
        return 0.0
    first, _, third = statistics.quantiles(sorted(values), n=4, method="inclusive")
    return third - first


def summarise(rows: list[dict], roles: dict) -> list[dict]:
    """The case summary, generated from the raw trial rows rather than kept alongside."""
    summary = []
    for case_id in roles:
        case = [r for r in rows if r["case_id"] == case_id and not r["warmup"]]
        memory = [r["peak_memory_mb"] for r in case
                  if r["pass_type"] == "memory" and r["valid_output"]
                  and r["peak_memory_mb"] != ""]
        timing = [r["time_s"] for r in case
                  if r["pass_type"] == "time" and r["valid_output"] and r["time_s"] != ""]
        failures = [r for r in case if not r["valid_output"]]
        checks = [r for r in case if r["valid_output"]]
        summary.append({
            "case_id": case_id,
            "role": roles[case_id],
            "valid_memory_trials": len(memory),
            "valid_time_trials": len(timing),
            "memory_median_mb": _round(statistics.median(memory), 6) if memory else "",
            "memory_iqr_mb": _round(iqr(memory), 6) if memory else "",
            "time_median_s": _round(statistics.median(timing), 9) if timing else "",
            "time_iqr_s": _round(iqr(timing), 9) if timing else "",
            "all_mesh_checks_passed": bool(checks) and not failures,
            "failure_count": len(failures),
        })
    return summary


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="1 case, 3 + 3 trials: proves the harness before the official run.")
    parser.add_argument("--memory-trials", type=int, default=None)
    parser.add_argument("--time-trials", type=int, default=None)
    args = parser.parse_args(argv)

    config = load_config()
    trials = config["trials"]
    if args.smoke:
        selected = [c for c in config["cases"] if c["case_id"] in trials["smoke"]["cases"]]
        warmups = trials["smoke"]["warmups"]
        memory_trials = trials["smoke"]["memory_trials"]
        time_trials = trials["smoke"]["time_trials"]
    else:
        selected = config["cases"]
        warmups = trials["warmups"]
        memory_trials = args.memory_trials or trials["memory_trials"]
        time_trials = args.time_trials or trials["time_trials"]

    outputs = config["outputs"]
    directory = ROOT / outputs["directory"]
    directory.mkdir(parents=True, exist_ok=True)

    fingerprint = config_fingerprint()["combined"]
    commit = git_state()["commitHash"]
    version = configuration_version()

    (directory / outputs["environment_json"]).write_text(
        json.dumps(environment_record(), indent=2, default=str), encoding="utf-8")
    (directory / outputs["config_manifest_json"]).write_text(
        json.dumps({
            "design": DESIGN,
            "configurationVersion": version,
            "configurationHash": fingerprint,
            "configurationHashesPerFile": config_fingerprint()["perFile"],
            "reconstructionConfigVersion": yaml.safe_load(
                (ROOT / "backend/app/config/reconstruction.yaml").read_text(encoding="utf-8")
            )["version"],
            "git": git_state(),
            "cases": selected,
            "trials": {"warmups": warmups, "memory_trials": memory_trials,
                       "time_trials": time_trials, "mode": "smoke" if args.smoke else "full"},
            "threading": config["threading"],
            "maintainabilityAllowList": [f["path"] for f in config["maintainability"]["files"]],
        }, indent=2, default=str),
        encoding="utf-8",
    )

    rows: list[dict] = []
    diagnostics: list[dict] = []
    failures: list[dict] = []
    total = len(selected) * (warmups + memory_trials + time_trials)
    done = 0
    started = time.perf_counter()

    for case in selected:
        case_id = case["case_id"]
        for pass_type, count in (("memory", memory_trials), ("time", time_trials)):
            for trial in range(-warmups, count):
                warmup = trial < 0
                record = run_trial(case_id, pass_type, max(trial, 0), warmup)
                done += 1
                row = trial_row(record, fingerprint, commit, version)
                rows.append(row)
                if not warmup:
                    diagnostics.extend(diagnostic_rows(
                        record, record.get("max_seam_mismatch_mm") or 0.0))
                if not record.get("valid_output", False):
                    failures.append({
                        "timestamp_utc": row["timestamp_utc"],
                        "case_id": case_id,
                        "role": case["role"],
                        "pass_type": pass_type,
                        "trial": row["trial"],
                        "warmup": warmup,
                        "error_code": record.get("error_code", ""),
                        "message": record.get("message", ""),
                        "details": record.get("details", []),
                        "traceback": record.get("traceback", ""),
                    })
                elapsed = time.perf_counter() - started
                print(f"[{done:>4}/{total}] {case_id} {pass_type} trial {row['trial']}"
                      f"{' (warmup)' if warmup else ''}"
                      f"  valid={row['valid_output']} code={row['error_code'] or '-'}"
                      f"  {elapsed:6.1f}s elapsed", flush=True)

    write_csv(directory / outputs["trials_csv"], TRIAL_FIELDS, rows)
    write_csv(directory / outputs["mesh_diagnostics_csv"], DIAGNOSTIC_FIELDS, diagnostics)
    roles = {c["case_id"]: c["role"] for c in config["cases"]}
    write_csv(directory / outputs["case_summary_csv"], SUMMARY_FIELDS, summarise(rows, roles))
    with (directory / outputs["failures_jsonl"]).open("w", encoding="utf-8") as fh:
        for failure in failures:
            fh.write(json.dumps(failure, default=str) + "\n")

    print(f"\n{len(rows)} trials, {len(failures)} failures, "
          f"{time.perf_counter() - started:.1f}s. Results in {outputs['directory']}/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())