"""The Design 2 benchmark runner.

Every trial runs in a fresh child process (`_child.py`), one process per trial, so no
trial inherits another's imports, caches or allocator state. The memory pass and the time
pass are separate passes, as the protocol requires.

    python benchmark/run_design_2.py --smoke     # 1 case, 3 + 3 trials, proves the harness
    python benchmark/run_design_2.py             # the compliant run

Nothing here filters, retries or replaces a trial. A case that fails stays in the trial
log with its error code, and every trial - warm-up included - is written out.

The run refuses to start if the frozen base block does not match the configuration: the
template bytes must hash to what `benchmark/config-design-2.yaml` pins, the catalog must
pin the same hash, and the loader must report the same block id and version. A benchmark
over a base block that is not the one that was frozen measures nothing.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark"))
sys.path.insert(0, str(ROOT / "backend"))

import yaml  # noqa: E402

from environment import (  # noqa: E402
    config,
    config_fingerprint,
    configuration_version,
    environment_record,
    git_state,
)

DESIGN = "Design 2 - GarmageSet-Based Parametric Base-Block Transformation"

TRIAL_FIELDS = [
    "design", "case_id", "pass_type", "trial", "warmup", "time_s", "peak_memory_mb",
    "valid_output", "error_code", "solver_status", "iterations", "bound_violation",
    "base_block_version", "source_sha256", "commit_hash", "python_version", "platform",
    "timestamp_utc",
]

SUMMARY_FIELDS = [
    "case_id", "role", "valid_memory_trials", "valid_time_trials", "memory_median_mb",
    "memory_iqr_mb", "time_median_s", "time_iqr_s", "all_checks_passed", "failure_count",
]


def load_config() -> dict:
    return config()


# ---- the frozen base block --------------------------------------------------


def verify_base_block(cfg: dict) -> dict:
    """Check the frozen block against the configuration and the catalog, and return the
    provenance record the protocol asks for.

    Three independent statements have to agree before any number is produced: the bytes
    on disk, the hash the benchmark configuration pins, and the hash the block catalog
    pins. Any disagreement is a stop, not a warning.
    """
    block = cfg["base_block"]
    source = ROOT / block["source_file"]

    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != block["source_sha256"]:
        raise SystemExit(
            f"The frozen base block template has changed.\n"
            f"  {block['source_file']}\n"
            f"  expected sha256 {block['source_sha256']}\n"
            f"  actual   sha256 {actual}\n"
            "Design 2 always transforms one fixed block. A new block is a new "
            "base_block_version and a complete rerun of every trial."
        )

    catalog = yaml.safe_load(
        (ROOT / "assets/base-blocks/catalog.yaml").read_text(encoding="utf-8"))
    entry = next((b for b in catalog["blocks"] if b["id"] == block["id"]), None)
    if entry is None or entry["sha256"] != block["source_sha256"]:
        raise SystemExit(
            f"catalog.yaml does not pin {block['id']} to the hash the benchmark "
            f"configuration pins ({block['source_sha256']})."
        )

    from app.base_block.loader import resolve_block

    loaded = resolve_block(block["id"])
    for field, expected in (("id", block["id"]), ("version", block["version"]),
                            ("reviewed", block["reviewed"])):
        if getattr(loaded, field) != expected:
            raise SystemExit(
                f"The loader reports {field}={getattr(loaded, field)!r} for "
                f"{block['id']}, but the benchmark configuration freezes {expected!r}."
            )

    return {
        "design": DESIGN,
        "source_dataset": block["source_dataset"],
        "source_file": block["source_file"],
        "source_sha256": actual,
        "source_unit": block["source_unit"],
        "source_unit_documented": block["source_unit_documented"],
        "unit_assumption": block["unit_assumption"],
        "internal_unit": block["internal_unit"],
        "unit_scale_factor": block["unit_scale_factor"],
        "axis_transform": block["axis_transform"],
        "origin_transform": block["origin_transform"],
        "importer_commit": git_state()["commitHash"],
        "base_block_id": loaded.id,
        "base_block_version": loaded.version,
        "base_block_label": loaded.label,
        "base_block_reviewed": loaded.reviewed,
        "catalog_pin_agrees": True,
        "team_added_features": block["team_added_features"],
        "derived_semantics": block["derived_semantics"],
        "expert_approval": block["expert_approval"],
        "reference_ease_preset": loaded.reference_ease_preset,
        "reference_measurements_mm": loaded.reference_measurements,
        "provenance_document": "assets/base-blocks/provenance.md",
    }


# ---- trials ----------------------------------------------------------------


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


def _round(value, places: int):
    return "" if value is None else round(value, places)


def trial_row(record: dict, frozen: dict, commit: str) -> dict:
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
        "solver_status": record.get("solver_status", ""),
        "iterations": record.get("iterations", ""),
        "bound_violation": record.get("bound_violation", ""),
        "base_block_version": frozen["base_block_version"],
        "source_sha256": frozen["source_sha256"],
        "commit_hash": commit,
        "python_version": record.get("python_version", ""),
        "platform": record.get("platform", ""),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


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
            "all_checks_passed": bool(checks) and not failures,
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

    cfg = load_config()
    trials_cfg = cfg["trials"]
    if args.smoke:
        selected = [c for c in cfg["cases"]
                    if c["case_id"] in trials_cfg["smoke"]["cases"]]
        warmups = trials_cfg["smoke"]["warmups"]
        memory_trials = trials_cfg["smoke"]["memory_trials"]
        time_trials = trials_cfg["smoke"]["time_trials"]
    else:
        selected = cfg["cases"]
        warmups = trials_cfg["warmups"]
        memory_trials = args.memory_trials or trials_cfg["memory_trials"]
        time_trials = args.time_trials or trials_cfg["time_trials"]

    outputs = cfg["outputs"]
    directory = ROOT / outputs["directory"]
    directory.mkdir(parents=True, exist_ok=True)

    frozen = verify_base_block(cfg)
    fingerprint = config_fingerprint()["combined"]
    commit = git_state()["commitHash"]
    version = configuration_version()

    (directory / outputs["environment_json"]).write_text(
        json.dumps(environment_record(), indent=2, default=str), encoding="utf-8")
    (directory / outputs["base_block_record_json"]).write_text(
        json.dumps(frozen, indent=2, default=str), encoding="utf-8")
    (directory / outputs["config_manifest_json"]).write_text(
        json.dumps({
            "design": DESIGN,
            "configurationVersion": version,
            "configurationHash": fingerprint,
            "configurationHashesPerFile": config_fingerprint()["perFile"],
            "transformationConfigVersion": yaml.safe_load(
                (ROOT / "backend/app/config/transformation.yaml").read_text(encoding="utf-8")
            )["version"],
            "baseBlock": frozen,
            "git": git_state(),
            "cases": selected,
            "trials": {"warmups": warmups, "memory_trials": memory_trials,
                       "time_trials": time_trials, "mode": "smoke" if args.smoke else "full"},
            "threading": cfg["threading"],
            "maintainabilityAllowList": [f["path"] for f in cfg["maintainability"]["files"]],
            "unmeasured": {
                "feature_completeness_macro_pct": "no frozen feature manifest approved",
                "pattern_dimension_mae_cm": "no independently approved reference pattern",
            },
        }, indent=2, default=str),
        encoding="utf-8",
    )

    rows: list[dict] = []
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
                row = trial_row(record, frozen, commit)
                rows.append(row)
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
    roles = {c["case_id"]: c["role"] for c in cfg["cases"]}
    write_csv(directory / outputs["case_summary_csv"], SUMMARY_FIELDS, summarise(rows, roles))
    with (directory / outputs["failures_jsonl"]).open("w", encoding="utf-8") as fh:
        for failure in failures:
            fh.write(json.dumps(failure, default=str) + "\n")

    print(f"\n{len(rows)} trials, {len(failures)} failures, "
          f"{time.perf_counter() - started:.1f}s. Results in {outputs['directory']}/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())