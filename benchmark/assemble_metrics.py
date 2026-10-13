"""Generate results/design_3_final_metrics.json from the raw result files.

This is the only thing that writes the final metrics file, and it computes every value
from the raw trial and maintainability rows. A null in the output means the value was not
measured; it is never a placeholder for a number that was inconvenient.

Metric 4 (pattern-dimension MAE) is null by decision: no independently approved reference
pattern exists in this repository for the garment Design 3 generates, and a reference
synthesised from a drafting oracle would not be an independent approval. The field is
left null rather than filled with a number the protocol does not support.

    python benchmark/assemble_metrics.py
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark"))

import yaml  # noqa: E402

from environment import config_fingerprint, configuration_version, git_state  # noqa: E402

DESIGN = "Design 3 - Anthropometric Surface Reconstruction and Constrained Mesh Flattening"


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _median(values: list[float]) -> float | None:
    return round(statistics.median(values), 9) if values else None


def _pooled(trials: list[dict], case_ids: set[str], pass_type: str, field: str) -> list[float]:
    """Every valid measured trial of one pass, pooled across the mandatory cases.

    Warm-ups are excluded: they exist to warm the caches and are not measurements.
    """
    return [
        float(row[field])
        for row in trials
        if row["case_id"] in case_ids
        and row["pass_type"] == pass_type
        and row["warmup"] == "False"
        and row["valid_output"] == "True"
        and row[field]
    ]


def assemble() -> dict:
    config = yaml.safe_load(
        (ROOT / "benchmark" / "config-design-3.yaml").read_text(encoding="utf-8"))
    outputs = config["outputs"]
    directory = ROOT / outputs["directory"]

    trials = _read_csv(directory / outputs["trials_csv"])
    maintainability = _read_csv(directory / outputs["maintainability_csv"])

    mandatory = {c["case_id"] for c in config["cases"] if c["role"] == "mandatory"}

    memory = _pooled(trials, mandatory, "memory", "peak_memory_mb")
    timing = _pooled(trials, mandatory, "time", "time_s")

    sloc_total = sum(int(row["sloc"]) for row in maintainability)
    mi = (
        sum(float(row["maintainability_index"]) * int(row["sloc"]) for row in maintainability)
        / sloc_total
        if sloc_total else None
    )

    expected_memory = config["trials"]["memory_trials"]
    expected_time = config["trials"]["time_trials"]
    complete = {
        case_id: (
            len(_pooled(trials, {case_id}, "memory", "peak_memory_mb")) >= expected_memory
            and len(_pooled(trials, {case_id}, "time", "time_s")) >= expected_time
        )
        for case_id in mandatory
    }
    per_case_failures = {
        case["case_id"]: sum(
            1 for row in trials
            if row["case_id"] == case["case_id"] and row["warmup"] == "False"
            and row["valid_output"] != "True"
        )
        for case in config["cases"]
    }

    fingerprint = config_fingerprint()["combined"]
    return {
        "design": DESIGN,
        # Metric 1 - Python-tracked peak memory, median across mandatory measured trials.
        "peak_memory_mb_median": _median(memory),
        # Metric 2 - not measured: no frozen feature manifest was approved for this run.
        "feature_completeness_macro_pct": None,
        "feature_completeness_micro_pct": None,
        # Metric 3 - generation time, median across mandatory measured trials.
        "generation_time_s_median": _median(timing),
        # Metric 4 - not measured: no independently approved reference pattern exists.
        "pattern_dimension_mae_cm": None,
        # Metric 5 - source-line-weighted Radon MI on the 0-100 scale.
        "maintainability_index_0_100": round(mi, 6) if mi is not None else None,
        "mandatory_cases": len(mandatory),
        "measured_memory_trials_per_case": len(memory),
        "measured_time_trials_per_case": len(timing),
        "all_cases_valid": all(complete.values()) and not any(per_case_failures.values()),
        "configuration_version": configuration_version(),
        "configuration_hash": fingerprint,
        "commit_hash": git_state()["commitHash"],
        "supporting": {
            "memoryTrialsPooled": len(memory),
            "timeTrialsPooled": len(timing),
            "maintainabilityFiles": len(maintainability),
            "maintainabilitySloc": sloc_total,
            "mandatoryCases": sorted(mandatory),
            "everyMandatoryCaseFullyMeasured": complete,
            "measuredTrialFailuresPerCase": per_case_failures,
            "unmeasured": ["feature_completeness_macro_pct", "pattern_dimension_mae_cm"],
            "memoryMetricScope": "Python-tracked allocations only; compiled-library "
                                 "allocations are outside tracemalloc and are not included.",
        },
    }


def main() -> int:
    metrics = assemble()
    config = yaml.safe_load(
        (ROOT / "benchmark" / "config-design-3.yaml").read_text(encoding="utf-8"))
    target = ROOT / config["outputs"]["directory"] / config["outputs"]["final_metrics_json"]
    target.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if k != "supporting"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())