"""Generate results/design_2_final_metrics.json from the raw result files.

This is the only thing that writes the final metrics file, and it computes every value
from the raw trial and maintainability rows. A null in the output means the value was not
measured; it is never a placeholder for a number that was inconvenient.

Metrics 2 and 4 are null by decision, and for the same reason Design 3 leaves them null:

* Feature completeness needs a frozen, approved `feature_manifest.csv`. None exists for
  Design 2, Design 1 or Design 3, so the protocol's "identical fixtures across designs"
  rule cannot be satisfied and a manifest authored now would be a measurement written to
  fit the output.
* Pattern-dimension MAE needs reference seam-line dimensions from independently approved
  patterns. The only reference in this repository, `fixtures/garmageset-reference.json`,
  is a 19-panel skirt-and-dress set, not a bodice block, and a reference derived from the
  engine's own influence map would be the engine grading itself.

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

from environment import (  # noqa: E402
    config,
    config_fingerprint,
    configuration_version,
    git_state,
)

DESIGN = "Design 2 - GarmageSet-Based Parametric Base-Block Transformation"


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _median(values: list[float]) -> float | None:
    return round(statistics.median(values), 9) if values else None


def _pooled(trials: list[dict], case_ids: set[str], pass_type: str, field: str) -> list[float]:
    """Every valid measured trial of one pass, pooled across the given cases.

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
    cfg = config()
    outputs = cfg["outputs"]
    directory = ROOT / outputs["directory"]

    trials = _read_csv(directory / outputs["trials_csv"])
    maintainability = _read_csv(directory / outputs["maintainability_csv"])
    base_block = _read_json(directory / outputs["base_block_record_json"])

    mandatory = {c["case_id"] for c in cfg["cases"] if c["role"] == "mandatory"}

    memory = _pooled(trials, mandatory, "memory", "peak_memory_mb")
    timing = _pooled(trials, mandatory, "time", "time_s")

    sloc_total = sum(int(row["sloc"]) for row in maintainability)
    mi = (
        sum(float(row["maintainability_index"]) * int(row["sloc"]) for row in maintainability)
        / sloc_total
        if sloc_total else None
    )

    expected_memory = cfg["trials"]["memory_trials"]
    expected_time = cfg["trials"]["time_trials"]
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
        for case in cfg["cases"]
    }
    bound_violations = sum(
        1 for row in trials
        if row.get("bound_violation") == "True" and row["warmup"] == "False"
    )

    # A control case exists to be refused, so counting its refusals as invalid output
    # would make `all_cases_valid` permanently false and say nothing about the design.
    # It is judged the other way round: it passed if it refused, with a structured code.
    control = {c["case_id"] for c in cfg["cases"] if c["role"] == "control"}
    control_rows = [row for row in trials
                    if row["case_id"] in control and row["warmup"] == "False"]
    control_codes = sorted({row["error_code"] for row in control_rows
                            if row["error_code"]})
    control_refused_as_expected = bool(control_rows) and all(
        row["valid_output"] != "True" for row in control_rows
    )

    return {
        "design": DESIGN,
        # Metric 1 - Python-tracked peak memory, median across mandatory measured trials.
        "peak_memory_mb_median": _median(memory),
        # Metric 2 - not measured: no frozen feature manifest has been approved.
        "feature_completeness_macro_pct": None,
        "feature_completeness_micro_pct": None,
        # Metric 3 - generation time, median across mandatory measured trials.
        "generation_time_s_median": _median(timing),
        # Metric 4 - not measured: no independently approved reference pattern exists.
        "pattern_dimension_mae_cm": None,
        # Metric 5 - source-line-weighted Radon MI on the 0-100 scale.
        "maintainability_index_0_100": round(mi, 6) if mi is not None else None,
        "mandatory_cases": len(mandatory),
        "control_cases": len(control),
        "measured_memory_trials_per_case": len(memory),
        "measured_time_trials_per_case": len(timing),
        # Judged over the mandatory cases only: a control case is supposed to be refused,
        # so counting it here would make this permanently false.
        "all_cases_valid": all(complete.values()) and not any(
            per_case_failures[case_id] for case_id in mandatory
        ),
        "control_cases_refused_as_expected": control_refused_as_expected,
        "garmageset_source_sha256": base_block.get("source_sha256"),
        "base_block_version": base_block.get("base_block_version"),
        "commit_hash": git_state()["commitHash"],
        "configuration_version": configuration_version(),
        "configuration_hash": config_fingerprint()["combined"],
        "supporting": {
            "memoryTrialsPooled": len(memory),
            "timeTrialsPooled": len(timing),
            "maintainabilityFiles": len(maintainability),
            "maintainabilitySloc": sloc_total,
            "mandatoryCases": sorted(mandatory),
            "everyMandatoryCaseFullyMeasured": complete,
            "measuredTrialFailuresPerCase": per_case_failures,
            "controlCases": sorted(control),
            "controlCasesRefusedAsExpected": control_refused_as_expected,
            "controlRefusalCodes": control_codes,
            "measuredBoundViolations": bound_violations,
            "unmeasured": ["feature_completeness_macro_pct", "feature_completeness_micro_pct",
                           "pattern_dimension_mae_cm"],
            "memoryMetricScope": "Python-tracked allocations only; compiled-library "
                                 "allocations are outside tracemalloc and are not included.",
            "timeMetricScope": "Measurement differences through DXF file write, starting "
                               "after the frozen base block has been loaded.",
        },
    }


def main() -> int:
    metrics = assemble()
    cfg = config()
    target = ROOT / cfg["outputs"]["directory"] / cfg["outputs"]["final_metrics_json"]
    target.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if k != "supporting"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())