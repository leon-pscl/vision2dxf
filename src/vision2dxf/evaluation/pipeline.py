from __future__ import annotations
import json
from pathlib import Path
import platform
import sys
import statistics
from ..core.models import MeasurementProfile, PatternGenerator
from ..core.validation import validate_pattern
from .benchmark import benchmark_generator
from .reliability import measure_reliability
from .code_metrics import measure_code_metrics


def run_full_evaluation(
    generators: list[PatternGenerator],
    profiles: list[MeasurementProfile],
    design_paths: list[str],
    warmup: int = 1,
    repetitions: int = 30,
    output_dir: str = "outputs/raw",
) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_perf = []
    all_rel = []
    for gen in generators:
        all_perf.extend(benchmark_generator(gen, profiles, warmup, repetitions))
        all_rel.extend(measure_reliability(gen, profiles))

    code_metrics = measure_code_metrics(design_paths)

    # Build raw constraint values per design
    design_ids = [g.design_id for g in generators]
    cm_by_path = {c["path"]: c for c in code_metrics if "error" not in c}
    raw_values = {}
    for did, gen in zip(design_ids, generators):
        perf = [r for r in all_perf if r["design_id"] == did]
        rel = [r for r in all_rel if r["design_id"] == did]
        dpath = design_paths[generators.index(gen)]
        cm = cm_by_path.get(dpath, {})

        median_ms = statistics.median([p["median_ms"] for p in perf]) if perf else 0
        mean_failure = statistics.mean([r["failure_rate_percent"] for r in rel]) if rel else 0
        halstead_vol = cm.get("halstead_volume", 0) if cm else 0
        mi_score = cm.get("maintainability_index", 0) if cm else 0

        raw_values[did] = {
            "halstead_program_volume": halstead_vol,
            "median_generation_time_ms": median_ms,
            "failure_rate_percent": mean_failure,
            "maintainability_index": mi_score,
            "mean_sus_score": None,  # pending
        }

    # Save CSVs
    import pandas as pd
    df_perf = pd.DataFrame(all_perf)
    df_perf.to_csv(out / "performance_runs.csv", index=False)

    df_rel = pd.DataFrame([{k: v for k, v in r.items() if k != "runs"} for r in all_rel])
    df_rel.to_csv(out / "reliability_runs.csv", index=False)

    df_cm = pd.DataFrame(code_metrics)
    df_cm.to_csv(out / "code_metrics.csv", index=False)

    # Raw constraint matrix
    df_raw = pd.DataFrame(raw_values).T
    df_raw.index.name = "design_id"
    df_raw.to_csv(out / "raw_constraint_values.csv")

    # Constraint matrix (same as raw for now)
    df_raw.to_csv(out / "constraint_matrix.csv")

    # SUS placeholder
    df_sus = pd.DataFrame(columns=["profile_id", "design_id", "mean_sus_score"])
    df_sus.to_csv(out / "sus_responses.csv", index=False)

    metadata = {
        "run_id": "prototype_run",
        "python_version": sys.version,
        "platform": platform.platform(),
        "warmup_runs": warmup,
        "measured_runs": repetitions,
        "design_paths": design_paths,
        "sus_status": "pending",
    }
    (out / "evaluation_metadata.json").write_text(json.dumps(metadata, indent=2))

    return {
        "raw_values": raw_values,
        "performance": all_perf,
        "reliability": all_rel,
        "code_metrics": code_metrics,
    }
