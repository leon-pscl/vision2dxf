from __future__ import annotations
import json
from pathlib import Path
import platform
import sys
import statistics
from ..core.models import MeasurementProfile, PatternGenerator
from .code_metrics import measure_code_metrics


def run_full_evaluation(
    generators: list[PatternGenerator],
    profiles: list[MeasurementProfile],
    design_paths: list[str],
    output_dir: str = "outputs/raw",
) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    design_ids = [g.design_id for g in generators]
    code_metrics = measure_code_metrics(design_paths)

    cm_by_path = {c["path"]: c for c in code_metrics if "error" not in c}
    raw_values = {}
    for did, gen in zip(design_ids, generators):
        dpath = design_paths[generators.index(gen)]
        cm = cm_by_path.get(dpath, {})

        raw_values[did] = {
            "halstead_program_volume": cm.get("halstead_volume", 0) if cm else 0,
            "maintainability_index": cm.get("maintainability_index", 0) if cm else 0,
        }

    import pandas as pd
    df_raw = pd.DataFrame(raw_values).T
    df_raw.index.name = "design_id"
    df_raw.to_csv(out / "raw_constraint_values.csv")

    metadata = {
        "run_id": "prototype_run",
        "python_version": sys.version,
        "platform": platform.platform(),
        "design_paths": design_paths,
    }
    (out / "evaluation_metadata.json").write_text(json.dumps(metadata, indent=2))

    return {
        "raw_values": raw_values,
        "code_metrics": code_metrics,
    }
