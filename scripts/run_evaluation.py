#!/usr/bin/env python3
from vision2dxf.evaluation.code_metrics import measure_code_metrics


def main():
    design_paths = [
        "src/vision2dxf/designs/formula_based",
        "src/vision2dxf/designs/parametric_block",
        "src/vision2dxf/designs/template_grading",
    ]
    code_met = measure_code_metrics(design_paths)

    import json
    from pathlib import Path
    out = Path("outputs/raw")
    out.mkdir(parents=True, exist_ok=True)

    raw_values = {}
    for cm in code_met:
        if "error" in cm:
            continue
        path = cm["path"]
        if "formula_based" in path:
            did = "design_a_formula"
        elif "parametric_block" in path:
            did = "design_b_parametric"
        else:
            did = "design_c_template"
        raw_values[did] = {
            "halstead_program_volume": cm.get("halstead_volume", 0),
            "maintainability_index": cm.get("maintainability_index", 0),
        }

    import pandas as pd
    df = pd.DataFrame(raw_values).T
    df.index.name = "design_id"
    df.to_csv(out / "raw_constraint_values.csv")

    print("Evaluation complete.")
    for did, vals in raw_values.items():
        print(f"\n{did}:")
        for k, v in vals.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
