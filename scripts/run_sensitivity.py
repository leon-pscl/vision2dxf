#!/usr/bin/env python3
import json
from pathlib import Path

import pandas as pd
import yaml
from vision2dxf.sensitivity.score import compute_scores
from vision2dxf.sensitivity.summarize import summarize_robustness, generate_recommendation
from vision2dxf.evaluation.pipeline import run_full_evaluation
from vision2dxf.core.interfaces import load_profiles
from vision2dxf.designs.formula_based.generator import FormulaBasedGenerator
from vision2dxf.designs.parametric_block.generator import ParametricBlockGenerator
from vision2dxf.designs.template_grading.generator import TemplateGradingGenerator


def main():
    raw_path = Path("outputs/raw/raw_constraint_values.csv")
    if not raw_path.exists():
        print("No raw constraint values found. Running evaluation first...")
        profiles = load_profiles("config/test_profiles.json")
        gens = [FormulaBasedGenerator(), ParametricBlockGenerator(), TemplateGradingGenerator()]
        design_paths = [
            "src/vision2dxf/designs/formula_based",
            "src/vision2dxf/designs/parametric_block",
            "src/vision2dxf/designs/template_grading",
        ]
        run_full_evaluation(gens, profiles, design_paths)

    Path("outputs/processed").mkdir(parents=True, exist_ok=True)
    Path("outputs/sensitivity").mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(raw_path, index_col=0)

    criteria_cfg = yaml.safe_load(open("config/criteria.yaml"))["criteria"]
    directions = {k: v["direction"] for k, v in criteria_cfg.items()}

    # Normalize
    norm_df, missing = compute_scores(raw_df, directions)

    # Save normalized matrix (scores and ranks)
    norm_df.to_csv("outputs/processed/normalized_matrix.csv")

    # Save scenario results
    norm_df.to_csv("outputs/sensitivity/scenario_results.csv", index=False)

    # Weight scenarios
    weight_cols = ["loi_combo"] + [c for c in norm_df.columns if c.startswith("weight_")]
    norm_df[weight_cols].to_csv("outputs/sensitivity/weight_scenarios.csv", index=False)

    # Robustness
    design_ids = [c.replace("score_", "") for c in norm_df.columns if c.startswith("score_")]
    robust = summarize_robustness(norm_df, design_ids)
    robust.to_csv("outputs/sensitivity/design_robustness.csv", index=False)

    # Recommendation
    rec = generate_recommendation(robust, raw_df.to_dict(orient="index"), list(raw_df.columns))
    if rec:
        rec["missing_criteria"] = missing
    if rec:
        Path("outputs/sensitivity/recommendation.json").write_text(json.dumps(rec, indent=2))
        print("Recommendation generated:", rec["recommended_design"])
    else:
        print("Recommendation blocked: missing required criterion values (e.g. SUS)")

    print("Sensitivity analysis complete.")


if __name__ == "__main__":
    main()
