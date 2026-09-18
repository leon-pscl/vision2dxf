#!/usr/bin/env python3
from vision2dxf.core.interfaces import load_profiles
from vision2dxf.designs.formula_based.generator import FormulaBasedGenerator
from vision2dxf.designs.parametric_block.generator import ParametricBlockGenerator
from vision2dxf.designs.template_grading.generator import TemplateGradingGenerator
from vision2dxf.evaluation.pipeline import run_full_evaluation


def main():
    profiles = load_profiles("config/test_profiles.json")
    generators = [FormulaBasedGenerator(), ParametricBlockGenerator(), TemplateGradingGenerator()]
    design_paths = [
        "src/vision2dxf/designs/formula_based",
        "src/vision2dxf/designs/parametric_block",
        "src/vision2dxf/designs/template_grading",
    ]
    result = run_full_evaluation(generators, profiles, design_paths)
    print("Evaluation complete.")
    print(f"Designs evaluated: {list(result['raw_values'].keys())}")
    for did, vals in result["raw_values"].items():
        print(f"\n{did}:")
        for k, v in vals.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
