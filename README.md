# Vision2DXF — Skeletal Design-Comparison Prototype

A local Streamlit application that demonstrates three distinct garment pattern-generation strategies and produces preliminary measured values for sensitivity analysis.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## What This Does

Runs three structurally different front-bodice pattern generators through one shared interface, then evaluates and compares them:

| Design | Strategy |
|--------|----------|
| **A — Formula-Based** | Computes landmarks from body measurements using explicit equations |
| **B — Parametric Block** | Transforms one master medium base block by ratio |
| **C — Template Grading** | Selects closest of three discrete templates, applies bounded grading |

## Repository Layout

```
vision2dxf/
├── app.py                          # Streamlit app (4 tabs)
├── config/                         # Criteria, evaluation settings, test profiles
├── data/                           # Prototype data (all status: prototype_only)
│   ├── drafting/formula_parameters.yaml
│   ├── base_blocks/bodice_medium.json
│   ├── templates/{small,medium,large}.json
│   └── usability/sus_responses.csv
├── src/vision2dxf/
│   ├── core/                       # Shared models, validation, preview, export
│   ├── designs/                    # Three design packages (structurally distinct)
│   ├── evaluation/                 # Benchmark, reliability, code metrics
│   └── sensitivity/                # Normalize, weights, score, summarize
├── scripts/                        # CLI entry points
├── tests/                          # pytest suite
└── outputs/                        # Generated CSVs and JSONs
```

## Commands

```bash
# Run the app
streamlit run app.py

# Run tests
pytest -v

# Run full evaluation (benchmark + reliability + code metrics)
python scripts/run_evaluation.py

# Run sensitivity analysis (normalization + 120 LOI scenarios)
python scripts/run_sensitivity.py

# Run everything
python scripts/run_all.py
```

## Evaluation Metrics

| Criterion | Metric | Source | Direction |
|-----------|--------|--------|-----------|
| Economic | Halstead Program Volume | Radon | minimize |
| Performance | Median generation time (ms) | benchmark | minimize |
| Reliability | Failure rate (%) | test suite | minimize |
| Maintainability | Maintainability Index | Radon | maximize |
| Usability | Mean SUS score | participants | maximize (pending) |

## Sensitivity Analysis

- Normalizes all criteria to a 1–10 scale (10 = favorable)
- Generates all 120 permutations of LOI [10, 9, 8, 7, 6]
- Reports win counts, mean rank, score range, and ties
- Blocks final recommendation when SUS data is missing

## Data Status

All data files are `prototype_only` and not validated anthropometric data. Every value includes source, status, and units metadata.

## Definition of Done

- [x] App launches with one command
- [x] All three designs generate a 2D pattern through one interface
- [x] Algorithms are structurally distinct
- [x] Shared code not duplicated inside design packages
- [x] Preliminary automated metrics from actual executions
- [x] Sensitivity module reproduces 1–10 normalization and 120 LOI permutations
- [x] Missing SUS prevents final five-criterion winner
- [x] No fabricated measurements or validation results
