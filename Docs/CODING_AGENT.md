# Vision2DXF Skeletal Design-Comparison Prototype

## Role

Build a small, reproducible web application that demonstrates three distinct pattern-generation designs and produces preliminary measured values for a later sensitivity analysis.

This is an experimental prototype, not the final Vision2DXF system. Prioritize working skeletal functionality, fair comparison, traceable outputs, and repeatable evaluation. Do not spend time on visual polish, complete garment drafting, 3D capture, automated fitting, or production deployment.

## Immediate Goal

Within a few hours, deliver a local Streamlit application that can:

1. Load or accept one body-measurement profile.
2. Run any of the three pattern-generation designs.
3. Display a simple 2D front-bodice preview.
4. Apply the same geometric validation rules to all designs.
5. Export the result as JSON and, if time permits, DXF.
6. Run a controlled evaluation of all three designs.
7. Collect preliminary values for Economic, Performance, Reliability, and Maintainability.
8. Leave Usability pending until actual SUS responses are collected.
9. Normalize the five constraint values and run sensitivity analysis only when all five are available.

## Designs

Implement three genuinely different strategies.

### Design A — Formula-Based Drafting

Construct a simplified front-bodice pattern directly from body measurements and explicit equations. Calculate named landmarks for the center front, neckline, shoulder, armhole, side seam, waistline, and one dart.

Design A must not load the base block or size templates used by the other designs.

### Design B — Parametric Base-Block Transformation

Load one predefined medium base block with reference measurements and named landmarks. Compare the input measurements with its reference profile, move the affected landmarks, and reconstruct connected lines or curves.

Design B must always transform this one master block. It must not select among multiple templates.

### Design C — Template Retrieval and Bounded Grading

Load three discrete templates: Small, Medium, and Large. Select the closest template using normalized measurement distance, then apply bounded grading adjustments to its landmarks.

Design C must record the selected template, distance, requested adjustment, applied adjustment, and whether any grading limit was reached.

## Fair-Comparison Rules

- Use the same measurement profiles for all designs.
- Use the same units, preferably centimeters.
- Use the same input model and output schema.
- Use the same validator, preview, and exporter.
- Time only the pattern-generation call.
- Keep shared code outside the design packages.
- Analyze only design-specific source code for Program Volume and Maintainability Index.
- Do not optimize one design more extensively than the others.
- Do not invent measurements, SUS responses, benchmark results, or validation outcomes.
- Clearly label unverified pattern data as `prototype_only`.

## Technology

Use a minimal Python stack:

- Python 3.10+
- Streamlit
- pandas
- NumPy
- pytest
- Radon
- Plotly or SVG for 2D preview
- ezdxf if DXF export can be completed within the time limit
- PyYAML for criterion configuration

## Repository Structure

```text
vision2dxf-comparison/
├── app.py
├── README.md
├── requirements.txt
├── pyproject.toml
├── config/
│   ├── criteria.yaml
│   ├── evaluation.yaml
│   └── test_profiles.json
├── data/
│   ├── README.md
│   ├── drafting/
│   │   └── formula_parameters.yaml
│   ├── base_blocks/
│   │   └── bodice_medium.json
│   ├── templates/
│   │   ├── small.json
│   │   ├── medium.json
│   │   ├── large.json
│   │   └── template_index.json
│   └── usability/
│       └── sus_responses.csv
├── src/vision2dxf/
│   ├── core/
│   │   ├── models.py
│   │   ├── interfaces.py
│   │   ├── validation.py
│   │   ├── preview.py
│   │   └── dxf_export.py
│   ├── designs/
│   │   ├── formula_based/
│   │   │   ├── generator.py
│   │   │   └── rules.py
│   │   ├── parametric_block/
│   │   │   ├── generator.py
│   │   │   └── transformer.py
│   │   └── template_grading/
│   │       ├── generator.py
│   │       ├── selector.py
│   │       └── grader.py
│   ├── evaluation/
│   │   ├── benchmark.py
│   │   ├── reliability.py
│   │   ├── code_metrics.py
│   │   └── pipeline.py
│   └── sensitivity/
│       ├── normalize.py
│       ├── weights.py
│       ├── score.py
│       └── summarize.py
├── scripts/
│   ├── run_evaluation.py
│   ├── run_sensitivity.py
│   └── run_all.py
├── tests/
│   ├── test_contract.py
│   ├── test_designs.py
│   ├── test_validation.py
│   └── test_sensitivity.py
└── outputs/
    ├── raw/
    ├── processed/
    ├── sensitivity/
    └── patterns/
```

## Shared Models

Implement one input model:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class MeasurementProfile:
    profile_id: str
    chest: float
    waist: float
    hip: float
    shoulder: float
    torso_length: float
    armhole_depth: float
    sleeve_length: float
    ease: float
```

Implement one result model:

```python
@dataclass
class PatternResult:
    design_id: str
    points: dict[str, tuple[float, float]]
    segments: list[dict]
    valid: bool
    validation_errors: list[str]
    metadata: dict
```

All generators must satisfy:

```python
from typing import Protocol

class PatternGenerator(Protocol):
    design_id: str

    def generate(self, measurements: MeasurementProfile) -> PatternResult:
        ...
```

## Data to Use Now

Do not download a large training dataset. These designs are deterministic and do not require model training.

Create a compact, documented prototype dataset:

1. `test_profiles.json`: 10–20 body-measurement profiles used identically by all designs.
2. `bodice_medium.json`: one simplified medium base block for Design B.
3. `small.json`, `medium.json`, and `large.json`: three discrete templates for Design C.
4. `formula_parameters.yaml`: Design A drafting constants and ease settings.

Use one of these approaches, in order of preference:

1. Encode measurements and pattern geometry supplied or reviewed by the tailoring client.
2. Adapt example body presets or permitted pattern structures from GarmentCode: https://github.com/maria-korosteleva/GarmentCode
3. Use a small BodyM measurement subset only if its fields and license fit the project: https://registry.opendata.aws/bodym/

Do not silently estimate missing anthropometric fields. If a required value is derived, record the formula, source field, and status as `derived`. If a value is only a placeholder for software testing, mark it `prototype_only` and do not present it as empirical anthropometric data.

Each data file must include or be accompanied by:

- Source or creator
- License or use condition
- Units
- Field definitions
- Any conversion or derivation
- Validation status
- Date acquired or created

## Initial Data Format

Example measurement profile structure:

```json
{
  "profile_id": "P001",
  "units": "cm",
  "chest": 92.0,
  "waist": 76.0,
  "hip": 98.0,
  "shoulder": 40.0,
  "torso_length": 43.0,
  "armhole_depth": 22.0,
  "sleeve_length": 59.0,
  "ease": 4.0,
  "source": "prototype",
  "status": "prototype_only"
}
```

The numbers above demonstrate the schema only. Do not treat them as validated measurements.

Pattern files should contain:

```json
{
  "pattern_id": "medium",
  "units": "cm",
  "reference_measurements": {},
  "points": {},
  "segments": [],
  "grading_limits": {},
  "source": "team_created",
  "status": "prototype_only"
}
```

## Shared Validation

A generation run passes only if:

- All required landmarks exist.
- Every coordinate is finite.
- No required segment has zero length.
- Width and length are positive.
- The boundary is closed.
- Coordinates remain within configured bounds.
- DXF serialization succeeds, when DXF is enabled.

Return specific errors. Do not return only `False`.

## Web Application

Create four simple tabs.

### Generate

- Load a preset or enter measurements.
- Select Design A, B, or C.
- Generate the pattern.
- Display 2D geometry, validation status, metadata, and generation time.
- Download JSON and optionally DXF.

### Compare

- Run one measurement profile through all designs.
- Display the three previews side by side.
- Show generation time, validation status, and design-specific metadata.

### Evaluate

- Run fixed profiles through every design.
- Show raw constraint values and detailed execution records.
- Export CSV files.
- Show Usability as `pending` until real SUS data exists.

### Sensitivity

- Load the completed design-constraint matrix.
- Show normalized ratings.
- Run all Level-of-Importance permutations.
- Display win counts, mean rank, score range, and ties.
- Block the final recommendation when required values are missing.

## Constraint Definitions

Configure the criteria in `config/criteria.yaml`:

```yaml
criteria:
  economic:
    metric: halstead_program_volume
    unit: bits
    direction: minimize
    source: radon

  usability:
    metric: mean_sus_score
    unit: score_0_100
    direction: maximize
    source: participant

  performance:
    metric: median_generation_time_ms
    unit: milliseconds
    direction: minimize
    source: benchmark

  reliability:
    metric: failure_rate_percent
    unit: percent
    direction: minimize
    source: test_suite

  maintainability:
    metric: maintainability_index
    unit: score_0_100
    direction: maximize
    source: radon
```

## Evaluation Procedure

### Economic

Run Radon on each design-specific implementation directory. Use Halstead Program Volume as the raw value. Exclude shared core, UI, evaluation, sensitivity, test, preview, and export modules.

### Performance

- Use `time.perf_counter_ns()`.
- Time only `generator.generate(profile)`.
- Run one untimed warm-up.
- Run at least 30 measured repetitions per profile and design.
- Save every observation.
- Use median generation time as the primary value.

### Reliability

A measured run fails if generation raises an exception, shared geometric validation fails, or enabled DXF serialization fails.

Calculate:

```text
failure rate = failed runs / total runs * 100
```

Keep invalid-input tests separate from valid-profile reliability evaluation.

### Maintainability

Use Radon Maintainability Index on the same design-specific source boundaries used for Program Volume. Record cyclomatic complexity as supporting information only.

### Usability

Do not fabricate SUS scores. Add a CSV schema and optional web form for the standard ten SUS responses. Calculate participant scores in code and then calculate the mean for each design. Keep this criterion `pending` until actual participants complete equivalent tasks for all three designs.

## Raw Output Files

Generate:

```text
outputs/raw/
├── raw_constraint_values.csv
├── constraint_matrix.csv
├── performance_runs.csv
├── reliability_runs.csv
├── code_metrics.csv
├── sus_responses.csv
└── evaluation_metadata.json
```

`evaluation_metadata.json` must record:

- Run ID
- Date and time
- Git commit, if available
- Python version
- Package versions
- Operating system
- Processor description
- Warm-up count
- Repetition count
- Test-profile file
- Analyzed design paths
- SUS status

## Sensitivity Analysis

Reproduce the supplied notebook's method in reusable Python modules rather than leaving the logic only in a notebook.

### Criterion Order

Use this fixed order:

```python
CRITERIA_ORDER = [
    "economic",
    "usability",
    "performance",
    "reliability",
    "maintainability",
]
```

### Normalization

Convert every criterion to a 1–10 scale where 10 is always favorable.

For minimized criteria:

```text
rating = 1 + 9 * (maximum - value) / (maximum - minimum)
```

For maximized criteria:

```text
rating = 1 + 9 * (value - minimum) / (maximum - minimum)
```

If all three raw values are equal, assign all three a rating of 10 and mark the criterion as non-discriminating. Never divide by zero.

### LOI Scenarios

Generate all permutations of:

```python
[10, 9, 8, 7, 6]
```

With five criteria, this must produce exactly 120 unique scenarios. Convert every LOI tuple to proportional weights:

```text
weight = LOI / sum of all LOIs
```

### Weighted Score

For every design and scenario:

```text
score = sum(normalized criterion rating * proportional weight)
```

Save all scenarios in one tidy CSV. Include criterion weights, all design scores, ranks, winner, winning margin, and tie status.

Use a numerical tie tolerance, such as `1e-9`. Do not select the first column arbitrarily when scores tie.

### Recommendation

Generate the final recommendation only if all five verified criterion values are present for all designs. Report:

- Outright wins
- Tied wins
- Win percentage
- Mean score
- Score standard deviation
- Mean rank
- Worst rank
- Mean margin

Use win frequency as the primary robustness indicator. If results are close, state that the selected design depends on criterion priorities rather than claiming an unconditional winner.

## Sensitivity Outputs

```text
outputs/processed/
├── normalized_matrix.csv
└── validation_report.csv

outputs/sensitivity/
├── weight_scenarios.csv
├── scenario_results.csv
├── design_robustness.csv
└── recommendation.json
```

## Tests

At minimum, test that:

- All generators satisfy the shared contract.
- All generators accept the same measurement object.
- All generators return the same output schema.
- Every required landmark is present.
- Minimized criteria normalize in the correct direction.
- Maximized criteria normalize in the correct direction.
- Equal raw values do not cause division by zero.
- Exactly 120 unique LOI scenarios are generated.
- Proportional weights sum to 1.
- Weighted scores remain from 1 to 10.
- Missing SUS data blocks the final five-criterion recommendation.
- Ties are handled explicitly.
- Repeated sensitivity runs with the same matrix are deterministic.

## Commands

The repository must support:

```bash
pip install -r requirements.txt
streamlit run app.py
pytest
python scripts/run_evaluation.py
python scripts/run_sensitivity.py
python scripts/run_all.py
```

`run_all.py` must run tests, evaluation, matrix validation, normalization, the 120 LOI scenarios, robustness summarization, and recommendation generation. It must stop before final sensitivity ranking if a required criterion remains missing.

## Work Order

Implement in this order:

1. Create the repository and dependency files.
2. Implement shared models and generator interface.
3. Add the compact prototype data files and document their status.
4. Implement Design A.
5. Implement shared preview and validation.
6. Implement Design B.
7. Implement Design C.
8. Build the Generate and Compare tabs.
9. Add tests for the design contract and validator.
10. Implement performance and reliability collection.
11. Add Radon code-metric collection.
12. Implement normalization and all 120 LOI scenarios.
13. Add Evaluate and Sensitivity tabs.
14. Add JSON export, then DXF export if time remains.
15. Write exact setup, run, and evaluation instructions in the README.

## Stop Conditions

If development time expires, preserve this minimum vertical slice:

- All three designs run using the same measurement profile.
- Each returns a visibly distinct simplified bodice pattern.
- Shared validation works.
- Performance, failure rate, Program Volume, and Maintainability Index are exported.
- SUS is visibly marked pending.
- Normalization and LOI analysis are tested with a complete fixture matrix, but that fixture must be labeled `test_fixture` and must never be reported as measured project results.

## Definition of Done

The current prototype is complete when:

- The app launches locally with one documented command.
- All three designs generate a 2D pattern through one interface.
- Their algorithms are structurally distinct.
- Shared code is not duplicated inside design packages.
- Preliminary automated metrics come from actual executions and code analysis.
- Every data value has source and status metadata.
- The sensitivity module reproduces the notebook's 1–10 normalization and 120 LOI permutations.
- Missing real SUS data prevents a final five-criterion winner.
- Outputs are reproducible and auditable.
- No measured value, dataset field, participant response, or validation result is fabricated.
