# Coding-Agent Instructions — Design 2

## GarmageSet-Based Parametric Base-Block Transformation

## Mission

Instrument and evaluate the existing **Design 2** program, which transforms one GarmageSet-derived dress base block according to a confirmed body-measurement profile. Obtain defensible values for:

1. Python-tracked peak memory consumption in MB — minimize.
2. Required-feature completeness rate in percent — maximize.
3. Median pattern-generation time in seconds — minimize.
4. Pattern-dimension mean absolute error in cm — minimize.
5. Radon Maintainability Index on a 0–100 scale — maximize.

All values must be generated from the program. Do not insert published results, estimates, or values from Designs 1 or 3.

## Correct Data Roles

- **GarmageSet JSON:** source geometry for the one Design 2 base block.
- **Adapted GarmentCode-style YAML:** interchange structure for camera-estimated and tailor-confirmed body measurements.
- **Project reference patterns:** source of expert-approved target dimensions for MAE.

Do not state or encode that GarmentCode supplied the Design 2 pattern. Do not treat the adapted YAML as training data. The engine must consume only the confirmed measurement values.

## Design Boundary

Design 2 must always transform one fixed, versioned GarmageSet-derived base block. It must not:

- Search among multiple pattern templates.
- Reconstruct the whole pattern from drafting equations.
- Switch to Design 1 or Design 3 on an unsupported case.
- Mutate the original GarmageSet source file.

Use the shared interface:

```python
def generate_pattern(measurements: dict, garment: dict) -> dict:
    """Transform the fixed base block and return the common pattern object in mm."""
```

## First Actions

1. Inspect the existing Design 2 repository and identify its current entry point.
2. Create `benchmark/design-2` and record the starting commit.
3. Locate the exact GarmageSet JSON source and calculate its SHA-256 hash.
4. Preserve that source as read-only.
5. Identify the current importer, internal schema, reference profile, transformation rules, and DXF path.
6. Confirm whether the source JSON's unit and curve semantics are documented.
7. Freeze the successful environment as `requirements-design-2.lock`.
8. Record Python, OS, CPU, RAM, source-file ID, source hash, importer version, and base-block version.

## Required Libraries

Use the current working program dependencies. Install missing benchmark tools in an isolated environment:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install pytest radon==6.0.1 pyyaml pandas
```

Retain existing NumPy, SciPy, Shapely, and ezdxf versions if the program uses them. Do not upgrade geometry libraries after viewing benchmark results.

## Required Repository Additions

```text
design_2/
  assets/
    original_garmageset/
    base_block/
      base_block.json
      base_profile.yaml
      provenance.json
  importer/
  transformation/
benchmark/
  config-design-2.yaml
  run_design_2.py
  feature_check.py
  dimension_check.py
  maintainability_check.py
  environment.py
fixtures/
  measurements/
  garments/
  references/
  feature_manifest.csv
  dimension_reference.csv
results/
  design_2_trials.csv
  design_2_case_summary.csv
  design_2_features.csv
  design_2_dimensions.csv
  design_2_maintainability.csv
  design_2_final_metrics.json
  design_2_environment.json
  design_2_failures.jsonl
  design_2_preprocessing_report.json
```

Shared fixtures and checkers must be identical to those used for Designs 1 and 3.

## Preprocessing the GarmageSet JSON

Preprocessing may map and validate the source, but it must not silently improve the geometry.

### Required Pipeline

```text
Original GarmageSet JSON
  -> JSON syntax and required-field validation
  -> dress-category confirmation
  -> source-schema to project-schema mapping
  -> unit verification and conversion to mm
  -> coordinate-axis and origin normalization
  -> ordered reconstruction of panels, vertices, edges, and curves
  -> stitch-reference reconstruction
  -> topology and geometry validation
  -> feature audit
  -> documented base-block preparation
  -> expert review
  -> versioned base_block.json and base_profile.yaml
```

### Provenance Record

`provenance.json` must contain:

```json
{
  "source_dataset": "GarmageSet",
  "source_file": null,
  "source_sha256": null,
  "source_unit": null,
  "internal_unit": "mm",
  "unit_scale_factor": null,
  "axis_transform": null,
  "origin_transform": null,
  "importer_commit": null,
  "base_block_version": null,
  "team_added_features": [],
  "expert_approval": null
}
```

Never label a team-added grainline, notch, seam allowance, label, fold marker, or reference profile as original GarmageSet content.

### Import Validation

Reject the base-block import if any of the following remain unresolved:

- Source unit cannot be verified.
- Coordinates are non-finite.
- Boundary edge order cannot be reconstructed.
- A required panel is missing.
- A panel loop is open or self-intersecting.
- An edge has zero length.
- Stitch references point to absent panels or edges.
- Curve conversion exceeds the approved geometric tolerance.

The base block must be frozen before benchmark results are collected. Any later geometry or feature modification requires a new version and a complete rerun of all Design 2 trials.

## Base Reference Profile

Design 2 needs a body or garment profile corresponding to the fixed base block. If GarmageSet does not provide it, derive it through documented seam-line measurements and expert interpretation. Mark it `team_derived: true`.

`base_profile.yaml` must include:

- Unit.
- Bust, waist, hip, bodice lengths, shoulder, armscye, dress length, and sleeve dimensions required by the transform.
- Whether each value is dataset-provided, pattern-derived, or expert-assigned.
- Extraction path or landmark definition.
- Approver and date.

Do not use undocumented assumptions to fill missing values.

## Frozen Inputs and References

Use the same confirmed measurement YAML and garment fixtures, feature manifest, and dimension reference CSV as Designs 1 and 3. Design 2 reads only `confirmed_value` from the adapted GarmentCode-style YAML.

`feature_manifest.csv`:

```text
case_id,feature_id,feature_type,applicable,validation_rule,approved_by,approval_date
```

`dimension_reference.csv`:

```text
case_id,dimension_id,panel_id,path_or_landmarks,reference_mm,tolerance_mm,approved_by,approval_date
```

## Common Validation Gate

Every generated pattern must pass the same common validator:

- Millimetre output unit.
- Required panels and construction features.
- Finite coordinates and positive areas.
- Closed, sequential boundaries.
- No unintended self-intersections.
- Valid edge, annotation, and stitch references.
- No zero-length edges.
- No transformation-bound violation.
- DXF opens successfully in the selected validation application.

Return a structured `OUT_OF_DOMAIN`, `BOUND_VIOLATION`, `INVALID_GEOMETRY`, or other declared error instead of forcing unsupported geometry. Retain all failures in the log.

## Metric 1 — Peak Memory

Measure the initialized base block retained in Python memory plus transformation, validation, internal-object creation, and DXF output.

Run every trial in a fresh child process. Import common packages before tracing, then:

```python
import gc
import tracemalloc

gc.collect()
tracemalloc.start()
engine = build_design_2_engine(
    base_block_path="design_2/assets/base_block/base_block.json",
    profile_path="design_2/assets/base_block/base_profile.yaml",
)
tracemalloc.reset_peak()
pattern = engine.generate_pattern(measurements, garment)
validated = validate_pattern(pattern)
dxf_path = export_common_dxf(pattern, temporary_output)
current_bytes, peak_bytes = tracemalloc.get_traced_memory()
tracemalloc.stop()
peak_memory_mb = peak_bytes / (1024 ** 2)
```

Starting tracing before engine construction includes the retained Python allocation of the parsed GarmageSet-derived block. Resetting the peak after construction removes temporary parsing spikes while preserving current retained allocations. Use the exact same rule for Designs 1 and 3.

Run five untimed warm-ups and at least 30 measured memory trials per case. The trade-off value is the median across all mandatory predefined trials; also report case medians and interquartile ranges.

## Metric 2 — Required-Feature Completeness

For each case:

```text
completeness_pct = 100 * complete_applicable_items / applicable_items
```

A feature is complete only when present, properly referenced, and valid. Imported features and team-added base-block features are evaluated by the same rule. Do not award a point merely because a field name exists.

Write:

```text
case_id,feature_id,applicable,present,valid,complete,source_or_generated,failure_reason
```

Report macro and micro completeness. Use the arithmetic mean of case percentages as the Chapter 3 trade-off value.

## Metric 3 — Pattern-Generation Time

The recurring generation boundary starts after the engine has loaded the frozen base block and profile. It includes:

- Measurement-difference calculation.
- Landmark influence mapping.
- Constraint or bounded-displacement solution.
- Dependent line and curve rebuilding.
- Annotation and allowance updates.
- Common validation.
- Internal pattern construction.
- DXF creation.

It excludes imports, source JSON preprocessing, base-block initialization, fixture loading, camera operations, UI rendering, manual review, and A0 nesting.

```python
from time import perf_counter_ns

engine = build_design_2_engine(base_block_path, profile_path)
start_ns = perf_counter_ns()
pattern = engine.generate_pattern(measurements, garment)
validated = validate_pattern(pattern)
dxf_path = export_common_dxf(pattern, temporary_output)
end_ns = perf_counter_ns()
time_s = (end_ns - start_ns) / 1_000_000_000
```

Run separate passes for memory and time; `tracemalloc` must be disabled in the time pass. Conduct five warm-ups and at least 30 measured time trials per case. The trade-off value is the median across mandatory cases.

For every trial also record solver status, iterations, convergence tolerance, and bound violations. Fix solver, tolerance, initial state, maximum iterations, and random seed before testing.

## Metric 4 — Pattern-Dimension MAE

Extract generated seam-line dimensions using the same paths and landmarks as the approved references:

```text
absolute_error_mm = abs(generated_mm - reference_mm)
overall_mae_cm = mean(all applicable absolute_error_mm) / 10
```

Write:

```text
case_id,dimension_id,reference_mm,generated_mm,absolute_error_mm,within_tolerance,extraction_status,transformation_status
```

Report per-dimension, per-case, and overall MAE plus percentage within tolerance. Overall MAE in cm is the trade-off value.

Never assign zero to missing geometry. If a mandatory case is outside Design 2's validated bounds, retain the structured failure. The design is not eligible for direct final scoring unless all three alternatives are evaluated on a common supported suite or the report explicitly applies a predeclared failure policy.

## Metric 5 — Maintainability Index

Create a fixed allow-list containing only unique Design 2 production modules, such as:

```yaml
maintainability_files:
  - design_2/engine.py
  - design_2/importer/garmageset_json.py
  - design_2/transformation/landmark_map.py
  - design_2/transformation/constraints.py
  - design_2/transformation/curve_rebuild.py
  - design_2/annotations.py
```

Include the importer only if it is required production code for creating or loading the run-time base block. If preprocessing is a one-time research utility and is not shipped with the engine, exclude it and explain that boundary. Apply the equivalent rule to data-preparation utilities for the other designs.

Run Radon 6.0.1:

```bash
radon mi -s <allow-listed files>
radon raw -s <allow-listed files>
radon cc -s -a <allow-listed files>
```

Compute:

```text
weighted_mi = sum(file_mi * file_sloc) / sum(file_sloc)
```

Report the 0–100 weighted MI without dividing by 100. Save path, MI, SLOC, complexity summary, inclusion rationale, and commit hash.

## Trial CSV

`design_2_trials.csv` columns:

```text
design,case_id,pass_type,trial,warmup,time_s,peak_memory_mb,valid_output,error_code,solver_status,iterations,bound_violation,base_block_version,source_sha256,commit_hash,python_version,platform,timestamp_utc
```

Leave unmeasured fields blank, not zero.

## Final Metrics JSON

```json
{
  "design": "Design 2 - GarmageSet-Based Parametric Base-Block Transformation",
  "peak_memory_mb_median": null,
  "feature_completeness_macro_pct": null,
  "feature_completeness_micro_pct": null,
  "generation_time_s_median": null,
  "pattern_dimension_mae_cm": null,
  "maintainability_index_0_100": null,
  "mandatory_cases": null,
  "measured_trials_per_case": 30,
  "all_cases_valid": null,
  "garmageset_source_sha256": null,
  "base_block_version": null,
  "commit_hash": null
}
```

Generate this file from raw CSV outputs. Never hand-enter values.

## Acceptance Checklist

- [ ] Exact GarmageSet source JSON and hash recorded.
- [ ] Original source remains unchanged.
- [ ] Import transformations and team-added features documented.
- [ ] Base reference profile provenance recorded.
- [ ] Design 2 transforms exactly one fixed base block.
- [ ] Same fixtures, validator, feature checker, dimensional references, and exporter as Designs 1 and 3.
- [ ] At least 30 measured memory and 30 measured time trials per case.
- [ ] Separate tracing and timing passes.
- [ ] Failures and out-of-domain cases retained.
- [ ] MI scope explicitly allow-listed.
- [ ] Final metrics generated from raw outputs without manual replacement.
