# Design 3 benchmark — how to run it

Measures the Chapter 3 metrics for Design 3 (anthropometric surface reconstruction and
constrained mesh flattening) against the frozen configuration in this repository.

Three metrics are measured:

| Metric | What it is | Script |
|---|---|---|
| **1 — Peak memory** | Python-tracked peak memory in MB, median across mandatory measured trials | `run_design_3.py` |
| **3 — Generation time** | Median pattern-generation time in seconds, same trial population | `run_design_3.py` |
| **5 — Maintainability** | Radon 6.0.1 MI on the 0–100 scale, source-line weighted | `maintainability_check.py` |

Two are deliberately **null**, not missing — see [Why two metrics are null](#why-two-metrics-are-null).

---

## 1. Prerequisites

Run everything with an interpreter that has **`radon==6.0.1`**, plus numpy, shapely, ezdxf,
pydantic, PyYAML and reportlab. The exact interpreter that produced the committed results is
`requirements-design-3.lock`, frozen from `pip freeze`.

```powershell
cd C:\Users\Leon\Documents\Github\vision2dxf
python -c "import radon, numpy, shapely, ezdxf; print('ok')"
```

If that fails, install Radon only — it is the one library the backend itself does not need:

```powershell
python -m pip install radon==6.0.1
```

`run_design_3.py` and `maintainability_check.py` both need Radon. `environment.py` and the
backend test suite do not.

---

## 2. Smoke run first — always

Two minutes, one case. Proves the harness before you commit 80 minutes to it.

```powershell
cd C:\Users\Leon\Documents\Github\vision2dxf
python benchmark\run_design_3.py --smoke
```

Expect:

```text
[   1/8] d3-case-001-central memory trial 0 (warmup)  valid=True code=-   ~20s elapsed
...
8 trials, 0 failures, ~100.0s elapsed. Results in results/.
```

Then check it produced something sane:

```powershell
Get-Content results\design_3_case_summary.csv
```

`d3-case-001-central` should show 3 valid memory and 3 valid time trials with non-empty
medians. If it shows zeros or blanks, something is wrong — do not start the full run.

---

## 3. The full run

```powershell
python benchmark\run_design_3.py
```

260 trials (4 cases × [5 warm-ups + 30 memory + 30 time]), one fresh child process each.
**Expect 80–90 minutes.** It prints progress per trial, so you can watch it:

```text
[   1/260] d3-case-001-central memory trial 0 (warmup)  valid=True code=-   21.5s elapsed
```

Shorter runs are available if you want a faster read, but they are **not** protocol-compliant
and `assemble_metrics.py` will report `all_cases_valid: false`:

```powershell
python benchmark\run_design_3.py --memory-trials 5 --time-trials 5
```

Then generate the metrics:

```powershell
python benchmark\maintainability_check.py
python benchmark\assemble_metrics.py
```

`assemble_metrics.py` prints the final numbers and writes `results/design_3_final_metrics.json`.

---

## 4. Reading the results

```powershell
Get-Content results\design_3_final_metrics.json
```

| File | Contents |
|---|---|
| `design_3_final_metrics.json` | **The five metric values.** Generated, never hand-edited. |
| `design_3_trials.csv` | One row per trial. The raw record. |
| `design_3_case_summary.csv` | Per-case medians and interquartile ranges. |
| `design_3_mesh_diagnostics.csv` | Per-panel mesh and flattening diagnostics. |
| `design_3_failures.jsonl` | Every failure, with its code and message. |
| `design_3_maintainability.csv` | One row per allow-listed module. |
| `design_3_environment.json` | Interpreter, CPU, RAM, libraries, git commit, config hash. |
| `design_3_config_manifest.json` | Config version + hash, cases, trial counts, MI allow-list. |

Check the provenance lines before quoting any number:

```powershell
python -c "import json; d=json.load(open('results/design_3_environment.json')); print(d['git']['commitHash']); print(d['git']['dirtyWorkingTree']); print(d['configurationHash'])"
```

`dirtyWorkingTree` should be `false`. If it is `true`, something changed after the commit and
the numbers no longer correspond to a single source state — commit it and rerun.

---

## 5. Rules that make the numbers defensible

These come from the benchmark protocol. Breaking any of them invalidates the result, and
they are the first thing a reviewer will check.

- **Commit before you run.** Every trial records the commit it ran on. A run over a dirty
  tree cannot be tied to a source state.
- **Do not edit anything under `backend/app/` or the config while a run is in progress.**
  Each child process imports the files fresh, so trials would span two source states. Stop,
  commit, restart.
- **Do not edit `benchmark/config-design-3.yaml` or `fixtures/garments/d3-basic-torso.yaml`
  to improve a number.** That changes `configuration_hash` and invalidates every earlier result.
- **Never delete or rerun a single unfavourable trial.** `run_design_3.py` does not retry and
  does not filter; keep it that way. If a trial fails it stays in the CSV with its code.
- **Do not rerun until you like the number.** The protocol compares against the first
  measured run.

---

## 6. Individual scripts

Each can be run on its own.

```powershell
# Metric 5 alone — a few seconds, no trials
python benchmark\maintainability_check.py

# Final metrics from whatever raw files are present
python benchmark\assemble_metrics.py

# Environment record alone
python benchmark\environment.py

# One single trial, for debugging
python benchmark\_child.py --case d3-case-001-central --pass memory
python benchmark\_child.py --case d3-case-001-central --pass time
```

Each prints one JSON object to stdout.

---

## 7. Tests

```powershell
cd backend
python -m pytest
```

223 tests. `tests/test_benchmark_harness.py` covers the bookkeeping rules — an unmeasured
field is blank rather than zero, the case summary is derived from the raw rows, and the MI
allow-list names files that exist. The allow-list test **skips** if Radon is not installed,
because Radon is a benchmark tool rather than a backend dependency.

---

## Why two metrics are null

**Metric 4 — pattern-dimension MAE.** It needs reference seam-line dimensions from
*independently approved patterns* for the garment this design generates. The only reference
in this repository, `fixtures/garmageset-reference.json`, is a 19-panel skirt-and-dress set,
not a torso bodice. A reference computed from a drafting oracle would be a reference Design 3
itself helped produce, which the protocol specifically rules out. So the field is `null` rather
than a number the protocol cannot support.

To fill it in, you would need to add an approved `fixtures/references/dimension_reference.csv`
(roughly: `case_id,dimension_id,panel_id,path_or_landmarks,reference_mm,tolerance_mm,approved_by,approval_date`)
plus a `dimension_check.py` to compare against it.

**Metric 2 — feature completeness.** It needs a frozen, approved `feature_manifest.csv`.
Note that darts and notches are modelled, rendered and exported throughout the backend but
are **never populated** by the pipeline, so every Design 3 pattern currently has zero of
each. Writing a manifest today would mean either overstating completeness or declaring them
non-applicable — that is a scoping decision about the garment, not a measurement.

---

## 8. Notes on interpreting the numbers

- **Memory is Python-tracked only.** `tracemalloc` does not see allocations inside NumPy,
  BLAS or ezdxf. The value is small (~3 MB) for that reason. Process RSS is not measured and
  must not be substituted for this number.
- **Generation time includes the DXF file write**, because the protocol's timed boundary runs
  generate → validate → export → file.
- **The control case is meant to fail.** `d3-case-004-off-average` lies outside the supported
  domain. It is expected to report `OUT_OF_DOMAIN`. A run where it succeeds means something
  is wrong.
- **Only 3 mandatory cases exist**, not the 4 the protocol suggests. `pascual` — the fourth
  fixture — is out of the supported domain, which is why it is a control case.
- **Metric 5 depends directly on the allow-list** in `config-design-3.yaml`. Adding or
  removing a file there moves the number. The 17 listed files are the production modules
  implementing stages A–I; shared schemas, the common validator, the common DXF exporter,
  config and tests are excluded by the protocol.