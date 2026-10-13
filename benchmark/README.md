# Design 2 benchmark — how to run it

Measures the Chapter 3 metrics for **Design 2** (GarmageSet-derived base-block
transformation) against the frozen configuration in this repository.

| Metric | What it is | Script | Status |
|---|---|---|---|
| **1 — Peak memory** | Python-tracked peak memory in MB, median across mandatory measured trials | `run_design_2.py` | measured |
| **3 — Generation time** | Median pattern-generation time in seconds, same trial population | `run_design_2.py` | measured |
| **5 — Maintainability** | Radon 6.0.1 MI on the 0–100 scale, source-line weighted | `maintainability_check.py` | blocked, see Step 0 |
| **2 — Feature completeness** | Required-feature completeness in percent | — | `null`, see [Why two metrics are null](#why-two-metrics-are-null) |
| **4 — Dimension MAE** | Pattern-dimension mean absolute error in cm | — | `null`, see [Why two metrics are null](#why-two-metrics-are-null) |

Everything runs against **one frozen base block**, `bodice-ref-a`. The runner refuses to
start if the template bytes, the benchmark configuration and the block catalog do not all
agree on its hash.

---

## Step 0 — Recover the source (required before Metric 5)

`backend/app/base_block/`, `backend/app/transformation/` and `backend/app/validation/topology.py`
exist **only as CPython 3.13 `.pyc` files with no `.py` source**, in this repository or
anywhere in its git history. `RECONSTRUCTION.md` records how that happened and why it is
not fixed. Radon cannot read bytecode, so Metric 5 cannot be computed until the source is
back. Metrics 1 and 3 do not need it — they run against the bytecode.

The tools that handle 3.13 bytecode:

```powershell
# Option A - PyLingual, pure Python, supports 3.6 to 3.13. Needs uv and model downloads.
python -m pip install pylingual
pylingual -v 3.13 backend\app\transformation\pipeline.pyc -o recovered\

# Option B - pycdc, C++, 2.0 to 3.14. Needs a C++ toolchain and a build.
git clone https://github.com/zrsx/pycdc
cd pycdc && cmake -B build -S . && cmake --build build --config Release
.\build\Release\pycdc.exe ..\..\backend\app\transformation\pipeline.pyc
```

Neither tool is trusted here, so **every recovered module has to clear this gate** before
it replaces the bytecode:

1. It imports cleanly and `dir(module)` matches the bytecode module's public names.
2. It produces **byte-identical** `pattern.json`, validation report and diagnostics, for
   all four case fixtures, against the bytecode.
3. The full backend test suite stays green.

```powershell
cd backend; python -m pytest -q          # must be 145 + the benchmark harness tests
```

A module that fails the gate stays `.pyc`-only, and says so in `RECONSTRUCTION.md`.

Once every module passes, delete the `.pyc` files, drop the three `!` exceptions from
`.dockerignore`, and unpin the `Dockerfile` from `python:3.13-slim` — the interpreter is
only pinned because of this loss.

---

## Step 1 — Prerequisites

Run everything with an interpreter that has **`radon==6.0.1`**, plus numpy, shapely, ezdxf,
pydantic, PyYAML and reportlab. Freeze the exact interpreter that produced the committed
results once the run is done:

```powershell
python -c "import radon, numpy, shapely, ezdxf; print('ok')"
python -m pip install radon==6.0.1      # only needed if the check above fails
```

Freeze it after the official run, not before:

```powershell
python -m pip freeze > requirements-design-2.lock
```

> Design 2's bytecode-only modules are CPython **3.13**-only. Another interpreter cannot
> load them at all, so the version is part of the environment record, not an incidental
> detail.

---

## Step 2 — Commit before you run

Every trial records the commit it ran on. A run over a dirty tree cannot be tied to a
single source state.

```powershell
git status --short          # must be empty
```

---

## Step 3 — Smoke run first — always

Two minutes, one case. Proves the harness before you commit to the full run.

```powershell
python benchmark\run_design_2.py --smoke
```

Then check it produced something sane:

```powershell
Get-Content results\design_2_case_summary.csv
```

`d2-case-002-larger` should show 3 valid memory and 3 valid time trials with non-empty
medians. If it shows zeros or blanks, something is wrong — do not start the full run.

---

## Step 4 — Maintainability

```powershell
python benchmark\maintainability_check.py
```

A few seconds, no trials. If Step 0 is not done it **stops and names the missing
modules** rather than scoring a shortened list, and
`maintainability_index_0_100` stays `null` in the final metrics.

---

## Step 5 — The full run

```powershell
python benchmark\run_design_2.py
```

260 trials (4 cases × [5 warm-ups + 30 memory + 30 time]), one fresh child process each.
Design 2 transforms a whole block in well under a tenth of a second, so process start-up
dominates and this should take roughly 10–25 minutes. It prints progress per trial:

```text
[   1/260] d2-case-001-identity memory trial 0 (warmup)  valid=True code=-   2.1s elapsed
```

Shorter runs exist for a faster read, but they are **not** protocol-compliant and
`assemble_metrics.py` will report `all_cases_valid: false`:

```powershell
python benchmark\run_design_2.py --memory-trials 5 --time-trials 5
```

Then generate the metrics:

```powershell
python benchmark\assemble_metrics.py
```

---

## Step 6 — Reading the results

```powershell
Get-Content results\design_2_final_metrics.json
```

| File | Contents |
|---|---|
| `design_2_final_metrics.json` | **The metric values.** Generated, never hand-edited. |
| `design_2_trials.csv` | One row per trial. The raw record. |
| `design_2_case_summary.csv` | Per-case medians and interquartile ranges. |
| `design_2_failures.jsonl` | Every failure, with its code and message. |
| `design_2_maintainability.csv` | One row per allow-listed module. |
| `design_2_environment.json` | Interpreter, CPU, RAM, libraries, git commit, config hashes. |
| `design_2_base_block_record.json` | Base-block provenance: source file, hash, unit, transforms. |
| `design_2_config_manifest.json` | Config version + hash, cases, trial counts, MI allow-list. |

Check the provenance before quoting any number:

```powershell
python -c "import json; d=json.load(open('results/design_2_environment.json')); print(d['git']['commitHash']); print(d['git']['dirtyWorkingTree']); print(d['configurationHash'])"
```

`dirtyWorkingTree` should be `false`. If it is `true`, something changed after the commit
and the numbers no longer correspond to one source state — commit it and rerun.

---

## Rules that make the numbers defensible

These come from the benchmark protocol. Breaking any of them invalidates the result, and
they are the first thing a reviewer will check.

- **Commit before you run.** Every trial records the commit it ran on.
- **Do not edit anything under `backend/app/`, `assets/` or the config while a run is in
  progress.** Each child process imports the files fresh, so trials would span two source
  states. Stop, commit, restart.
- **Do not edit `benchmark/config-design-2.yaml`, `assets/base-blocks/catalog.yaml`, the
  frozen template or any case fixture to improve a number.** That changes
  `configuration_hash` and invalidates every earlier result.
- **Never delete or rerun a single unfavourable trial.** `run_design_2.py` does not retry
  and does not filter; keep it that way. A failed trial stays in the CSV with its code.
- **Do not rerun until you like the number.** The protocol compares against the first
  measured run.

---

## The cases

| Case | Role | What it is |
|---|---|---|
| `d2-case-001-identity` | mandatory | The block's own reference profile. Every difference is zero, so a correct transformation reproduces the block exactly. |
| `d2-case-002-larger` | mandatory | Larger than the block is declared to fit, inside the tested envelope. |
| `d2-case-003-smaller` | mandatory | Smaller, every difference running the opposite way, so a transformation that only works in one direction shows up. |
| `d2-case-004-off-domain` | control | **Expected to be refused** with `OUT_OF_DOMAIN`. It exercises the failure policy and keeps the refusal in the log. A run where it succeeds means something is wrong. |

Each case fixture carries the same `case:` provenance block the Design 3 fixtures use:
pseudonymous, no direct identifier, `value_status: confirmed`, approved and dated. The
engine reads only the `body` mapping and converts to millimetres internally.

---

## Individual scripts

Each can be run on its own.

```powershell
# Metric 5 alone - a few seconds, no trials
python benchmark\maintainability_check.py

# Final metrics from whatever raw files are present
python benchmark\assemble_metrics.py

# Environment record alone
python benchmark\environment.py

# One single trial, for debugging
python benchmark\_child.py --case d2-case-002-larger --pass memory
python benchmark\_child.py --case d2-case-002-larger --pass time
```

Each prints one JSON object to stdout.

---

## Tests

```powershell
cd backend
python -m pytest
```

`tests/test_benchmark_harness.py` covers the bookkeeping rules — an unmeasured field is
blank rather than zero, the case summary is derived from the raw rows, the frozen base
block still matches its recorded hash, and the MI allow-list names files that exist. The
allow-list test **skips with a named reason** until Step 0 is done; running
`maintainability_check.py` itself is the hard gate.

---

## Why two metrics are null

**Metric 4 — pattern-dimension MAE.** It needs reference seam-line dimensions from
*independently approved patterns* for the block Design 2 generates. No such reference
exists in this repository for any design: the only one here,
`fixtures/garmageset-reference.json`, is a 19-panel skirt-and-dress set, not a bodice
block. A reference computed from the engine's own influence map would be the engine
grading itself, which the protocol rules out. So the field is `null` rather than a number
the protocol cannot support.

To fill it in you would need an approved `fixtures/references/dimension_reference.csv`
(`case_id,dimension_id,panel_id,path_or_landmarks,reference_mm,tolerance_mm,approved_by,approval_date`)
plus a `dimension_check.py`.

**Metric 2 — feature completeness.** It needs a frozen, approved `feature_manifest.csv`.
Note that darts and notches are modelled, rendered and exported throughout the backend but
are **never populated** by the pipeline, so every Design 2 pattern currently has zero of
each. Writing a manifest today would mean either overstating completeness or declaring
them non-applicable, and that is a scoping decision about the garment for a pattern maker
to make, not a measurement to take.

---

## Notes on interpreting the numbers

- **Memory is Python-tracked only.** `tracemalloc` does not see allocations inside NumPy,
  BLAS or ezdxf. The value will be small for that reason. Process RSS is not measured and
  must not be substituted for this number.
- **Generation time includes the DXF file write**, because the protocol's timed boundary
  runs generate → validate → export → file.
- **The identity case is the one to read first.** Its differences are all zero, so it is
  the cheapest possible check that the transformation reproduces the block it came from.
- **Unmapped differences are expected and are not failures.** None of the three blocks has
  a hip edge or a marked bust apex, so hip and bust-apex measurements have no landmark to
  move. They are reported as unmapped rather than absorbed somewhere they do not belong.
  See the README's "Known limits".
- **Metric 5 depends directly on the allow-list** in `config-design-2.yaml`. The 14 listed
  modules are the base-block and transformation code; shared schemas, the common
  validator, the common DXF exporter, the HTTP API, config and tests are excluded by the
  protocol. The `role` strings were written from each module's interface and the
  configuration around it, because the source these describe is not yet recovered.