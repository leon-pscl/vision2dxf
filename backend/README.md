# Backend benchmark and analysis scripts

The backend contains two stand-alone scripts used to collect the design metrics
(constraints values) reported for Design 1. Run them from the `backend`
directory with the project's development interpreter.

## Prerequisites

The repository already provides a working environment at `../.venv-dev`.
Use it by default; do not install into a fresh/MSYS environment.

```bash
cd backend
../.venv-dev/Scripts/python.exe -m pip --version
```

## 1. Design 1 drafting benchmark

Measures wall-clock inference time, storage/efficiency usage, scalability
growth and mean memory consumption for a single `generate()` call plus PDF export.
Results are written to `results/benchmark_design1.csv` and
`results/benchmark_design1.json`, and a PDF to
`results/benchmark_design1_output.pdf`.

```bash
cd backend
../.venv-dev/Scripts/python.exe benchmark_design1.py
```

On Linux/macOS use `../.venv-dev/bin/python` instead.

## 2. Maintainability index

Computes an SLOC-weighted mean of file-level Maintainability Index scores for the
files listed in `maintainability_manifest_design1.json`. Per-file results and a
summary are written to `results/maintainability_index/`.

```bash
cd backend
../.venv-dev/Scripts/python.exe compute_maintainability_index.py
```

Optional arguments:

```bash
../.venv-dev/Scripts/python.exe compute_maintainability_index.py \
  --manifest maintainability_manifest_design1.json \
  --out-dir results/maintainability_index
```

## Output locations

- `backend/results/benchmark_design1.csv`
- `backend/results/benchmark_design1.json`
- `backend/results/benchmark_design1_output.pdf`
- `backend/results/maintainability_index/design1_per_file.csv`
- `backend/results/maintainability_index/design1_summary.json`
