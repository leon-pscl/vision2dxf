# vision2dxf

Tools to search a corpus of Style3D (GarmageSet) sewing-pattern JSON files and
render the matching ones as PDF sewing patterns (and optionally DXF).

## Scripts

| Script | What it does |
| --- | --- |
| `search_patterns.py` | Scans `sample_patterns/*.json` for polo/t-shirt-like patterns (bodice front + back + sleeves, no skirt, no ruffle) and copies them into `pattern_templates/`. |
| `pattern_generator.py` | Converts every JSON in `pattern_templates/` into a PDF sewing pattern in `pattern_template_output/`. Picks the best curve interpretation (polyline / Catmull-Rom / Bézier) by minimizing stitch-length mismatch between joined edges. |

Dependencies (see `requirements.txt`): `numpy`, `matplotlib`.

## Setup

Requires Python 3.10+ (tested on 3.14).

```bash
# 1. Create a virtual environment (creates ./.venv)
python3 -m venv .venv

# 2. Activate it
# Linux / macOS
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Windows (cmd)
.venv\Scripts\activate.bat

# 3. Install dependencies
pip install -r requirements.txt
```

The `.venv/` directory is git-ignored — only the scripts and this README are
synced to GitHub. To deactivate later, run `deactivate` in the terminal.

## Usage

Run the scripts from the repository root, with the virtual environment
activated (or use `.venv/bin/python ...` directly).

```bash
# 1. Filter polo/tshirt patterns out of sample_patterns/ into pattern_templates/
python search_patterns.py

# 2. Render each pattern in pattern_templates/ to PDF
python pattern_generator.py
```

Notes:

- `pattern_generator.py` reads from `pattern_templates/` and writes PDFs to
  `pattern_template_output/` (created automatically). Files whose PDF already
  exists are skipped, so re-runs only process new input.
- Both `sample_patterns/` and the output folders are git-ignored.
