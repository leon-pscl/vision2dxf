# pattern_drafter

Draft 1:1 sewing patterns from a simple **YAML measurement file** using the
[OpenPattern](https://github.com/fmetivier/OpenPattern) library, then export
the result as:

- a structured **JSON** description (panels, vertices, construction points,
  bounds) that other tools can consume, and
- a visual **PDF** rendering of the drafted pattern for inspection or printing.

Three base templates are supported out of the box: **bodice**, **skirt** and
**trousers**, all through a single command line interface (`app.py`).

```
examples/*.yaml ──▶ pattern_drafter ──▶ outputs/*.json
                          │                    outputs/*.pdf
                          └── OpenPattern (drafting math)
```

## Installation

```bash
#linux
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```bash
#ps
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`OpenPattern` is not on PyPI, so `requirements.txt` installs it from its Git
repository. It depends on `matplotlib`, `numpy` and `scipy` and is licensed
under **GPL-3.0**; make sure that license is acceptable for your use.

Only the measurement parsing / validation code (`pattern_drafter.measurements`,
`app.py check` and `app.py templates`) works without OpenPattern installed.
Actually drafting a pattern (`app.py build`) requires the full stack above.

## Quick start

```bash
# List the supported base templates and their available styles
python app.py templates

# Validate the example measurement files without drafting
python app.py check examples

# Draft the examples: writes outputs/<profile_name>.json and .pdf
python app.py build examples
```

## Command line reference

| Command | Description |
|---------|-------------|
| `python app.py templates` | List supported templates, styles and required length alternatives. |
| `python app.py check FILE...` | Parse and validate YAML files only. No OpenPattern needed. |
| `python app.py build FILE...` | Draft and export. Accepts files and/or directories. |

`build` options:

| Option | Default | Description |
|--------|---------|-------------|
| `-o`, `--outdir` | `outputs` | Directory for the generated `.json` / `.pdf` files. |
| `--formats` | `json,pdf` | Comma separated list of `json`, `pdf` or both. |

Exit codes: `0` success, `1` some files failed, `2` bad arguments / no files,
`3` OpenPattern (or another drafting dependency) is not installed.

## YAML measurement file

Each file describes one pattern profile:

```yaml
template: bodice        # bodice | skirt | trousers
style: Gilewska         # Gilewska | Donnanno | Chiappetta (template dependent)
gender: w               # w = woman, m = man
profile_name: my_bodice # used for the output file names
ease: 8                 # optional wearing ease in cm
darts: true             # optional: add bust + waist darts (bodice)

measurements:           # native OpenPattern (French) keys, values in cm
  tour_poitrine: 84.0
  tour_taille: 62.0
  # ...
```

Defaults when omitted: `style` and `gender` fall back to the template default,
`profile_name` falls back to the file name, `ease` defaults to `8` and `darts`
to `false`.

Copy [`examples/bodice.yaml`](examples/bodice.yaml),
[`examples/skirt.yaml`](examples/skirt.yaml) or
[`examples/trousers.yaml`](examples/trousers.yaml) to get started.

### Templates

| Template | OpenPattern class | Default style / gender | Styles |
|----------|-------------------|------------------------|--------|
| `bodice` | `Basic_Bodice` | `Gilewska` / `w` | Gilewska, Donnanno, Chiappetta |
| `skirt` | `Basic_Skirt` | `Gilewska` / `w` | Gilewska, Donnanno, Chiappetta |
| `trousers` | `Basic_Trousers` | `Donnanno` / `m` | Donnanno |

Trousers additionally accept **one of** `longueur_tot`, `hauteur_taille_terre`,
`longueur_taille_terre` or `entrejambe_terre` to define the total length.
Validation reports exactly which measurements are missing for the chosen
template/style/gender.

### Measurement keys

Values are the native (French) OpenPattern keys. The most common ones:

| Key | Meaning |
|-----|---------|
| `tour_poitrine` | bust circumference |
| `tour_taille` | waist circumference |
| `tour_bassin` | hip circumference |
| `tour_encolure` | neck circumference |
| `tour_bras` | upper-arm circumference |
| `tour_poignet` | wrist circumference |
| `carrure_dos` / `carrure_devant` | back / front shoulder width |
| `ecart_poitrine` | bust point separation |
| `hauteur_poitrine` | bust height |
| `hauteur_bassin` | hip depth |
| `hauteur_taille_genou` | waist to knee length |
| `longueur_dos` / `longueur_devant` | back / front length |
| `longueur_epaule` | shoulder length |
| `longueur_manche` | sleeve length |
| `montant` | rise / crotch depth |

The full glossary lives in
[`pattern_drafter/measurements.py`](pattern_drafter/measurements.py)
(`MEASUREMENT_GLOSSARY`).

## Output

For a profile named `bodice_gilewska_w36` the program writes:

- `outputs/bodice_gilewska_w36.json`
- `outputs/bodice_gilewska_w36.pdf`

### JSON schema (v1.0)

```jsonc
{
  "schema_version": "1.0",
  "template": "bodice",
  "style": "Gilewska",
  "gender": "w",
  "units": "cm",
  "profile_name": "bodice_gilewska_w36",
  "input_measurements":   { "tour_poitrine": 84.0, "...": 0 },
  "resolved_measurements": { "...": 0 },   // incl. values computed by OpenPattern
  "panels": [
    { "name": "front", "closed": true, "vertices": [[x, y], "..."] },
    { "name": "back",  "closed": true, "vertices": [[x, y], "..."] },
    { "name": "sleeve","closed": true, "vertices": [[x, y], "..."] }
  ],
  "points": {
    "front": { "points": { "WF": [x, y] }, "curves": { "Front_Sleeve": [[x, y]] } },
    "back":  { "points": { "...": [x, y] }, "curves": {} }
  },
  "bounds": { "xmin": 0.0, "xmax": 0.0, "ymin": 0.0, "ymax": 0.0 },
  "metadata": {
    "generator": "pattern_drafter",
    "openpattern_class": "Basic_Bodice",
    "openpattern_version": "1.0",
    "source_yaml": "examples/bodice.yaml",
    "generated_at": "2026-10-11T00:00:00+00:00"
  }
}
```

All coordinates are in centimetres. Each entry in `panels` is a closed polygon;
`points` holds the named construction points and any spline curves per piece.

### PDF

The PDF is rendered through OpenPattern's own matplotlib drawing routines at
1:1 scale (1 unit = 1 cm) and includes the point labels, fold/grain lines,
waist/bust/hip guide lines and a 5 cm scale bar.

## Repository layout

```
.
├── app.py                         # CLI entry point
├── pattern_drafter/
│   ├── measurements.py            # YAML parsing, validation, sqlite bridge
│   ├── drafting.py                # OpenPattern integration + geometry extraction
│   ├── export.py                  # JSON + PDF writers
│   └── pipeline.py                # orchestration used by app.py and tests
├── examples/
│   ├── bodice.yaml
│   ├── skirt.yaml
│   └── trousers.yaml
└── tests/                         # pytest suite
```

## Troubleshooting

### `build` fails with the "OpenPattern is required" message

If `app.py build` reports `OpenPattern is required to draft patterns` even
though the installation succeeded (and `app.py templates` / `check` still work),
the underlying cause is usually the deprecated `pkg_resources` module: OpenPattern
imports it at load time, and `pkg_resources` was **removed** from `setuptools`
starting with **setuptools 82** (February 2026).

`pattern_drafter` handles this automatically by injecting a minimal
`pkg_resources` stand-in before importing OpenPattern (OpenPattern only uses it
to locate a bundled database that this tool overrides anyway), so no action is
needed. If you still hit the error, run:

```bash
python -c "import pkg_resources"
```

and if it fails, reinstall a setuptools that still ships `pkg_resources`:

```bash
pip install --force-reinstall "setuptools<82"
```

## Development

```bash
# Run the test suite
pytest -q

# The measurement / export tests run without OpenPattern; drafting tests are
# skipped automatically when the library is unavailable.
```

## Limitations

- Drafts use OpenPattern's base blocks; no seam allowance, hem allowance or
  final sizing is applied.
- Measurements are in centimetres; the native OpenPattern keys are used as-is.
- Trousers are only implemented for the Donnanno style upstream.
- Bodice darts (`darts: true`) call OpenPattern's `add_bust_dart` /
  `add_waist_dart`; leave `darts: false` for the plain block.
