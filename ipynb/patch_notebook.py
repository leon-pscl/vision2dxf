"""Patch model_training_measurement.ipynb per ipynb/fixbrief.md.

Applies three kinds of change, in this order so that indices stay stable:

1. REPLACE — swap a cell's source in place, keyed by its index in the ORIGINAL notebook.
2. DELETE  — remove a cell that a new cell supersedes.
3. INSERT  — add cells after a named anchor cell (original index), so later indices do not shift
   out from under steps 1 and 2.

Usage
-----
    python patch_notebook.py --check     validate every patch source parses; write nothing
    python patch_notebook.py --dry-run   print the plan; write nothing
    python patch_notebook.py             apply the patch

The script refuses to write unless every patch source parses, and refuses to replace a cell whose
type does not match the source. Both are guards against silently corrupting the notebook.

Notes
-----
The patched notebook is a Kaggle artefact; this script runs anywhere Python 3.8+ is available and
needs only the standard library.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

HERE = Path(__file__).resolve().parent
NB_PATH = HERE / "model_training_measurement.ipynb"
SRC_DIR = HERE / "patch_cells"

#: Cell index in the ORIGINAL notebook -> patch source filename.
REPLACE: Dict[int, str] = {
    0: "md_000_header.md",
    3: "cell_003_setup.py",
    8: "cell_008_config.py",
    10: "cell_010_bodym_data.py",
    16: "cell_016_bodym_eda.py",
    34: "cell_034_bodym_metrics.py",
    37: "md_037_b1_baselines.md",
    38: "cell_038_bodym_baselines.py",
    44: "cell_044_b1_fit.py",
    47: "cell_047_bodym_cnn.py",
    50: "cell_050_net_probe.py",
    52: "cell_052_bodym_training.py",
    54: "cell_054_training_import.py",
    58: "md_058_sensitivity.md",
    59: "cell_059_bodym_perturb.py",
    62: "cell_062_bodym_sensitivity.py",
    64: "cell_064_sweep_run.py",
    66: "cell_066_sensitivity_report.py",
    67: "md_067_sensitivity_interp.md",
    68: "md_068_production.md",
    69: "cell_069_augmenter_checks.py",
    72: "cell_072_train_variants.py",
    73: "md_073_variant_eval.md",
    74: "cell_074_variants_eval.py",
    75: "md_075_variants_interp.md",
    76: "md_076_conformal.md",
    77: "cell_077_bodym_conformal.py",
    79: "cell_079_conformal_check.py",
    81: "cell_081_conformal_fit.py",
    84: "cell_084_conformal_freeze.py",
    86: "md_086_final_eval.md",
    87: "cell_087_protocol_audit.py",
    89: "cell_089_final_eval.py",
    93: "md_093_final_interp.md",
    95: "cell_095_strata.py",
    97: "md_099_preprocess.md",
    100: "cell_100_preprocess.py",
    102: "cell_102_mirror.py",
    103: "md_103_preprocess_interp.md",
    106: "cell_106_infer.py",
    112: "cell_112_smoke_run.py",
    115: "cell_115_schema_map.py",
    117: "cell_117_model_card.py",
    119: "cell_119_decisions.py",
    121: "cell_121_references.py",
    123: "cell_123_bundle.py",
    125: "cell_125_verify_bundle.py",
    126: "md_126_bundle_interp.md",
    132: "cell_132_linter.py",
    134: "cell_134_summary.py",
    136: "cell_136_summary_figure.py",
}

#: Cells removed because a new cell supersedes them.
DELETE: Tuple[int, ...] = (98,)   # old reference-tensor cell; the new §9.1 cell does it plus ONNX

#: (anchor index in the ORIGINAL notebook, filename), inserted in listed order after that anchor.
INSERT_AFTER: List[Tuple[int, str]] = [
    (70, "cell_070_augmenter_timing.py"),       # fix C2: time 100 augmenter calls
    (75, "md_075_sensitivity_shipped.md"),      # fix B9: markdown for the shipped-model sweep
    (75, "cell_076_sensitivity_shipped.py"),    # fix B9: the sweep itself
    (96, "md_096_weights_onnx.md"),             # fix A8: markdown for the new §9.1
    (96, "cell_097_weights_onnx.py"),           # fix A8: weights + ONNX + equivalence check
    (99, "md_099_export.md"),                   # §9 header before preprocess.py
]


def read_source(name: str) -> str:
    """Read a patch source file.

    Parameters
    ----------
    name : str
        Filename under ``patch_cells/``.

    Returns
    -------
    str
        File contents, BOM-stripped, with a guaranteed trailing newline.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    p = SRC_DIR / name
    if not p.is_file():
        raise FileNotFoundError(f"missing patch source: {p}")
    return p.read_text(encoding="utf-8-sig")


def to_lines(text: str) -> List[str]:
    """Split text into notebook source lines, keeping line endings.

    Parameters
    ----------
    text : str
        Raw file text.

    Returns
    -------
    list of str
        Lines, each retaining its newline except possibly the last.
    """
    return text.splitlines(keepends=True)


def make_cell(name: str, source_text: str) -> dict:
    """Build a notebook cell of the right type from a patch source.

    Parameters
    ----------
    name : str
        Patch source filename; the extension determines the cell type.
    source_text : str
        Cell source.

    Returns
    -------
    dict
        A cell dict valid for nbformat 4.
    """
    if name.endswith(".md"):
        return {"cell_type": "markdown", "metadata": {}, "source": to_lines(source_text)}
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": to_lines(source_text)}


def validate() -> List[str]:
    """Check that every patch source exists and parses.

    Markdown sources are checked only for existence; Python sources are parsed. A ``%%writefile``
    first line is stripped before parsing, because a cell magic is not valid Python.

    Returns
    -------
    list of str
        Problems found; empty when everything is fine.
    """
    problems: List[str] = []
    all_items = ([(i, n) for i, n in REPLACE.items()]
                 + [(a, n) for a, n in INSERT_AFTER])
    for idx, name in sorted(all_items, key=lambda kv: (kv[1], kv[0])):
        if not (SRC_DIR / name).is_file():
            problems.append(f"cell {idx}: missing source {name}")
            continue
        if not name.endswith(".py"):
            continue
        src = read_source(name)
        body = src
        if body.lstrip().startswith("%%"):
            body = body.split("\n", 1)[1] if "\n" in body else ""
        try:
            ast.parse(body)
        except SyntaxError as exc:
            problems.append(f"{name} (cell {idx}): line {exc.lineno}: {exc.msg}")
    return problems


def main() -> int:
    """Apply the patch.

    Returns
    -------
    int
        0 on success, 1 on a validation failure, 2 if the notebook is missing.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    ap.add_argument("--check", action="store_true", help="validate sources only")
    args = ap.parse_args()

    problems = validate()
    if problems:
        print("VALIDATION FAILED — nothing written:", file=sys.stderr)
        for p in problems:
            print("  " + p, file=sys.stderr)
        return 1
    n_sources = len(REPLACE) + len(INSERT_AFTER)
    print(f"validated {n_sources} patch sources "
          f"({len(REPLACE)} replace, {len(INSERT_AFTER)} insert, {len(DELETE)} delete)")

    if not NB_PATH.is_file():
        print(f"not found: {NB_PATH}", file=sys.stderr)
        return 2
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cells = nb["cells"]
    print(f"notebook: {NB_PATH.name} ({len(cells)} cells)")

    if args.dry_run or args.check:
        print("\nPLAN")
        for idx in sorted(REPLACE):
            print(f"  replace {idx:>4}  {REPLACE[idx]}")
        for anchor, name in INSERT_AFTER:
            print(f"  insert  after {anchor:>4}  {name}")
        for idx in sorted(DELETE):
            print(f"  delete  {idx:>4}")
        print("\n(dry run: nothing written)")
        return 0

    # ---- 1. replace, using ORIGINAL indices ------------------------------------------------
    for idx in sorted(REPLACE):
        name = REPLACE[idx]
        if idx >= len(cells):
            print(f"  SKIP cell {idx}: does not exist", file=sys.stderr)
            continue
        cell = cells[idx]
        want = "markdown" if name.endswith(".md") else "code"
        if cell["cell_type"] != want:
            print(f"  SKIP cell {idx}: is {cell['cell_type']}, source {name} is {want}",
                  file=sys.stderr)
            continue
        before = len("".join(cell["source"]))
        text = read_source(name)
        cell["source"] = to_lines(text)
        print(f"  replace {idx:>4}  {want:8s} {name}  ({before} -> {len(text)} chars)")

    # ---- 2. delete, highest index first ----------------------------------------------------
    for idx in sorted(DELETE, reverse=True):
        if idx < len(cells):
            head = "".join(cells[idx]["source"])[:60].replace("\n", " ")
            cells.pop(idx)
            print(f"  delete  {idx:>4}  ({head}...)")

    # ---- 3. insert, lowest anchor first so earlier inserts do not shift later anchors -----
    # Build the new list in one pass so anchors always refer to original indices.
    new_cells: List[dict] = []
    by_anchor: Dict[int, List[dict]] = {}
    for anchor, name in INSERT_AFTER:
        by_anchor.setdefault(anchor, []).append(make_cell(name, read_source(name)))

    for i, cell in enumerate(cells):
        new_cells.append(cell)
        for extra in by_anchor.get(i, []):
            new_cells.append(extra)
    # anchors pointing past the end (should not happen, but be explicit)
    for anchor in sorted(a for a in by_anchor if a >= len(cells)):
        for extra in by_anchor[anchor]:
            new_cells.append(extra)
    for anchor in sorted(by_anchor):
        for extra in by_anchor[anchor]:
            kind = extra["cell_type"]
            print(f"  insert  after {anchor:>4}  {kind:8s} "
                  f"{''.join(extra['source'])[:40].splitlines()[0][:40]}...")

    # ---- 4. normalise code cells ---------------------------------------------------------
    # nbformat 4.5 requires `id`, `execution_count` and `outputs` on every code cell, and
    # rejects a notebook missing them. The source notebook omitted them, so `nbformat.validate`
    # failed on it before this patch. Adding the keys costs nothing and makes the file valid.
    _normalised = 0
    for i, cell in enumerate(new_cells):
        if cell["cell_type"] != "code":
            continue
        for key, default in (("id", f"cell-{i:03d}"), ("execution_count", None),
                             ("outputs", []), ("metadata", {})):
            if key not in cell:
                cell[key] = default
                _normalised += 1
    print(f"  normalise: added {_normalised} required nbformat keys to code cells")

    nb["cells"] = new_cells
    nb.setdefault("metadata", {})["patched_by"] = "patch_notebook.py (fixbrief 2026-10-07)"
    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwrote {NB_PATH}: {len(new_cells)} cells "
          f"(was {len(cells)}, +{len(new_cells)-len(cells)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())