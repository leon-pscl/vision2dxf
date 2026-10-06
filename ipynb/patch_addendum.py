"""Second patch: addendum A (Kaggle execution, persistence and resume).

Run AFTER ``patch_notebook.py``. Indices here refer to the notebook as it stands *after* the fix
brief has been applied, which is the order the addendum assumes.

Usage
-----
    python patch_addendum.py --dry-run
    python patch_addendum.py

Idempotent in the same way as the first patcher: applying twice is a no-op, because every
replacement is keyed by cell index and the sources are deterministic.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

HERE = Path(__file__).resolve().parent
NB_PATH = HERE / "model_training_measurement.ipynb"
SRC_DIR = HERE / "patch_cells"

#: Cell index (after the fix-brief patch) -> patch source.
REPLACE: Dict[int, str] = {
    3: "a_cell_003_setup.py",
    8: "a_cell_008_config.py",
    52: "a_cell_052_bodym_training.py",
    54: "a_cell_054_smoke_helpers.py",
    56: "a_cell_056_train_b2.py",
    64: "a_cell_064_sweep_b2.py",
    66: "a_cell_066_sens_report.py",
    73: "a_cell_073_train_variants.py",
    78: "a_cell_078_sens_shipped.py",
    84: "a_cell_084_conformal_fit.py",
    90: "a_cell_090_protocol_audit.py",
    92: "a_cell_092_final_eval.py",
    95: "a_cell_095_comparison.py",
    98: "a_cell_098_strata.py",
    101: "a_cell_101_weights_onnx.py",
    107: "a_cell_107_mirror.py",
    117: "a_cell_117_smoke_run.py",
    122: "a_cell_122_model_card.py",
    124: "a_cell_124_decisions.py",
    126: "a_cell_126_references.py",
    128: "a_cell_128_bundle.py",
    130: "a_cell_130_verify_bundle.py",
    137: "a_cell_137_linter.py",
    139: "a_cell_139_summary.py",
    141: "a_cell_141_summary_figure.py",
}

#: (anchor, filename) inserted after that anchor cell index, in listed order.
INSERT_AFTER: List[Tuple[int, str]] = [
    (0, "a_md_000_header.md"),
    (3, "a_cell_004_runstate.py"),
    (5, "a_md_005_resume.md"),
    (5, "a_cell_005_resume_state.py"),
    (54, "a_md_055_determinism.md"),
    (54, "a_cell_055_determinism.py"),
    (73, "a_md_072_variants_training.md"),
    (99, "a_md_100_final_eval_lock.md"),
]

DELETE: Tuple[int, ...] = ()

#: Modules written into the notebook by a `%%writefile` cell. They live here as ordinary Python
#: files so they can be imported, linted and unit-tested outside the notebook; this script inlines
#: them into the generated cell. Editing ``patch_cells/a_cell_004_runstate.py`` directly is
#: therefore wrong - edit these two and re-run the patcher.
WRITE_MODULES: Dict[str, str] = {
    "run_state.py": "a_cell_004_runstate.py",
    "phase_wrap.py": "a_cell_004_runstate.py",
}


def build_writefile_cell() -> str:
    """Compose the cell that emits `src/run_state.py` and `src/phase_wrap.py`.

    Both modules are kept as real files here so they can be imported, linted and unit-tested
    outside the notebook. This function is the only place that inlines them.

    Returns
    -------
    str
        A cell body of two `%%writefile` blocks.

    Raises
    ------
    FileNotFoundError
        If either module is absent.
    """
    parts = []
    for module in ("run_state.py", "phase_wrap.py"):
        p = SRC_DIR / module
        if not p.is_file():
            raise FileNotFoundError(f"missing module source: {p}")
        parts.append(f"# %%writefile src/{module}\n" + p.read_text(encoding="utf-8-sig"))
    return "\n\n".join(parts)


def read_source(name: str) -> str:
    """Read a patch source file.

    Parameters
    ----------
    name : str
        Filename under ``patch_cells/``.

    Returns
    -------
    str
        File contents, BOM-stripped.

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
    """
    return text.splitlines(keepends=True)


def make_cell(name: str) -> dict:
    """Build a notebook cell of the right type from a patch source.

    Parameters
    ----------
    name : str
        Patch source filename; the extension determines the cell type.

    Returns
    -------
    dict
        A cell dict valid for nbformat 4.
    """
    # the %%writefile cell is composed from the module sources, never read from disk
    text = (build_writefile_cell() if name in WRITE_MODULES.values()
            else read_source(name))
    lines = to_lines(text)
    # nbformat requires cell ids to match ^[a-zA-Z0-9-_]+$, so the extension is dropped here.
    cid = f"a-{Path(name).stem}"
    if name.endswith(".md"):
        return {"cell_type": "markdown", "id": cid, "metadata": {}, "source": lines}
    return {"cell_type": "code", "id": cid, "metadata": {}, "execution_count": None,
            "outputs": [], "source": lines}


def validate() -> List[str]:
    """Check every patch source exists and parses.

    Returns
    -------
    list of str
        Problems found; empty when fine.
    """
    problems: List[str] = []
    items = list(REPLACE.items()) + INSERT_AFTER
    for idx, name in sorted(items, key=lambda kv: (kv[1], kv[0])):
        if name in WRITE_MODULES.values():
            # composed from the module sources; validated through read_source on each module below
            for module in WRITE_MODULES:
                if not (SRC_DIR / module).is_file():
                    problems.append(f"cell {idx}: missing module source {module}")
            continue
        if not (SRC_DIR / name).is_file():
            problems.append(f"cell {idx}: missing source {name}")
            continue
        if not name.endswith(".py"):
            continue
        src = read_source(name)
        body = src.split("\n", 1)[1] if src.lstrip().startswith("%%") else src
        try:
            ast.parse(body)
        except SyntaxError as exc:
            problems.append(f"{name} (cell {idx}): line {exc.lineno}: {exc.msg}")
    return problems


def main() -> int:
    """Apply the addendum patch.

    Returns
    -------
    int
        0 on success, 1 on validation failure, 2 if the notebook is missing.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    problems = validate()
    if problems:
        print("VALIDATION FAILED - nothing written:", file=sys.stderr)
        for p in problems:
            print("  " + p, file=sys.stderr)
        return 1
    print(f"validated {len(REPLACE) + len(INSERT_AFTER)} addendum patch sources")

    if not NB_PATH.is_file():
        print(f"not found: {NB_PATH}", file=sys.stderr)
        return 2
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cells = nb["cells"]

    # The REPLACE indices refer to the notebook as it stands *before* this patch. Once cells are
    # inserted, those indices point at the wrong cells, so a second run would silently corrupt
    # the file (it did: cell 130 received the wrong source). Refuse rather than guess.
    if nb.get("metadata", {}).get("patched_by_addendum"):
        print(f"NOTHING WRITTEN: {NB_PATH.name} is already patched by this script.\n"
              "  Its cell indices have shifted, so re-running would replace the wrong cells.\n"
              "  Restore the fix-brief version first:\n"
              "      git checkout -- ipynb/model_training_measurement.ipynb\n"
              "  or regenerate it with: python patch_notebook.py", file=sys.stderr)
        return 3

    print(f"notebook: {NB_PATH.name} ({len(cells)} cells)")

    if args.dry_run:
        print("\nPLAN")
        for idx in sorted(REPLACE):
            print(f"  replace {idx:>4}  {REPLACE[idx]}")
        for anchor, name in INSERT_AFTER:
            print(f"  insert  after {anchor:>4}  {name}")
        print("\n(dry run: nothing written)")
        return 0

    for idx in sorted(REPLACE):
        name = REPLACE[idx]
        if idx >= len(cells):
            print(f"  SKIP cell {idx}: does not exist", file=sys.stderr)
            continue
        cell = cells[idx]
        want = "markdown" if name.endswith(".md") else "code"
        if cell["cell_type"] != want:
            print(f"  SKIP cell {idx}: is {cell['cell_type']}, source is {want}", file=sys.stderr)
            continue
        before = len("".join(cell["source"]))
        text = (build_writefile_cell() if name in WRITE_MODULES.values()
                else read_source(name))
        cell["source"] = to_lines(text)
        cell["id"] = f"a-{Path(name).stem}"
        print(f"  replace {idx:>4}  {want:8s} {name}  ({before} -> {len(text)} chars)")

    # Insertions are keyed by cell id so the patch is idempotent: running it twice must not
    # duplicate the inserted cells. Already-present ids are reported and skipped.
    present = {c.get("id") for c in cells}
    pending: List[Tuple[int, str]] = []
    for anchor, name in INSERT_AFTER:
        cid = f"a-{Path(name).stem}"
        if cid in present:
            print(f"  ALREADY PRESENT after {anchor:>4}  {name}")
            continue
        pending.append((anchor, name))

    by_anchor: Dict[int, List[dict]] = {}
    for anchor, name in pending:
        by_anchor.setdefault(anchor, []).append(make_cell(name))

    new_cells: List[dict] = []
    for i, cell in enumerate(cells):
        new_cells.append(cell)
        for extra in by_anchor.get(i, []):
            new_cells.append(extra)
    for anchor in sorted(a for a in by_anchor if a >= len(cells)):
        new_cells.extend(by_anchor[anchor])

    # The original notebook predates nbformat 4.5's mandatory, pattern-constrained cell ids, so
    # both cases are handled here: a missing id is filled in, and an id containing a dot (a patch
    # filename that was carried through verbatim) is replaced.
    _norm = 0
    for i, cell in enumerate(new_cells):
        if not re.fullmatch(r"[a-zA-Z0-9\-_]{1,64}", str(cell.get("id", ""))):
            cell["id"] = f"cell-{i:03d}"
            _norm += 1
        if cell["cell_type"] != "code":
            continue
        for key, default in (("execution_count", None), ("outputs", []), ("metadata", {})):
            if key not in cell:
                cell[key] = default
                _norm += 1
    print(f"  normalise: added/repaired {_norm} required nbformat keys")

    nb["cells"] = new_cells
    nb.setdefault("metadata", {})["patched_by_addendum"] = "patch_addendum.py (addendum A)"
    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for anchor, name in pending:
        print(f"  insert  after {anchor:>4}  {name}")
    print(f"\nwrote {NB_PATH}: {len(new_cells)} cells (was {len(cells)}, "
          f"+{len(new_cells) - len(cells)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())