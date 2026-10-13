"""Metric 5 - Radon Maintainability Index.

The index is computed over the frozen allow-list in `benchmark/config-design-2.yaml`, not
over the whole repository, because the protocol scores Design 2's own production modules
and excludes the components it shares with the other designs.

The reported value is the source-line-weighted mean:

    weighted_mi = sum(file_mi * file_sloc) / sum(file_sloc)

on radon's 0-100 scale. It is not divided by 100.

    python benchmark/maintainability_check.py

Every allow-listed module currently exists only as CPython 3.13 bytecode with no source
(RECONSTRUCTION.md), so this script stops rather than reporting a number over a shortened
list. Recover the source first - see benchmark/README.md.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import yaml
from radon.complexity import cc_rank, cc_visit
from radon.metrics import mi_visit
from radon.raw import analyze

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "benchmark" / "config-design-2.yaml"

FIELDS = [
    "path", "role", "maintainability_index", "rank", "sloc", "lloc", "loc", "comments",
    "blank", "cc_blocks", "cc_average", "cc_max", "cc_rank", "weighted_contribution",
    "inclusion_rationale", "commit_hash",
]


def analyse_file(root: Path, entry: dict) -> dict:
    """radon MI, raw counts and cyclomatic complexity for one allow-listed module."""
    path = root / entry["path"]
    source = path.read_text(encoding="utf-8")
    raw = analyze(source)
    blocks = cc_visit(source)
    complexities = [block.complexity for block in blocks] or [0]
    index = mi_visit(source, True)
    return {
        "path": entry["path"],
        "role": entry["role"],
        "maintainability_index": round(index, 6),
        "rank": _mi_rank(index),
        "sloc": raw.sloc,
        "lloc": raw.lloc,
        "loc": raw.loc,
        "comments": raw.comments,
        "blank": raw.blank,
        "cc_blocks": len(complexities),
        "cc_average": round(sum(complexities) / len(complexities), 4),
        "cc_max": max(complexities),
        "cc_rank": cc_rank(max(complexities)),
        "inclusion_rationale": entry["role"],
    }


def _mi_rank(value: float) -> str:
    """radon's MI band, on the same A-F scale `radon mi` prints."""
    if value < 10:
        return "E"
    if value < 20:
        return "D"
    if value < 40:
        return "C"
    if value < 60:
        return "B"
    return "A"


def measure(root: Path = ROOT, config: dict | None = None) -> tuple[float, list[dict]]:
    """The weighted MI and one row per file. Raises if the allow-list names a file that
    is not there: a silently shortened allow-list would report a flattering number."""
    config = config or yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    entries = config["maintainability"]["files"]

    missing = [e["path"] for e in entries if not (root / e["path"]).is_file()]
    if missing:
        raise FileNotFoundError(
            f"allow-listed module(s) have no readable source: {', '.join(missing)}\n"
            "Design 2's base_block and transformation packages currently exist only as "
            "CPython 3.13 bytecode (see RECONSTRUCTION.md). Recover the source first - "
            "benchmark/README.md, 'Step 0' - then rerun this."
        )

    rows = [analyse_file(root, entry) for entry in entries]
    total_sloc = sum(row["sloc"] for row in rows)
    for row in rows:
        row["weighted_contribution"] = round(
            row["maintainability_index"] * row["sloc"] / total_sloc, 6
        ) if total_sloc else 0.0
    weighted_mi = (
        sum(row["maintainability_index"] * row["sloc"] for row in rows) / total_sloc
        if total_sloc else 0.0
    )
    return weighted_mi, rows


def main() -> int:
    sys.path.insert(0, str(ROOT / "benchmark"))
    from environment import git_state

    weighted_mi, rows = measure()
    commit = git_state()["commitHash"]

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    outputs = config["outputs"]
    directory = ROOT / outputs["directory"]
    directory.mkdir(parents=True, exist_ok=True)

    with (directory / outputs["maintainability_csv"]).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "commit_hash": commit})

    print(json.dumps({
        "maintainability_index_0_100": round(weighted_mi, 6),
        "files": len(rows),
        "totalSloc": sum(row["sloc"] for row in rows),
        "csv": str((directory / outputs["maintainability_csv"]).relative_to(ROOT)),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())