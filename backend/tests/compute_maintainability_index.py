"""Compute an SLOC-weighted mean of file-level Maintainability Index scores."""
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import subprocess
import sys
import tokenize
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import radon
from radon.metrics import mi_visit
from radon.raw import analyze

BACKEND_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = Path(__file__).resolve().with_name("maintainability_manifest_design2.json")
RESULTS_DIR = BACKEND_ROOT / "results" / "maintainability_index"


class AnalysisError(Exception):
    pass


class ManifestError(Exception):
    pass


def _fail(path: str, reason: str) -> None:
    raise AnalysisError(f"{path}: {reason}")


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise ManifestError(f"{manifest_path}: manifest file is missing")
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ManifestError(f"{manifest_path}: cannot read/parse manifest ({exc})") from exc
    if not isinstance(data, dict):
        raise ManifestError(f"{manifest_path}: manifest must be a JSON object")
    if not isinstance(data.get("design"), str) or not data["design"].strip():
        raise ManifestError(f"{manifest_path}: missing non-empty 'design'")
    included = data.get("included")
    if not isinstance(included, list) or not included:
        raise ManifestError(f"{manifest_path}: 'included' must be a non-empty array")
    for item in included:
        if not isinstance(item, dict):
            raise ManifestError(f"{manifest_path}: included entry must be an object")
        if not isinstance(item.get("path"), str) or not item["path"].strip():
            raise ManifestError(f"{manifest_path}: included entry missing 'path'")
        if not isinstance(item.get("reason"), str) or not item["reason"].strip():
            raise ManifestError(f"{manifest_path}: included entry missing 'reason' for {item.get('path')}")
    if "exclusions" in data and not isinstance(data["exclusions"], list):
        raise ManifestError(f"{manifest_path}: 'exclusions' must be an array")
    return data


def _within_boundary(resolved: Path, backend_root: Path) -> bool:
    try:
        resolved.relative_to(backend_root.resolve())
        return True
    except ValueError:
        return False


def resolve_selected(manifest: dict[str, Any], backend_root: Path) -> list[dict[str, str]]:
    backend_root = backend_root.resolve()
    seen: set[Path] = set()
    selected: list[dict[str, str]] = []
    for item in manifest["included"]:
        rel = item["path"].strip().replace("\\", "/")
        p = Path(rel)
        if p.is_absolute():
            _fail(rel, "path must be relative to the backend root")
        resolved = (backend_root / p).resolve()
        if not _within_boundary(resolved, backend_root):
            _fail(rel, "path escapes the backend repository boundary")
        if not resolved.exists():
            _fail(rel, "file does not exist")
        if not resolved.is_file():
            _fail(rel, "not a regular file")
        if resolved.suffix != ".py":
            _fail(rel, "not a Python source file")
        if resolved in seen:
            _fail(rel, "duplicate selected path after resolution")
        seen.add(resolved)
        selected.append({"path": rel, "reason": item["reason"].strip()})
    if not selected:
        raise ManifestError("empty eligible file set")
    return selected


def measure_file(rel_path: str, backend_root: Path) -> tuple[int, float]:
    full = (backend_root / rel_path).resolve()
    try:
        with tokenize.open(full) as fh:
            source = fh.read()
    except Exception as exc:
        _fail(rel_path, f"encoding or file-reading failure ({exc})")

    try:
        ast.parse(source, filename=rel_path)
    except SyntaxError as exc:
        _fail(rel_path, f"invalid Python source ({exc.msg} at line {exc.lineno})")

    try:
        raw = analyze(source)
    except Exception as exc:
        _fail(rel_path, f"Radon raw analysis error ({exc})")
    try:
        sloc = int(raw.sloc)
    except (AttributeError, TypeError, ValueError) as exc:
        _fail(rel_path, f"invalid SLOC result ({exc})")
    if sloc < 0 or not math.isfinite(sloc):
        _fail(rel_path, "nonfinite or negative SLOC")

    try:
        mi = float(mi_visit(source, multi=True))
    except Exception as exc:
        _fail(rel_path, f"Radon MI analysis error ({exc})")
    if not math.isfinite(mi):
        _fail(rel_path, "nonfinite MI value")

    return sloc, mi


def collect_rows(selected: list[dict[str, str]], backend_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in selected:
        sloc, mi = measure_file(item["path"], backend_root)
        rows.append(
            {
                "path": item["path"],
                "reason": item["reason"],
                "sloc": sloc,
                "mi": mi,
            }
        )
    return rows


def finalize_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int, float, float, list[str]]:
    if not rows:
        raise AnalysisError("manifest: empty eligible file set")
    total_sloc = sum(r["sloc"] for r in rows)
    eligible = [r for r in rows if r["sloc"] > 0]
    if not eligible:
        raise AnalysisError("manifest: empty eligible file set (all files have zero SLOC)")
    if total_sloc <= 0:
        raise AnalysisError("manifest: total included SLOC is zero")
    weighted = sum(r["sloc"] * r["mi"] for r in eligible) / total_sloc
    unweighted = sum(r["mi"] for r in eligible) / len(eligible)
    zero_paths = [r["path"] for r in rows if r["sloc"] == 0]
    for r in rows:
        if r["sloc"] > 0:
            r["weight"] = r["sloc"] / total_sloc
            r["contribution"] = r["sloc"] * r["mi"] / total_sloc
        else:
            r["weight"] = 0.0
            r["contribution"] = 0.0
    if not math.isfinite(weighted) or not math.isfinite(unweighted):
        raise AnalysisError("manifest: nonfinite weighted result")
    return rows, total_sloc, weighted, unweighted, zero_paths


def git_info(repo_root: Path) -> tuple[Any, Any]:
    commit: Any = "unavailable"
    dirty: Any = "unavailable"
    try:
        res = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if res.returncode == 0 and res.stdout.strip():
            commit = res.stdout.strip()
    except Exception:
        commit = "unavailable"
    try:
        res = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if res.returncode == 0:
            dirty = bool(res.stdout.strip())
    except Exception:
        dirty = "unavailable"
    return commit, dirty


def write_csv(path: Path, design: str, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "Design name",
        "Relative file path",
        "Inclusion reason",
        "SLOC",
        "File-level MI",
        "SLOC weight",
        "Contribution to weighted mean",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(rows, key=lambda x: x["path"]):
            writer.writerow(
                {
                    "Design name": design,
                    "Relative file path": r["path"],
                    "Inclusion reason": r["reason"],
                    "SLOC": r["sloc"],
                    "File-level MI": repr(r["mi"]),
                    "SLOC weight": repr(r["weight"]),
                    "Contribution to weighted mean": repr(r["contribution"]),
                }
            )


def write_json(path: Path, design: str, manifest_name: str, rows, total_sloc, weighted, unweighted,
               zero_paths, manifest_exclusions, commit, dirty) -> None:
    all_exclusions = list(manifest_exclusions or []) + [
        {"path": p, "reason": "zero SLOC"} for p in zero_paths
    ]
    payload = {
        "Design name": design,
        "Included-file count": len(rows),
        "Total included SLOC": total_sloc,
        "Weighted mean MI": weighted,
        "Unweighted mean MI": unweighted,
        "Radon version": radon.__version__,
        "Python version": sys.version,
        "Multiline-string setting": True,
        "Measurement timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "Git commit": commit,
        "Working-tree dirty status": dirty,
        "Manifest used": manifest_name,
        "Exclusions": all_exclusions,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run(manifest_path: Path, backend_root: Path | None = None, results_dir: Path | None = None) -> dict[str, Any]:
    backend_root = (backend_root or BACKEND_ROOT).resolve()
    manifest = load_manifest(manifest_path)
    design = manifest["design"]
    selected = resolve_selected(manifest, backend_root)
    rows = collect_rows(selected, backend_root)
    rows, total_sloc, weighted, unweighted, zero_paths = finalize_rows(rows)

    rows_sorted = sorted(rows, key=lambda r: r["path"])
    print(f"{'File':<50}  {'SLOC':>6}  {'MI':>12}  {'SLOC weight (%)':>16}")
    for r in rows_sorted:
        print(f"{r['path']:<50}  {r['sloc']:>6}  {r['mi']:>12.6f}  {r['weight'] * 100:>16.2f}")
    print()
    print(f"Design: {design}")
    print(f"Included files: {len(rows)}")
    print(f"Total included SLOC: {total_sloc}")
    print(f"SLOC-weighted mean of file-level MI: {weighted:.2f}")
    print(f"Unweighted mean of file-level MI: {unweighted}")
    print(f"Radon version: {radon.__version__}")
    print("Multiline-string setting: multi=True")
    repo_root = backend_root.parent
    commit, dirty = git_info(repo_root)
    print(f"Source revision: {commit}")

    out_dir = (results_dir or RESULTS_DIR).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = "design2"
    csv_path = out_dir / f"{slug}_per_file.csv"
    json_path = out_dir / f"{slug}_summary.json"
    write_csv(csv_path, design, rows_sorted)
    manifest_exclusions = manifest.get("exclusions", []) or []
    write_json(json_path, design, manifest_path.name, rows_sorted, total_sloc, weighted, unweighted,
               zero_paths, manifest_exclusions, commit, dirty)
    return {
        "design": design,
        "rows": rows_sorted,
        "total_sloc": total_sloc,
        "weighted": weighted,
        "unweighted": unweighted,
        "csv_path": csv_path,
        "json_path": json_path,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute SLOC-weighted Maintainability Index.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--backend-root", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    manifest_path = args.manifest
    if not manifest_path.is_absolute():
        manifest_path = (Path.cwd() / manifest_path).resolve()
    try:
        run(manifest_path, args.backend_root, args.results_dir)
    except (AnalysisError, ManifestError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
