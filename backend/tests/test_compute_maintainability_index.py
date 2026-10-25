import json
import math
from pathlib import Path

import pytest

from compute_maintainability_index import (
    AnalysisError,
    ManifestError,
    collect_rows,
    finalize_rows,
    load_manifest,
    measure_file,
    resolve_selected,
    run,
)

BACKEND = Path(__file__).resolve().parents[1]


def _write_manifest(tmp_path: Path, included, exclusions=None, design="D") -> Path:
    data = {"design": design, "included": included, "exclusions": exclusions or []}
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_weighted_arithmetic_mocked():
    rows = [
        {"path": "a.py", "reason": "r", "sloc": 100, "mi": 20.0},
        {"path": "b.py", "reason": "r", "sloc": 300, "mi": 80.0},
    ]
    rows, total, weighted, unweighted, zero = finalize_rows(rows)
    assert total == 400
    assert weighted == 65.0
    assert unweighted == 50.0


def test_zero_sloc_no_effect():
    rows = [
        {"path": "a.py", "reason": "r", "sloc": 100, "mi": 20.0},
        {"path": "z.py", "reason": "r", "sloc": 0, "mi": 100.0},
        {"path": "b.py", "reason": "r", "sloc": 300, "mi": 80.0},
    ]
    rows, total, weighted, unweighted, zero = finalize_rows(rows)
    assert total == 400
    assert weighted == 65.0
    assert zero == ["z.py"]


def test_empty_set_fails():
    with pytest.raises(AnalysisError):
        finalize_rows([])


def test_zero_total_sloc_fails():
    rows = [{"path": "a.py", "reason": "r", "sloc": 0, "mi": 10.0}]
    with pytest.raises(AnalysisError):
        finalize_rows(rows)


def test_missing_file_fails(tmp_path):
    manifest = _write_manifest(tmp_path, [{"path": "missing.py", "reason": "r"}])
    (tmp_path / "backend").mkdir()
    backend = tmp_path / "backend"
    with pytest.raises(AnalysisError):
        run(manifest, backend_root=backend, results_dir=tmp_path / "res")


def test_duplicate_paths_fail(tmp_path):
    backend = tmp_path / "backend"
    backend.mkdir()
    (backend / "a.py").write_text("x=1\n", encoding="utf-8")
    manifest = _write_manifest(tmp_path, [
        {"path": "a.py", "reason": "r1"},
        {"path": "a.py", "reason": "r2"},
    ])
    with pytest.raises(AnalysisError):
        run(manifest, backend_root=backend, results_dir=tmp_path / "res")


def test_invalid_source_fails(tmp_path):
    backend = tmp_path / "backend"
    backend.mkdir()
    (backend / "bad.py").write_text("def (broken\n", encoding="utf-8")
    manifest = _write_manifest(tmp_path, [{"path": "bad.py", "reason": "r"}])
    with pytest.raises(AnalysisError):
        run(manifest, backend_root=backend, results_dir=tmp_path / "res")


def test_manifest_order_does_not_change(tmp_path, monkeypatch):
    backend = tmp_path / "backend"
    backend.mkdir()
    (backend / "a.py").write_text("x=1\n", encoding="utf-8")
    (backend / "b.py").write_text("y=2\n", encoding="utf-8")

    def fake_measure(rel_path, backend_root):
        return (100, 20.0) if rel_path.endswith("a.py") else (300, 80.0)

    monkeypatch.setattr("compute_maintainability_index.measure_file", fake_measure)

    manifest1 = _write_manifest(tmp_path, [
        {"path": "a.py", "reason": "r"},
        {"path": "b.py", "reason": "r"},
    ])
    r1 = run(manifest1, backend_root=backend, results_dir=tmp_path / "r1")

    manifest2 = _write_manifest(tmp_path, [
        {"path": "b.py", "reason": "r"},
        {"path": "a.py", "reason": "r"},
    ])
    r2 = run(manifest2, backend_root=backend, results_dir=tmp_path / "r2")

    assert r1["weighted"] == r2["weighted"] == 65.0


def test_file_scoring_100_included(tmp_path, monkeypatch):
    backend = tmp_path / "backend"
    backend.mkdir()
    (backend / "good.py").write_text("x=1\n", encoding="utf-8")

    def fake_measure(rel_path, backend_root):
        return (10, 100.0)

    monkeypatch.setattr("compute_maintainability_index.measure_file", fake_measure)
    manifest = _write_manifest(tmp_path, [{"path": "good.py", "reason": "r"}])
    result = run(manifest, backend_root=backend, results_dir=tmp_path / "res")
    assert result["weighted"] == 100.0
    assert result["rows"][0]["mi"] == 100.0


def test_csv_json_agree(tmp_path, monkeypatch):
    backend = tmp_path / "backend"
    backend.mkdir()
    (backend / "a.py").write_text("x=1\n", encoding="utf-8")

    def fake_measure(rel_path, backend_root):
        return (100, 20.0)

    monkeypatch.setattr("compute_maintainability_index.measure_file", fake_measure)
    manifest = _write_manifest(tmp_path, [{"path": "a.py", "reason": "r"}])
    result = run(manifest, backend_root=backend, results_dir=tmp_path / "res")
    import csv as _csv
    with result["csv_path"].open(newline="", encoding="utf-8") as fh:
        rows = list(_csv.DictReader(fh))
    assert len(rows) == 1
    json_data = json.loads(result["json_path"].read_text(encoding="utf-8"))
    assert json_data["Included-file count"] == 1
    assert json_data["Total included SLOC"] == 100
    assert math.isclose(json_data["Weighted mean MI"], result["weighted"])


def test_working_directory_independence(tmp_path, monkeypatch):
    backend = (tmp_path / "backend")
    backend.mkdir()
    (backend / "a.py").write_text("x=1\n", encoding="utf-8")

    def fake_measure(rel_path, backend_root):
        return (100, 20.0)

    monkeypatch.setattr("compute_maintainability_index.measure_file", fake_measure)

    manifest = _write_manifest(tmp_path, [{"path": "a.py", "reason": "r"}])
    # Run with a different cwd.
    import os
    old = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = run(manifest, backend_root=backend, results_dir=tmp_path / "res")
    finally:
        os.chdir(old)
    assert result["weighted"] == 20.0


def test_manifest_exclusions_recorded(tmp_path, monkeypatch):
    backend = tmp_path / "backend"
    backend.mkdir()
    (backend / "a.py").write_text("x=1\n", encoding="utf-8")

    def fake_measure(rel_path, backend_root):
        return (100, 20.0)

    monkeypatch.setattr("compute_maintainability_index.measure_file", fake_measure)
    manifest = _write_manifest(tmp_path, [{"path": "a.py", "reason": "r"}],
                               exclusions=[{"path": "tests", "reason": "excluded"}])
    result = run(manifest, backend_root=backend, results_dir=tmp_path / "res")
    import json as _json
    data = _json.loads(result["json_path"].read_text(encoding="utf-8"))
    assert any(e.get("path") == "tests" for e in data["Exclusions"])


def test_resolve_selected_missing(tmp_path):
    backend = tmp_path / "backend"
    backend.mkdir()
    manifest = {"design": "D", "included": [{"path": "nope.py", "reason": "r"}]}
    with pytest.raises(AnalysisError):
        resolve_selected(manifest, backend)


def test_load_manifest_missing(tmp_path):
    with pytest.raises(ManifestError):
        load_manifest(tmp_path / "does_not_exist.json")
