"""The provisioning and compatibility tools, and the constraints behind them."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

from bodymeasure import vision
from bodymeasure.config import ROOT, load_app_config, load_schema


def _load(name: str) -> object:
    """Import one of the scripts in tools/ by path, as a module."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_a_weight_file_hash_is_computed_over_the_whole_file(tmp_path: Path) -> None:
    path = tmp_path / "weights.pt"
    path.write_bytes(b"a" * (1 << 21))  # larger than one read block

    import hashlib

    assert vision.file_sha256(path) == hashlib.sha256(b"a" * (1 << 21)).hexdigest()


def test_a_weight_file_that_is_not_the_configured_one_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Brief 5: the results were recorded against one weight file, not any file."""
    models = load_app_config().models
    for name in ("seg", "pose"):
        (tmp_path / f"{name}.pt").write_bytes(b"not the real weights")
    monkeypatch.setattr(vision, "ROOT", tmp_path)
    monkeypatch.setattr(models, "seg", "seg.pt")
    monkeypatch.setattr(models, "pose", "pose.pt")
    monkeypatch.setattr(models, "seg_sha256", "0" * 64)

    with pytest.raises(ValueError, match="not the configured weight file"):
        vision.verify_weights()

    monkeypatch.setattr(models, "seg_sha256", vision.file_sha256(tmp_path / "seg.pt"))
    assert set(vision.verify_weights()) == {"seg", "pose"}


def test_a_missing_weight_file_is_never_downloaded_on_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models = load_app_config().models
    monkeypatch.setattr(vision, "ROOT", tmp_path)
    monkeypatch.setattr(models, "seg", "nowhere/seg.pt")
    with pytest.raises(FileNotFoundError, match="model weights missing"):
        vision.verify_weights()


def test_the_provisioning_script_reports_what_is_missing(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    provision = _load("provision_models")
    monkeypatch.setattr(
        provision, "model_paths", lambda: (tmp_path / "seg.pt", tmp_path / "pose.pt")
    )

    assert provision.main([]) == 1
    assert "MISSING" in capsys.readouterr().out

    for name in ("seg", "pose"):
        (tmp_path / f"{name}.pt").write_bytes(b"weights")
    assert provision.main(["--write-hashes"]) == 0
    out = capsys.readouterr().out
    assert "no hash configured" in out
    assert "seg_sha256:" in out


def test_the_compatibility_script_reports_the_exact_difference() -> None:
    """Brief 3.2: report the difference, never rename a key to hide it."""
    verify = _load("verify_garmentcode")
    unexpected, wanted = verify.compare({"bust", "waist"}, {"waist", "hips"})

    assert unexpected == {"bust"}
    assert wanted == {"hips"}
    assert verify.compare(set(load_schema()), set(load_schema())) == (set(), set())


def test_a_missing_pygarment_is_reported_as_unchecked_not_as_passing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    verify = _load("verify_garmentcode")
    if verify.pinned_keys() is not None:  # pragma: no cover - only where it is installed
        pytest.skip("pygarment is installed, so the unchecked path cannot be exercised")

    assert verify.main([]) == 2
    assert "could NOT be made" in capsys.readouterr().out


def test_an_exported_file_is_checked_against_the_schema(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    verify = _load("verify_garmentcode")
    path = tmp_path / "body_measurements.yaml"
    body = dict.fromkeys(load_schema(), 1.0)
    body["bust_points"] = None
    path.write_text(yaml.safe_dump({"metadata": {}, "body": body}), encoding="utf-8")

    verify.main([str(path)])
    assert "the file matches the schema; 1 key(s) are null" in capsys.readouterr().out

    del body["waist"]
    path.write_text(yaml.safe_dump({"body": body}), encoding="utf-8")
    assert verify.main([str(path)]) == 1
    assert "missing ['waist']" in capsys.readouterr().out
