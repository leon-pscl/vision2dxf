import json

import pytest

from conftest import EXAMPLES_DIR
from pattern_drafter import pipeline
from stubs import FakePattern


@pytest.fixture
def fake_draft(monkeypatch):
    monkeypatch.setattr(
        pipeline, "draft_pattern", lambda config: FakePattern()
    )


def test_process_file_json(fake_draft, tmp_path):
    result = pipeline.process_file(
        EXAMPLES_DIR / "skirt.yaml", tmp_path, formats=("json",)
    )

    assert result.json_path is not None and result.json_path.exists()
    assert result.pdf_path is None
    data = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert data["template"] == "skirt"
    assert result.config["profile_name"] == "skirt_gilewska_w36"


def test_process_file_json_and_pdf(fake_draft, tmp_path, monkeypatch):
    written = {}

    def fake_export_pdf(pattern, data, path, info=None):
        path = str(path)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("%PDF-1.4 stub")
        written["path"] = path
        return path

    monkeypatch.setattr(pipeline, "export_pdf", fake_export_pdf)

    result = pipeline.process_file(
        EXAMPLES_DIR / "bodice.yaml", tmp_path, formats=("json", "pdf")
    )

    assert result.json_path.exists()
    assert result.pdf_path.exists()
    assert written["path"] == str(result.pdf_path)
    assert set(result.written) == {result.json_path, result.pdf_path}


def test_process_file_reports_missing(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "template: skirt\nmeasurements:\n  tour_taille: 62\n",
        encoding="utf-8",
    )
    with pytest.raises(Exception) as excinfo:
        pipeline.process_file(bad, tmp_path)
    assert "tour_bassin" in str(excinfo.value)


def test_iter_yaml_files_directory():
    files = pipeline.iter_yaml_files([str(EXAMPLES_DIR)])
    assert len(files) == 3
