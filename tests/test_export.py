import json

import pytest

from pattern_drafter.drafting import extract_pattern_data
from pattern_drafter.export import export_json, export_pdf
from stubs import FakeDrawPattern, FakePattern

CONFIG = {
    "template": "bodice",
    "style": "Gilewska",
    "gender": "w",
    "profile_name": "unit",
    "measurements": {"tour_taille": 62.0},
    "source": "unit.yaml",
}


def test_extract_pattern_data_panels_and_points():
    data = extract_pattern_data(FakePattern(), CONFIG)

    assert [p["name"] for p in data["panels"]] == ["front", "back"]
    assert data["panels"][0]["vertices"][0] == [0.0, 0.0]

    front = data["points"]["front"]
    assert front["points"]["A"] == [0.0, 0.0]
    assert "Front_Armhole" in front["curves"]
    assert front["curves"]["Front_Armhole"][1] == [5.0, 5.0]

    assert data["bounds"] == {"xmin": 0.0, "xmax": 30.0, "ymin": 0.0, "ymax": 20.0}


def test_extract_pattern_data_uses_resolved_measurements():
    data = extract_pattern_data(FakePattern(), CONFIG)
    assert data["input_measurements"] == {"tour_taille": 62.0}
    assert data["resolved_measurements"]["tour_bassin"] == 88.0


def test_export_json_round_trip(tmp_path):
    data = extract_pattern_data(FakePattern(), CONFIG)
    path = export_json(data, tmp_path / "unit.json")

    written = json.loads(open(path, encoding="utf-8").read())
    assert written["schema_version"] == "1.0"
    assert written["panels"][1]["name"] == "back"


def test_export_pdf_writes_file(tmp_path):
    pytest.importorskip("matplotlib")
    data = extract_pattern_data(FakeDrawPattern(), CONFIG)
    path = export_pdf(FakeDrawPattern(), data, tmp_path / "unit.pdf")

    assert path.endswith("unit.pdf")
    assert (tmp_path / "unit.pdf").stat().st_size > 0
