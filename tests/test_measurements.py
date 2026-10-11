import sqlite3
from pathlib import Path

import pytest

from conftest import EXAMPLES_DIR
from pattern_drafter.measurements import (
    MEASUREMENT_GLOSSARY,
    TEMPLATES,
    MeasurementError,
    load_measurement_file,
    missing_measurements,
    validate_measurement_file,
    write_measurement_db,
)

EXAMPLE_FILES = sorted(EXAMPLES_DIR.glob("*.yaml"))


def test_examples_exist():
    names = {path.name for path in EXAMPLE_FILES}
    assert {"bodice.yaml", "skirt.yaml", "trousers.yaml"} <= names


@pytest.mark.parametrize("path", EXAMPLE_FILES, ids=lambda p: p.name)
def test_example_files_are_valid(path):
    config = load_measurement_file(path)
    assert config["template"] in TEMPLATES
    assert config["measurements"]
    validate_measurement_file(config)  # raises on missing keys


@pytest.mark.parametrize("path", EXAMPLE_FILES, ids=lambda p: p.name)
def test_example_keys_have_glossary(path):
    config = load_measurement_file(path)
    assert set(config["measurements"]) <= set(MEASUREMENT_GLOSSARY)


def test_missing_top_level_key(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("template: bodice\n", encoding="utf-8")
    with pytest.raises(MeasurementError):
        load_measurement_file(bad)


def test_unknown_template(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "template: hat\nmeasurements:\n  tour_taille: 60\n", encoding="utf-8"
    )
    with pytest.raises(MeasurementError):
        load_measurement_file(bad)


def test_non_numeric_measurement(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "template: skirt\nmeasurements:\n  tour_taille: wide\n",
        encoding="utf-8",
    )
    with pytest.raises(MeasurementError):
        load_measurement_file(bad)


def test_validate_reports_missing_measurements():
    config = {
        "template": "skirt",
        "style": "Gilewska",
        "gender": "w",
        "profile_name": "incomplete",
        "measurements": {"tour_taille": 62.0},
    }
    with pytest.raises(MeasurementError) as excinfo:
        validate_measurement_file(config)
    assert "tour_bassin" in str(excinfo.value)


def test_trousers_one_of_length_group():
    base = {
        "tour_taille": 80.0,
        "tour_bassin": 89.0,
        "hauteur_bassin": 19.2,
        "hauteur_taille_genou": 58.0,
        "montant": 23.2,
    }
    assert missing_measurements("trousers", "Donnanno", "m", base)
    base["longueur_taille_terre"] = 104.0
    assert missing_measurements("trousers", "Donnanno", "m", base) == []


def test_write_measurement_db_round_trip(tmp_path):
    db_path = tmp_path / "measurements.db"
    measurements = {"tour_taille": 62.0, "tour_bassin": 88.0}
    write_measurement_db(measurements, "profile_x", db_path)

    conn = sqlite3.connect(db_path)
    try:
        rows = dict(
            conn.execute(
                "select mtype, mval from measurements where wkey = ?",
                ("profile_x",),
            ).fetchall()
        )
    finally:
        conn.close()

    assert rows == measurements


def test_write_measurement_db_replaces_previous(tmp_path):
    db_path = tmp_path / "measurements.db"
    write_measurement_db({"tour_taille": 62.0}, "p", db_path)
    write_measurement_db({"tour_bassin": 88.0}, "p", db_path)

    conn = sqlite3.connect(db_path)
    try:
        keys = {row[0] for row in conn.execute("select mtype from measurements")}
    finally:
        conn.close()

    assert keys == {"tour_bassin"}
