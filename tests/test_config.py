"""Configuration loading."""

from pathlib import Path

from bodymeasure.config import load_app_config, load_calibration, load_schema

EXPECTED_KEYS = 26


def test_app_config_loads() -> None:
    config = load_app_config()
    assert config.camera.rotation == 90
    assert config.models.seg.endswith(".pt")
    assert config.thresholds["keypoint_conf"] == 0.5


def test_schema_has_the_26_garmentcode_keys() -> None:
    schema = load_schema()
    assert len(schema) == EXPECTED_KEYS
    assert set(schema) >= {"bust", "waist", "hips", "bust_points", "bum_points"}
    for key, entry in schema.items():
        assert entry["unit"] in {"cm", "deg"}, key
        assert entry["confidence"] in {"high", "medium", "low", "estimated"}, key


def test_missing_calibration_is_none(tmp_path: Path) -> None:
    assert load_calibration(tmp_path / "calibration.yaml") is None
