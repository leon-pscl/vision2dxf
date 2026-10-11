import pytest

from pattern_drafter.measurements import MeasurementError
from pattern_drafter.vision import (
    BODY_TO_OPENPATTERN,
    build_config,
    parse_body_measurements,
    reference_profiles,
)

VISION_YAML = """
body:
  height: 167.0
  bust: 85.8
  underbust: 85.8
  waist: 85.6
  hips: 100.5
  arm_length: 62.6
  armscye_depth: 20.1
  wrist: 16.2
"""

#: The sample shipped with the web UI.
SAMPLE_YAML = """body:
  arm_length: 62.6
  armscye_depth: 20.1
  bust: 85.8
  height: 167.0
  hips: 100.5
  waist: 85.6
  wrist: 16.2
"""


def test_parse_body_section():
    body = parse_body_measurements(VISION_YAML)

    assert body["bust"] == 85.8
    assert body["underbust"] == 85.8  # kept even without an OpenPattern key
    assert body["height"] == 167.0


def test_parse_accepts_flat_and_measurements_sections():
    flat = parse_body_measurements("bust: 85.8\n")
    nested = parse_body_measurements("measurements:\n  bust: 85.8\n")
    assert flat == nested == {"bust": 85.8}


def test_parse_normalises_names():
    assert parse_body_measurements({"Bust Points": 19}) == {"bust_points": 19.0}


def test_parse_rejects_non_numeric_and_non_positive():
    with pytest.raises(MeasurementError):
        parse_body_measurements("body:\n  bust: wide\n")
    with pytest.raises(MeasurementError):
        parse_body_measurements("body:\n  bust: -1\n")
    with pytest.raises(MeasurementError):
        parse_body_measurements("body:\n  bust: true\n")


def test_parse_rejects_empty_input():
    with pytest.raises(MeasurementError):
        parse_body_measurements("")
    with pytest.raises(MeasurementError):
        parse_body_measurements("- 1\n- 2\n")


def test_build_config_maps_and_overrides_body_values():
    config = build_config(SAMPLE_YAML, "skirt", "Gilewska", "w")

    measurements = config["measurements"]
    assert measurements["tour_poitrine"] == 85.8
    assert measurements["tour_taille"] == 85.6
    assert measurements["tour_bassin"] == 100.5
    assert measurements["longueur_manche"] == 62.6
    assert measurements["profondeur_emmanchure"] == 20.1
    assert measurements["tour_poignet"] == 16.2
    assert "tour_encolure" in measurements  # estimated from the reference block
    assert "bust" not in measurements  # vision-only keys never reach OpenPattern


def test_build_config_satisfies_validation_for_every_garment():
    for garment, style, gender, darts in [
        ("bodice", "Gilewska", "w", True),
        ("bodice", "Chiappetta", "w", False),
        ("bodice", "Donnanno", "m", False),
        ("skirt", "Gilewska", "w", False),
        ("skirt", "Donnanno", "m", False),
        ("trousers", "Donnanno", "m", False),
    ]:
        config = build_config(VISION_YAML, garment, style, gender, darts=darts)
        assert config["measurements"]["tour_poitrine"] == 85.8
        assert config["measurements"]["tour_bassin"] == 100.5


def test_build_config_reports_what_it_cannot_fill_in():
    # No bundled profile records the bust height the Donnanno dart needs.
    with pytest.raises(MeasurementError) as excinfo:
        build_config(VISION_YAML, "bodice", "Donnanno", "w", darts=True)
    assert "hauteur_poitrine" in str(excinfo.value)


def test_build_config_records_provenance():
    config = build_config(SAMPLE_YAML, "skirt", "Gilewska", "w")

    reference = reference_profiles()[config["reference"]]
    body = parse_body_measurements(SAMPLE_YAML)
    assert set(BODY_TO_OPENPATTERN.values()) <= set(config["from_body"])
    # The reference block is rescaled around the subject's own girths.
    assert config["scale"] == pytest.approx(
        sum(
            body[english] / reference[french]
            for english, french in (
                ("bust", "tour_poitrine"),
                ("waist", "tour_taille"),
                ("hips", "tour_bassin"),
            )
        )
        / 3,
        rel=0.001,
    )
    # Keys nobody measured keep the (rescaled) reference value.
    assert config["measurements"]["ecart_poitrine"] == pytest.approx(
        reference["ecart_poitrine"] * config["scale"], abs=0.01
    )


def test_build_config_picks_the_closest_reference():
    small = build_config("body:\n  bust: 84\n  waist: 62\n  hips: 88\n", "skirt", "Gilewska", "w")
    large = build_config("body:\n  bust: 110\n  waist: 100\n  hips: 113\n", "skirt", "Gilewska", "w")

    assert small["scale"] == pytest.approx(1.0, abs=0.05)
    assert large["scale"] > small["scale"]


def test_build_config_rejects_unknown_garment_and_variant():
    with pytest.raises(MeasurementError):
        build_config(SAMPLE_YAML, "hat", "Gilewska", "w")
    with pytest.raises(MeasurementError):
        build_config(SAMPLE_YAML, "trousers", "Gilewska", "m")


def test_build_config_rejects_measurements_without_girths():
    with pytest.raises(MeasurementError) as excinfo:
        build_config("body:\n  wrist: 16.2\n", "bodice", "Gilewska", "w")
    assert "bust, waist or hips" in str(excinfo.value)