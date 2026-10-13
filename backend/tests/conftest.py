from __future__ import annotations

from pathlib import Path

import pytest

from app.drafting.ease import Settings
from app.drafting.measurements import parse_source
from app.drafting.semantics import default_confirmations as schema_confirmations
from app.drafting.semantics import registry, resolve
from app.pipeline import generate, resolve_semantics

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"

# The profiles the golden results cover. `pascual` is carried deliberately: it must load
# and display, and then be refused with its reason.
TARGET_PROFILES = {
    "basic": "body_measurements.yaml",
    "target_a": "body_measurements_target_a.yaml",
    "target_b": "body_measurements_target_b.yaml",
}
REFUSED_PROFILES = {"pascual": "body_measurements_pascual.yaml"}


def yaml_text(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def semantics_registry():
    return registry()


@pytest.fixture(scope="session")
def default_confirmations() -> dict[str, str]:
    """The interpretation the prototype assumes for every ambiguous field."""
    return schema_confirmations()


@pytest.fixture(scope="session")
def drafting_config() -> dict:
    from app.config import drafting_config as load

    return load()


@pytest.fixture(scope="session")
def reconstruction_config() -> dict:
    from app.config import reconstruction_config as load

    return load()


@pytest.fixture(scope="session")
def basic_source():
    return parse_source(yaml_text("body_measurements.yaml"))


@pytest.fixture(scope="session")
def pascual_source():
    return parse_source(yaml_text("body_measurements_pascual.yaml"))


@pytest.fixture(scope="session")
def basic_resolution(basic_source, default_confirmations):
    return resolve(basic_source.values_cm, default_confirmations)


@pytest.fixture(scope="session")
def pascual_resolution(pascual_source, default_confirmations):
    return resolve(pascual_source.values_cm, default_confirmations)


@pytest.fixture(scope="session")
def measurements(basic_resolution):
    return basic_resolution.measurements


@pytest.fixture(scope="session")
def reference_fixture() -> dict:
    import json

    return json.loads((FIXTURES / "garmageset-reference.json").read_text(encoding="utf-8"))


# ---- generation ------------------------------------------------------------


def settings(**kwargs) -> Settings:
    """Settings with the unit already confirmed, which every generation requires."""
    kwargs.setdefault("confirm_units", True)
    return Settings(**kwargs)


def generate_profile(filename: str, confirmations: dict[str, str] | None = None, **kwargs):
    confirmations = confirmations if confirmations is not None else schema_confirmations()
    profile, resolution = resolve_semantics(yaml_text(filename), filename, confirmations)
    return generate(resolution, profile, settings(**kwargs))


@pytest.fixture(scope="session")
def generated() -> dict:
    """Every usable profile, generated once."""
    return {name: generate_profile(f) for name, f in TARGET_PROFILES.items()}


@pytest.fixture(scope="session")
def basic_result(generated):
    return generated["basic"]
