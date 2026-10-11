from __future__ import annotations

from pathlib import Path

import pytest

from app.drafting.ease import Settings
from app.drafting.measurements import parse_source
from app.drafting.pipeline import draft
from app.drafting.semantics import registry, resolve

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


def yaml_text(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def semantics_registry():
    return registry()


@pytest.fixture(scope="session")
def default_confirmations(semantics_registry) -> dict[str, str]:
    """The interpretation an operator picks for every ambiguous field."""
    return {
        name: f.candidates[0].id
        for name, f in semantics_registry.fields.items()
        if f.status == "ambiguous"
    }


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
def basic_result(basic_resolution):
    return draft(basic_resolution, Settings())


@pytest.fixture(scope="session")
def reference_fixture() -> dict:
    import json
    return json.loads((FIXTURES / "garmageset-reference.json").read_text(encoding="utf-8"))
