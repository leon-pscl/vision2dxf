from __future__ import annotations

from pathlib import Path

import pytest

from app.drafting.ease import Settings
from app.drafting.measurements import parse_measurements
from app.drafting.pipeline import draft

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


def yaml_text(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def basic_body():
    return parse_measurements(yaml_text("body_measurements.yaml"))


@pytest.fixture(scope="session")
def pascual_body():
    return parse_measurements(yaml_text("body_measurements_pascual.yaml"))


@pytest.fixture(scope="session")
def basic_result(basic_body):
    return draft(basic_body, Settings())


@pytest.fixture(scope="session")
def pascual_result(pascual_body):
    return draft(pascual_body, Settings())


@pytest.fixture(scope="session")
def reference_fixture() -> dict:
    import json
    return json.loads((FIXTURES / "garmageset-reference.json").read_text(encoding="utf-8"))
