from __future__ import annotations

from pathlib import Path

import pytest

from app.base_block.loader import load_block, resolve_block
from app.base_block.measurements import default_confirmations as schema_confirmations
from app.drafting.ease import Settings
from app.drafting.measurements import parse_source
from app.drafting.semantics import registry, resolve
from app.transformation.pipeline import resolve_semantics, transform

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"

# The blocks the catalog offers, and the profiles the golden results cover: a zero-delta
# target that must reproduce the block it came from, and two targets whose differences
# run in opposite directions.
BASE_BLOCKS = ("bodice-ref-a", "bodice-ref-b", "bodice-ref-c")
TARGET_PROFILES = {
    "zero_delta": "body_measurements.yaml",
    "target_a": "body_measurements_target_a.yaml",
    "target_b": "body_measurements_target_b.yaml",
}


def yaml_text(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def semantics_registry():
    return registry()


@pytest.fixture(scope="session")
def default_confirmations() -> dict[str, str]:
    """The interpretation an operator picks for every ambiguous field."""
    return schema_confirmations()


@pytest.fixture(scope="session")
def drafting_config() -> dict:
    from app.config import drafting_config as load

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
def reference_fixture() -> dict:
    import json
    return json.loads((FIXTURES / "garmageset-reference.json").read_text(encoding="utf-8"))


# ---- base blocks and transformation ----------------------------------------

@pytest.fixture(scope="session")
def base_block():
    """The catalog's default block."""
    return resolve_block(None)


@pytest.fixture(scope="session")
def base_blocks() -> dict:
    return {block_id: load_block(block_id) for block_id in BASE_BLOCKS}


@pytest.fixture(scope="session")
def base_confirmations(default_confirmations) -> dict[str, str]:
    return dict(default_confirmations)


def transform_profile(filename: str, confirmations: dict[str, str] | None = None,
                      **settings_kwargs):
    confirmations = confirmations or schema_confirmations()
    _profile, resolution = resolve_semantics(yaml_text(filename), filename, confirmations)
    return transform(resolution, Settings(**settings_kwargs))


@pytest.fixture(scope="session")
def transformed() -> dict:
    """Every profile on every catalog block: 3 x 3 results, built once."""
    return {
        (name, block_id): transform_profile(filename, base_block_id=block_id)
        for name, filename in TARGET_PROFILES.items()
        for block_id in BASE_BLOCKS
    }


@pytest.fixture(scope="session")
def basic_result(transformed):
    """One representative transformed pattern, for the export and cutting tests."""
    return transformed[("target_a", "bodice-ref-a")]


@pytest.fixture(scope="session")
def zero_delta(transformed):
    return {block_id: transformed[("zero_delta", block_id)] for block_id in BASE_BLOCKS}


@pytest.fixture(scope="session")
def target_a(transformed):
    return transformed[("target_a", "bodice-ref-a")]
