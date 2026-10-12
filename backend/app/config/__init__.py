"""Config loading. Drafting constants, the measurement dictionary and the measurement
semantics registry ship as YAML so no unexplained constant lives inside a geometry
function and no field meaning is inferred from its name."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).parent


@lru_cache(maxsize=None)
def _load(name: str) -> dict:
    with open(CONFIG_DIR / name, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def drafting_config() -> dict:
    return _load("drafting.yaml")


def transformation_config() -> dict:
    """Weights, solver limits, envelope and influence map for the base-block transform."""
    return _load("transformation.yaml")


def measurement_dictionary() -> dict:
    return _load("measurements.yaml")


def semantics_config() -> dict:
    """Schema configuration for the camera measurement subsystem."""
    return _load("measurement-semantics.yaml")


def measurement_index() -> dict[str, dict]:
    return {f["name"]: f for f in measurement_dictionary()["fields"]}


def seam_allowance(role: str) -> float:
    table = drafting_config()["seam_allowance_mm"]
    return float(table.get(role, table["default"]))
