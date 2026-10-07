"""Load adapter classes from adapters.yaml and instantiate them once.

Adding or swapping a model is an adapters.yaml edit. Nothing here knows what a
segmenter is.
"""

from __future__ import annotations

import functools
import importlib
import os
import sys
from typing import Any

import yaml

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
ADAPTERS_YAML = os.path.join(BACKEND_DIR, "adapters.yaml")
CONFIG_YAML = os.path.join(BACKEND_DIR, "config.yaml")


def _load_yaml(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@functools.lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    return _load_yaml(CONFIG_YAML)


@functools.lru_cache(maxsize=1)
def load_adapters_yaml() -> dict[str, str]:
    return _load_yaml(ADAPTERS_YAML)


def _import(path: str):
    """Import a dotted path relative to backend/."""
    if BACKEND_DIR not in sys.path:
        sys.path.insert(0, BACKEND_DIR)
    module_name, _, class_name = path.rpartition(".")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


@functools.lru_cache(maxsize=None)
def get_adapter(key: str) -> Any:
    """Instantiate and cache the adapter registered under ``key``."""
    mapping = load_adapters_yaml()
    if key not in mapping:
        raise KeyError(f"no adapter registered for {key!r}; known: {sorted(mapping)}")
    return _import(mapping[key])()


def reset_registry() -> None:
    """Drop cached classes. Used by tests after rewriting adapters.yaml."""
    get_adapter.cache_clear()
    load_adapters_yaml.cache_clear()
