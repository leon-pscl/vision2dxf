import builtins
import os
import sys
import types
import warnings

import pytest

from pattern_drafter import drafting
from pattern_drafter.drafting import (
    _import_openpattern,
    _install_pkg_resources_shim,
    OpenPatternMissing,
)


@pytest.fixture
def pkg_resources_state(monkeypatch):
    """Snapshot / restore ``pkg_resources`` on ``sys.modules`` around a test."""
    saved = sys.modules.pop("pkg_resources", None)
    yield
    if saved is None:
        sys.modules.pop("pkg_resources", None)
    else:
        sys.modules["pkg_resources"] = saved


@pytest.fixture
def block_import():
    """Return a helper raising a given exception for a named import."""

    def install(monkeypatch, blocking, exc):
        real_import = builtins.__import__

        def wrapped(name, *args, **kwargs):
            if name == blocking:
                raise exc
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", wrapped)

    return install


def test_shim_installed_when_pkg_resources_missing(
    pkg_resources_state, block_import, monkeypatch
):
    block = ModuleNotFoundError("No module named 'pkg_resources'")
    block_import(monkeypatch, "pkg_resources", block)

    _install_pkg_resources_shim()

    shim = sys.modules.get("pkg_resources")
    assert shim is not None
    assert callable(getattr(shim, "resource_filename", None))


def test_shim_resource_filename_resolves_package_file(
    pkg_resources_state, block_import, monkeypatch
):
    block = ModuleNotFoundError("No module named 'pkg_resources'")
    block_import(monkeypatch, "pkg_resources", block)

    _install_pkg_resources_shim()

    path = sys.modules["pkg_resources"].resource_filename(
        "pattern_drafter", "measurements.py"
    )
    assert isinstance(path, str)
    assert path.endswith(os.path.join("pattern_drafter", "measurements.py"))
    assert os.path.isfile(path)


def test_shim_preserves_existing_module(pkg_resources_state):
    sentinel = types.ModuleType("pkg_resources")
    sys.modules["pkg_resources"] = sentinel
    _install_pkg_resources_shim()
    assert sys.modules["pkg_resources"] is sentinel


def test_import_error_is_transparent(block_import, monkeypatch):
    block_import(
        monkeypatch,
        "OpenPattern",
        RuntimeError("boom: dependency exploded"),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(OpenPatternMissing) as excinfo:
            _import_openpattern()
    message = str(excinfo.value)
    assert "Install it with" in message
    assert "pip install git+https://github.com/fmetivier/OpenPattern.git" in message
    assert "RuntimeError: boom: dependency exploded" in message