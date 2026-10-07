"""The core rule: swapping a model is exactly one edit to adapters.yaml.

Rewrites adapters.yaml, clears the registry cache, and checks the new class is
in effect. Restores the file afterwards.
"""

from __future__ import annotations

import pytest

import registry
from conftest import run


@pytest.fixture
def swap():
    """Temporarily point a key at a different dotted path."""
    original = registry.ADAPTERS_YAML
    saved = open(original, encoding="utf-8").read()

    def _do(key: str, dotted_path: str):
        text = saved.replace(
            f"\n{key}: ", f"\n{key}: {dotted_path}  # swapped by test\ndisabled_{key}: ", 1
        )
        with open(original, "w", encoding="utf-8") as fh:
            fh.write(text)
        registry.reset_registry()

    yield _do

    with open(original, "w", encoding="utf-8") as fh:
        fh.write(saved)
    registry.reset_registry()


def test_default_mapping_loads():
    mapping = registry.load_adapters_yaml()
    assert mapping["segmentation"] == "adapters.mock.segmentation.MockSegmentation"
    assert registry.get_adapter("segmentation").__class__.__name__ == "MockSegmentation"


def test_swap_one_path_changes_the_result_and_nothing_else(swap, client, session_id, front_png_b64):
    before = run(client, session_id, 3, image=front_png_b64)
    assert before["model_name"] == "mock-silhouette-v1"

    swap("segmentation", "adapters.real.dummy_seg.DummySwapSegmenter")

    after = run(client, session_id, 3, image=front_png_b64)
    assert after["model_name"] == "dummy-swap-check"
    # the contract is unchanged, only the implementation moved
    assert set(after) == set(before)
    assert isinstance(after["boundary_quality"], float)


def test_swapped_adapter_is_the_real_class(swap, client, session_id, front_png_b64):
    swap("segmentation", "adapters.real.dummy_seg.DummySwapSegmenter")
    assert registry.get_adapter("segmentation").__class__.__name__ == "DummySwapSegmenter"
    run(client, session_id, 3, image=front_png_b64)


def test_unknown_key_raises(swap):
    swap("segmentation", "adapters.real.dummy_seg.DummySwapSegmenter")
    with pytest.raises(KeyError):
        registry.get_adapter("no_such_step")


def test_every_registered_key_instantiates():
    for key in registry.load_adapters_yaml():
        assert registry.get_adapter(key) is not None
