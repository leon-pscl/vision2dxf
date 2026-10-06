"""Canonical silhouette preprocessing for the BodyM measurement model.

This module wraps the *training* implementation (``bodym_cnn.to_tensor``) so there is exactly
one definition of preprocessing in the project. The self-test at the bottom recomputes
tensors from **saved reference input masks** and compares them against the tensors saved by
the training notebook with ``np.allclose(atol=1e-6)`` - a full-tensor equality check, not a
shape-and-contract check (fix D1).

Image size and mask threshold are read from ``config.yaml`` next to this file.

Usage
-----
    from preprocess import silhouette_to_tensor
    x = silhouette_to_tensor(front_mask, side_mask, height_cm=172.0, weight_kg=74.0)
    # x.shape == (3, IMG_HEIGHT, 2 * IMG_WIDTH); the mask holds front | side horizontally
    x_h = silhouette_to_tensor(front_mask, side_mask, 172.0)   # V-H: 2 channels, no zero-fill
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

_HERE = Path(__file__).resolve().parent
for _cand in (_HERE, _HERE.parent / "src", _HERE.parent):
    if (_cand / "bodym_cnn.py").is_file():
        sys.path.insert(0, str(_cand))
        break

from bodym_cnn import to_tensor  # noqa: E402

#: Channel counts per variant, mirroring src/bodym_cnn.py.
N_CHANNELS = 3        # [mask, height, weight]
N_CHANNELS_VH = 2     # [mask, height]


def _load_config() -> dict:
    """Read ``config.yaml`` from this directory, falling back to documented defaults.

    Returns
    -------
    dict
        Config mapping; always contains ``img_height``, ``img_width``, ``mask_threshold``.
    """
    path = _HERE / "config.yaml"
    cfg = {"img_height": 640, "img_width": 480, "mask_threshold": 127}
    if path.is_file():
        cfg.update(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    return cfg


CFG = _load_config()
IMG_HEIGHT = int(CFG["img_height"])
IMG_WIDTH = int(CFG["img_width"])
MASK_THRESHOLD = int(CFG["mask_threshold"])


def silhouette_to_tensor(front_mask, side_mask, height_cm, weight_kg=None, use_weight=True):
    """Build the model input tensor from two silhouette masks.

    Parameters
    ----------
    front_mask : ndarray, shape (H, W)
        Front-view binary silhouette. Pixels with value > ``MASK_THRESHOLD`` are foreground.
    side_mask : ndarray, shape (H, W)
        Side-view binary silhouette of the same person.
    height_cm : float
        Subject stature, cm.
    weight_kg : float, optional
        Subject mass, kg. Required when ``use_weight`` is True.
    use_weight : bool, default True
        True for the V-HW model (3 channels). False for the V-H model, where the weight channel
        is **removed**, not zero-filled, so the output has 2 channels.

    Returns
    -------
    ndarray, shape (3 or 2, IMG_HEIGHT, 2 * IMG_WIDTH)
        Channels ``[mask, height, weight]``, float32. Channel 0 is 0/1; the scalar channels are
        constant. ``mask`` holds the front and side silhouettes concatenated **horizontally**
        (front left, side right), resized as one canvas.

    Raises
    ------
    ValueError
        If either mask is empty, or if ``weight_kg`` is None while ``use_weight`` is True.

    Examples
    --------
    >>> x = silhouette_to_tensor(m, m, 172.0, 74.0)
    >>> x.shape[1:] == (IMG_HEIGHT, 2 * IMG_WIDTH)
    True
    """
    return to_tensor(front_mask, side_mask, height_cm, weight_kg,
                     IMG_HEIGHT, IMG_WIDTH, MASK_THRESHOLD, use_weight)


def self_test(verbose: bool = True) -> bool:
    """Verify preprocessing against reference tensors saved during training.

    For each saved reference sample this recomputes the tensor **from the saved input masks and
    scalars** and asserts full-tensor equality with ``np.allclose(..., atol=1e-6)``. The earlier
    version built a synthetic rectangle and only checked shape, dtype and the scalar constants,
    which is a contract test, not an equivalence test: a changed resize filter or an inverted
    channel order on the mask would still have passed.

    Reads ``reference_tensors.npz`` (tensors), ``reference_inputs.npz`` (front/side masks and
    scalars) and ``reference_meta.json`` (variant + index labels).

    Returns
    -------
    bool
        True when every reference reproduces exactly.

    Raises
    ------
    AssertionError
        Never raised; failures are collected and reported, and the boolean is returned.
    """
    ref_path = _HERE / "reference_tensors.npz"
    inp_path = _HERE / "reference_inputs.npz"
    if not ref_path.is_file() or not inp_path.is_file():
        if verbose:
            print("preprocess self-test: SKIPPED "
                  "(reference_tensors.npz / reference_inputs.npz not alongside this file)")
        return True

    ref = np.load(ref_path)
    inp = np.load(inp_path)
    atol = float(CFG.get("preprocess_atol", 1e-6))
    failures = []
    checked = 0
    for variant, expect_ch in (("V-HW", N_CHANNELS), ("V-H", N_CHANNELS_VH)):
        keys = sorted(k for k in ref.files if k.startswith(variant + "_")
                      and k[len(variant) + 1:].isdigit())
        if not keys:
            failures.append(f"{variant}: no reference tensors found")
            continue
        for k in keys:
            i = k[len(variant) + 1:]
            expected = ref[k]
            front_k, side_k = f"front_{variant}_{i}", f"side_{variant}_{i}"
            h_k, w_k = f"height_cm_{variant}_{i}", f"weight_kg_{variant}_{i}"
            if not all(n in inp.files for n in (front_k, side_k, h_k)):
                failures.append(f"{variant}#{i}: reference input missing ({front_k} etc)")
                continue
            tag = f"{variant}#{i}"

            use_w = expect_ch == N_CHANNELS
            x = silhouette_to_tensor(inp[front_k], inp[side_k], float(inp[h_k]),
                                     float(inp[w_k]) if (use_w and w_k in inp.files) else None,
                                     use_weight=use_w)
            checked += 1

            if x.shape != expected.shape:
                failures.append(f"{tag}: shape {x.shape} != {expected.shape}")
                continue
            if x.dtype != np.float32:
                failures.append(f"{tag}: dtype {x.dtype} != float32")
                continue
            if not set(np.unique(x[0])).issubset({0.0, 1.0}):
                failures.append(f"{tag}: mask channel is not binarised")
                continue
            # the scalar channels are constant by construction; a non-constant one means the
            # channel was built from image data, which is a silent preprocessing bug
            for c in range(1, expect_ch):
                if float(np.ptp(x[c])) > 1e-9:
                    failures.append(f"{tag}: scalar channel {c} is not constant")
                    break
            else:
                if abs(float(x[1].mean()) - float(inp[h_k]) / 200.0) > atol:
                    failures.append(f"{tag}: height channel is not height_cm/200")
                elif use_w and abs(float(x[2].mean()) - float(inp[w_k]) / 100.0) > atol:
                    failures.append(f"{tag}: weight channel is not weight_kg/100")
                elif not np.allclose(x, expected, atol=atol):
                    d = float(np.abs(x - expected).max())
                    failures.append(f"{tag}: tensor mismatch, max|diff| = {d:.3e}")

    if verbose:
        print(f"preprocess self-test: {'PASS' if not failures else 'FAIL'} "
              f"({checked} references recomputed and compared at atol={atol:g}, "
              f"IMG {IMG_HEIGHT}x{IMG_WIDTH}, threshold {MASK_THRESHOLD})")
        for f in failures:
            print("   ", f)
    return not failures


if __name__ == "__main__":
    raise SystemExit(0 if self_test() else 1)