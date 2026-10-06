"""Inference entry point for the BodyM measurement model.

Returns 14 body measurements in cm, each with 80% and 90% split-conformal prediction intervals.

Usage
-----
    from infer import predict_measurements
    result = predict_measurements(front_mask, side_mask, height_cm=172.0, weight_kg=74.0)
    print(result["chest"])   # {'value_cm':..., 'lo80':..., 'hi80':..., 'lo90':..., 'hi90':...,
                             #  'clamped': False}

    # height only -> the V-H variant is selected automatically
    result = predict_measurements(front_mask, side_mask, height_cm=172.0)

Model files, ``conformal.json``, ``config.yaml`` and ``bodym_cnn.py`` must sit alongside this file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
for _cand in (_HERE, _HERE.parent / "src", _HERE.parent):
    if (_cand / "bodym_cnn.py").is_file():
        sys.path.insert(0, str(_cand))
        break

from bodym_cnn import MeasurementNet  # noqa: E402
from preprocess import silhouette_to_tensor  # noqa: E402

#: Variant metadata. ``requires_weight`` mirrors the training-time channel contract.
VARIANTS: Dict[str, Dict] = {
    "V-HW": {"pt": "model_vhw.pt", "requires_weight": True},
    "V-H": {"pt": "model_vh.pt", "requires_weight": False},
}

_CACHE: Dict[str, Dict] = {}


def _conformal() -> Dict:
    """Read the conformal quantiles and plausibility bounds from ``conformal.json``.

    Returns
    -------
    dict
        The parsed file.
    """
    return json.loads((_HERE / "conformal.json").read_text(encoding="utf-8"))


def load_variant(name: str, device: Optional[str] = None) -> Dict:
    """Load a variant's weights and conformal quantiles, cached per process.

    Parameters
    ----------
    name : {'V-HW', 'V-H'}
        Variant to load.
    device : str, optional
        Torch device string; defaults to CUDA when available.

    Returns
    -------
    dict
        ``{'model', 'quantiles', 'targets', 'use_weight', 'device'}``.

    Raises
    ------
    ValueError
        If ``name`` is not a known variant.
    KeyError
        If the checkpoint does not record its backbone, so the architecture cannot be rebuilt.
    FileNotFoundError
        If the weights file is missing from the bundle.
    """
    if name not in VARIANTS:
        raise ValueError(f"unknown variant {name!r}; expected one of {list(VARIANTS)}")
    if name in _CACHE:
        return _CACHE[name]

    ckpt_path = _HERE / VARIANTS[name]["pt"]
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"missing weights file: {ckpt_path}")
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)

    # the backbone name is written by section 9.1; rebuilding the wrong trunk would fail
    # loudly on load_state_dict, but failing here names the actual cause
    backbone = ckpt.get("backbone")
    if not backbone:
        raise KeyError(f"{ckpt_path.name} does not record 'backbone'; re-run section 9.1")
    model = MeasurementNet(backbone=backbone, pretrained=False,
                           n_outputs=len(ckpt["targets"]), hidden=ckpt.get("hidden", 512),
                           dropout=ckpt.get("dropout", 0.2),
                           use_weight=ckpt.get("use_weight", True)).to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()

    payload = _conformal()
    _CACHE[name] = {"model": model,
                    "quantiles": payload["conformal_quantiles_cm"][name],
                    "targets": ckpt["targets"],
                    "use_weight": ckpt.get("use_weight", True),
                    "device": dev}
    return _CACHE[name]


def select_variant(weight_kg: Optional[float]) -> str:
    """Pick the variant from whether a weight was supplied.

    Parameters
    ----------
    weight_kg : float or None
        Subject mass, kg.

    Returns
    -------
    str
        ``'V-HW'`` when ``weight_kg`` is not None, else ``'V-H'``.
    """
    return "V-HW" if weight_kg is not None else "V-H"


def predict_measurements(front_mask, side_mask, height_cm: float,
                         weight_kg: Optional[float] = None,
                         variant: Optional[str] = None,
                         clamp: bool = True,
                         device: Optional[str] = None) -> Dict[str, Dict]:
    """Estimate all body measurements with prediction intervals.

    Parameters
    ----------
    front_mask, side_mask : ndarray, shape (H, W)
        Binary silhouette masks of the same person, front and side views. Values above the
        configured threshold are foreground.
    height_cm : float
        Subject stature, cm. Required.
    weight_kg : float, optional
        Subject mass, kg. When supplied, the V-HW variant is used.
    variant : str, optional
        Force a variant instead of auto-selecting from ``weight_kg``.
    clamp : bool, default True
        Clip interval endpoints to the plausibility envelope recorded in ``conformal.json``.
        A clamp never removes the point estimate from its own interval; when a clamp would
        have done so, the bound is relaxed and ``'clamped': True`` is returned for that
        measurement so the caller can tell the interval was truncated.
    device : str, optional
        Torch device string.

    Returns
    -------
    dict
        ``{measurement: {'value_cm', 'lo80', 'hi80', 'lo90', 'hi90', 'clamped'}}``, values in
        centimetres.

    Raises
    ------
    ValueError
        If ``variant`` is forced but inconsistent with whether ``weight_kg`` was supplied.

    Examples
    --------
    >>> r = predict_measurements(front, side, 172.0, 74.0)
    >>> sorted(r)[:3]
    ['ankle', 'arm-length', 'bicep']
    """
    name = variant or select_variant(weight_kg)
    if variant == "V-HW" and weight_kg is None:
        raise ValueError("V-HW was forced but weight_kg is None")
    if variant == "V-H" and weight_kg is not None:
        raise ValueError("V-H was forced but weight_kg was supplied")

    v = load_variant(name, device)
    x = silhouette_to_tensor(front_mask, side_mask, height_cm, weight_kg,
                             use_weight=v["use_weight"])
    with torch.no_grad():
        p = v["model"](torch.from_numpy(x).unsqueeze(0).to(v["device"])).float().cpu().numpy()[0]

    payload = _conformal()
    bounds = payload.get("plausibility_bounds_cm", {}) if clamp else {}

    out: Dict[str, Dict] = {}
    for j, meas in enumerate(v["targets"]):
        q = v["quantiles"][meas]
        lo80, hi80 = float(p[j]) - q["80"], float(p[j]) + q["80"]
        lo90, hi90 = float(p[j]) - q["90"], float(p[j]) + q["90"]
        was_clamped = False
        b = bounds.get(meas)
        if b:
            blo, bhi = float(b[0]), float(b[1])
            n_lo80, n_hi80 = max(lo80, blo), min(hi80, bhi)
            n_lo90, n_hi90 = max(lo90, blo), min(hi90, bhi)
            # keep the point estimate inside its own interval
            n_lo80, n_hi80 = min(n_lo80, float(p[j])), max(n_hi80, float(p[j]))
            n_lo90, n_hi90 = min(n_lo90, float(p[j])), max(n_hi90, float(p[j]))
            was_clamped = (abs(n_lo80 - lo80) > 1e-12 or abs(n_hi80 - hi80) > 1e-12
                           or abs(n_lo90 - lo90) > 1e-12 or abs(n_hi90 - hi90) > 1e-12)
            lo80, hi80, lo90, hi90 = n_lo80, n_hi80, n_lo90, n_hi90
        out[meas] = {"value_cm": float(p[j]),
                     "lo80": float(lo80), "hi80": float(hi80),
                     "lo90": float(lo90), "hi90": float(hi90),
                     "clamped": bool(was_clamped)}
    return out


if __name__ == "__main__":
    import argparse

    from PIL import Image

    ap = argparse.ArgumentParser(description="Estimate body measurements from two silhouette masks.")
    ap.add_argument("front", help="front silhouette PNG")
    ap.add_argument("side", help="side silhouette PNG")
    ap.add_argument("--height-cm", type=float, required=True)
    ap.add_argument("--weight-kg", type=float, default=None)
    a = ap.parse_args()
    out = predict_measurements(np.array(Image.open(a.front)), np.array(Image.open(a.side)),
                               a.height_cm, a.weight_kg)
    print(json.dumps(out, indent=2))