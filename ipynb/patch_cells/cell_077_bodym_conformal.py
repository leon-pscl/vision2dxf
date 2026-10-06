"""Split conformal prediction for per-measurement prediction intervals.

References
----------
Angelopoulos, A. N., & Bates, S. (2021). A gentle introduction to conformal prediction and
distribution-free uncertainty quantification (arXiv:2107.07511).
https://arxiv.org/abs/2107.07511
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


def conformal_quantile(abs_residuals: np.ndarray, coverage: float) -> float:
    """Finite-sample-corrected conformal quantile of absolute residuals.

    Parameters
    ----------
    abs_residuals : ndarray, shape (n,)
        Absolute residuals (cm) on the calibration split.
    coverage : float
        Target coverage, e.g. 0.80.

    Returns
    -------
    float
        The conformal quantile in cm: the ``ceil((n + 1) * coverage) / n`` empirical quantile,
        computed with ``method='higher'`` so it is a genuine order statistic.

    Notes
    -----
    The returned value is a **half-width** in cm: intervals are ``[pred - q, pred + q]``.
    Converting to millimetres is therefore ``q * 10``, not ``q * 20`` (fix B12).

    Raises
    ------
    ValueError
        If ``coverage`` is not in (0, 1) or ``abs_residuals`` is empty or non-finite.
    """
    r = np.asarray(abs_residuals, dtype=float).ravel()
    r = r[np.isfinite(r)]
    if r.size == 0:
        raise ValueError("no finite residuals supplied")
    if not 0.0 < coverage < 1.0:
        raise ValueError(f"coverage must be in (0, 1), got {coverage}")
    n = r.size
    level = min(1.0, np.ceil((n + 1) * coverage) / n)
    return float(np.quantile(r, level, method="higher"))


def fit_conformal(y_true_cm: np.ndarray, y_pred_cm: np.ndarray, columns: Sequence[str],
                  coverages: Sequence[float] = (0.80, 0.90)) -> Dict[str, Dict[str, float]]:
    """Compute per-measurement conformal quantiles on the calibration split.

    Parameters
    ----------
    y_true_cm, y_pred_cm : ndarray, shape (n_calib, m)
        Calibration ground truth and predictions, cm. **One row per calibration unit** - see
        the ``unit`` field written into ``conformal.json``.
    columns : sequence of str
        Measurement names.
    coverages : sequence of float, default (0.80, 0.90)
        Target coverages.

    Returns
    -------
    dict
        ``{column: {coverage_str: halfwidth_cm}}`` where ``coverage_str`` is e.g. ``'80'``.
    """
    resid = np.abs(np.asarray(y_pred_cm, float) - np.asarray(y_true_cm, float))
    out: Dict[str, Dict[str, float]] = {}
    for j, col in enumerate(columns):
        out[col] = {}
        for cov in coverages:
            out[col][str(int(round(cov * 100)))] = conformal_quantile(resid[:, j], cov)
    return out


def load_conformal(path: Path | str) -> Dict[str, Dict[str, float]]:
    """Load a ``conformal.json`` file.

    Parameters
    ----------
    path : path-like
        Path written by :func:`save_conformal`.

    Returns
    -------
    dict
        ``{variant: {column: {coverage_str: halfwidth_cm}}}``.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_conformal(payload: Dict[str, Dict[str, Dict[str, float]]], path: Path | str,
                   extra: Optional[Dict] = None) -> Path:
    """Write ``conformal.json`` with provenance metadata.

    Parameters
    ----------
    payload : dict
        ``{variant: {column: {coverage_str: halfwidth_cm}}}``.
    path : path-like
        Destination.
    extra : dict, optional
        Additional top-level keys (seeds, calibration unit, assumptions).

    Returns
    -------
    Path
        The written path.
    """
    body = {"conformal_quantiles_cm": payload}
    if extra:
        body.update(extra)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return p


def apply_intervals(y_pred_cm: np.ndarray, quantiles: Dict[str, Dict[str, float]],
                    columns: Sequence[str]) -> Dict[float, Dict[str, tuple]]:
    """Build symmetric prediction intervals around predictions.

    Parameters
    ----------
    y_pred_cm : ndarray, shape (n, m)
        Predictions, cm.
    quantiles : dict
        ``{column: {coverage_str: halfwidth_cm}}`` from :func:`fit_conformal`.
    columns : sequence of str
        Measurement names, matching the columns of ``y_pred_cm``.

    Returns
    -------
    dict
        ``{coverage: {column: (lo, hi)}}`` in cm.
    """
    y_pred_cm = np.asarray(y_pred_cm, dtype=float)
    out: Dict[float, Dict[str, tuple]] = {}
    for cov_str in sorted({c for v in quantiles.values() for c in v}, reverse=True):
        cov = int(cov_str) / 100.0
        per_col = {}
        for j, col in enumerate(columns):
            q = float(quantiles[col][cov_str])
            per_col[col] = (y_pred_cm[:, j] - q, y_pred_cm[:, j] + q)
        out[cov] = per_col
    return out


def clamp_intervals(intervals: Dict[float, Dict[str, tuple]], bounds: Dict[str, float]
                    ) -> Dict[float, Dict[str, tuple]]:
    """Clip interval endpoints to a plausibility range.

    Step 7 of the pipeline uses these intervals for plausibility gating, so an interval that
    extends to a physically impossible value (negative chest, 400 cm waist) is noise, not
    caution.

    Parameters
    ----------
    intervals : dict
        Output of :func:`apply_intervals`.
    bounds : dict
        ``{column: (lo, hi)}`` plausible range in cm. Columns absent from ``bounds`` are left
        unclamped.

    Returns
    -------
    dict
        The clamped intervals. The **centre is preserved**: a clamp that would push ``lo``
        above the interval's own centre is resolved by widening the bound rather than
        producing an interval that excludes the point estimate (fix B12). Callers that need
        to know a clamp occurred should compare against the unclamped intervals.
    """
    out: Dict[float, Dict[str, tuple]] = {}
    for cov, per_col in intervals.items():
        out[cov] = {}
        for col, (lo, hi) in per_col.items():
            if col in bounds:
                blo, bhi = float(bounds[col][0]), float(bounds[col][1])
                # lo/hi may be arrays (batched) or scalars
                centre_lo, centre_hi = 0.5 * (np.asarray(lo) + np.asarray(hi))
                new_lo = np.maximum(lo, blo)
                new_hi = np.minimum(hi, bhi)
                # keep the centre inside the clamped interval
                new_lo = np.minimum(new_lo, centre_lo)
                new_hi = np.maximum(new_hi, centre_hi)
                out[cov][col] = (new_lo, new_hi)
            else:
                out[cov][col] = (lo, hi)
    return out


def clamp_scalar_intervals(pred_cm: float, halfwidths_cm: Dict[str, float],
                           bounds: Optional[Sequence[float]]) -> Dict[str, object]:
    """Clamp one prediction's symmetric intervals, preserving the point estimate (fix B12).

    Parameters
    ----------
    pred_cm : float
        Point estimate, cm.
    halfwidths_cm : dict
        ``{'80': q80_cm, '90': q90_cm}``.
    bounds : sequence of float, optional
        ``(lo, hi)`` plausibility range in cm, or None for no clamping.

    Returns
    -------
    dict
        ``{'lo80', 'hi80', 'lo90', 'hi90', 'clamped'}``, all cm. ``clamped`` is True when any
        endpoint was moved by the envelope.
    """
    lo80, hi80 = pred_cm - halfwidths_cm["80"], pred_cm + halfwidths_cm["80"]
    lo90, hi90 = pred_cm - halfwidths_cm["90"], pred_cm + halfwidths_cm["90"]
    clamped = False
    if bounds is not None:
        blo, bhi = float(bounds[0]), float(bounds[1])
        lo80, hi80 = max(lo80, blo), min(hi80, bhi)
        lo90, hi90 = max(lo90, blo), min(hi90, bhi)
        # Never let the envelope exclude the point estimate: an interval that does not
        # contain its own centre is not an interval, it is a contradiction.
        lo80 = min(lo80, pred_cm)
        hi80 = max(hi80, pred_cm)
        lo90 = min(lo90, pred_cm)
        hi90 = max(hi90, pred_cm)
        clamped = not (abs(lo80 - (pred_cm - halfwidths_cm["80"])) < 1e-12
                       and abs(hi80 - (pred_cm + halfwidths_cm["80"])) < 1e-12)
    return {"lo80": float(lo80), "hi80": float(hi80), "lo90": float(lo90),
            "hi90": float(hi90), "clamped": bool(clamped)}