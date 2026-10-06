"""Metric suite: accuracy, agreement, tolerance pass rates, conformal coverage, strata.

Units
-----
Targets and predictions are centimetres (cm). Every error column below is millimetres (mm).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

TOLERANCES_MM: Sequence[float] = (9, 15, 25)
COVERAGES: Sequence[float] = (0.80, 0.90)


def per_measurement_metrics(
    y_true_cm: np.ndarray,
    y_pred_cm: np.ndarray,
    columns: Sequence[str],
    intervals: Optional[Dict[float, Dict[str, tuple]]] = None,
    tolerances_mm: Sequence[float] = TOLERANCES_MM,
) -> pd.DataFrame:
    """Compute the full per-measurement metric table.

    Parameters
    ----------
    y_true_cm : ndarray, shape (n, m)
        Ground-truth measurements, cm.
    y_pred_cm : ndarray, shape (n, m)
        Predictions, cm, same column order as ``columns``.
    columns : sequence of str
        Measurement names, length ``m``.
    intervals : dict, optional
        ``{coverage: {column: (lo_cm, hi_cm)}}`` as produced by
        :func:`src.bodym_conformal.apply_intervals`. When provided, ``coverage_80``,
        ``coverage_90``, ``width_80_mm`` and ``width_90_mm`` are filled.
    tolerances_mm : sequence of float, default (9, 15, 25)
        Tolerance thresholds for the pass-rate columns.

    Returns
    -------
    pandas.DataFrame
        One row per measurement.

    Raises
    ------
    ValueError
        If the two matrices have different shapes or the number of columns disagrees with
        ``columns``.
    """
    y_true_cm = np.asarray(y_true_cm, dtype=float)
    y_pred_cm = np.asarray(y_pred_cm, dtype=float)
    if y_true_cm.shape != y_pred_cm.shape:
        raise ValueError(f"shape mismatch: {y_true_cm.shape} vs {y_pred_cm.shape}")
    if y_true_cm.shape[1] != len(columns):
        raise ValueError(f"{y_true_cm.shape[1]} columns but {len(columns)} names given")

    err_mm = (y_pred_cm - y_true_cm) * 10.0
    abs_err_mm = np.abs(err_mm)
    n = y_true_cm.shape[0]

    rows: List[dict] = []
    for j, col in enumerate(columns):
        e, ae = err_mm[:, j], abs_err_mm[:, j]
        sd = float(np.std(e, ddof=1)) if n > 1 else float("nan")
        mean_diff = float(np.mean(e))
        row = {
            "measurement": col,
            "n": n,
            "mae_mm": float(np.mean(ae)),
            "bias_mm": mean_diff,
            "tp50_mm": float(np.percentile(ae, 50)),
            "tp75_mm": float(np.percentile(ae, 75)),
            "tp90_mm": float(np.percentile(ae, 90)),
            "ba_mean_diff_mm": mean_diff,
            "ba_loa_lo_mm": mean_diff - 1.96 * sd,
            "ba_loa_hi_mm": mean_diff + 1.96 * sd,
        }
        for tol in tolerances_mm:
            row[f"pct_within_{int(tol)}mm"] = float(100.0 * np.mean(ae <= tol))
        if intervals:
            for cov in COVERAGES:
                pair = intervals.get(cov, {}).get(col)
                if pair is not None:
                    lo, hi = np.asarray(pair[0]), np.asarray(pair[1])
                    inside = (y_true_cm[:, j] >= lo) & (y_true_cm[:, j] <= hi)
                    row[f"coverage_{int(cov * 100)}"] = float(100.0 * np.mean(inside))
                    row[f"width_{int(cov * 100)}_mm"] = float(np.mean((hi - lo) * 10.0))
        rows.append(row)
    return pd.DataFrame(rows)


def stratum_sizes(strata_values: Sequence[str]) -> Dict[str, int]:
    """Number of samples per stratum label.

    A standalone helper because ``df.groupby("stratum").n()`` raises
    ``TypeError: 'SeriesGroupBy' object is not callable`` on every pandas that has an
    ``n`` column, and because callers need the per-stratum counts in several places.

    Parameters
    ----------
    strata_values : sequence of str
        Stratum label per sample.

    Returns
    -------
    dict
        ``{label: n}``.
    """
    s = pd.Series(list(strata_values), dtype="object")
    return {str(k): int(v) for k, v in s.value_counts().items()}


def stratified_metrics(
    y_true_cm: np.ndarray,
    y_pred_cm: np.ndarray,
    columns: Sequence[str],
    strata_values: Sequence[str],
    min_stratum_n: int = 30,
    **kwargs,
) -> pd.DataFrame:
    """Per-measurement metrics repeated within each stratum, with ``n`` and a reliability flag.

    Parameters
    ----------
    y_true_cm, y_pred_cm : ndarray, shape (n, m)
        Ground truth and predictions, cm.
    columns : sequence of str
        Measurement names.
    strata_values : sequence of str
        Stratum label per sample, length ``n``.
    min_stratum_n : int, default 30
        Strata with fewer samples are flagged ``reliable=False``.
    **kwargs
        Forwarded to :func:`per_measurement_metrics`.

    Returns
    -------
    pandas.DataFrame
        Per-stratum metric tables concatenated, with a leading ``stratum`` column.
    """
    y_true_cm = np.asarray(y_true_cm, dtype=float)
    y_pred_cm = np.asarray(y_pred_cm, dtype=float)
    labels = np.asarray(list(strata_values), dtype=object)
    if labels.shape[0] != y_true_cm.shape[0]:
        raise ValueError(f"{labels.shape[0]} stratum labels for {y_true_cm.shape[0]} rows")

    out = []
    # enumerate over the unique labels rather than pd.Series.groupby(...).groups, whose
    # index semantics changed in pandas 2.2
    for value in pd.unique(labels):
        idx = np.flatnonzero(labels == value)
        tbl = per_measurement_metrics(y_true_cm[idx], y_pred_cm[idx], columns, **kwargs)
        tbl.insert(0, "stratum", value)
        tbl["reliable"] = tbl["n"] >= min_stratum_n
        out.append(tbl)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def format_metric_table(tbl: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    """Render a metric table with stable numeric formatting.

    Parameters
    ----------
    tbl : DataFrame
        Metric table.
    floatfmt : str, default ``"{:.2f}"``
        Format applied to every float column.

    Returns
    -------
    str
        Table as a string, ready to print.
    """
    view = tbl.copy()
    for c in view.columns:
        if pd.api.types.is_float_dtype(view[c]):
            view[c] = view[c].map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
    return view.to_string(index=False)
