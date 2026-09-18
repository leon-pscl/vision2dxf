from __future__ import annotations
import pandas as pd


def normalize_minimize(series: pd.Series) -> pd.Series:
    mx = series.max()
    mn = series.min()
    if mx == mn:
        return pd.Series(10.0, index=series.index)
    return 1 + 9 * (mx - series) / (mx - mn)


def normalize_maximize(series: pd.Series) -> pd.Series:
    mx = series.max()
    mn = series.min()
    if mx == mn:
        return pd.Series(10.0, index=series.index)
    return 1 + 9 * (series - mn) / (mx - mn)


def normalize_matrix(
    raw_df: pd.DataFrame,
    directions: dict[str, str],
) -> tuple[pd.DataFrame, list[str]]:
    available = [c for c in raw_df.columns if raw_df[c].notna().any()]
    norm_df = pd.DataFrame(index=raw_df.index)
    for col in available:
        if directions.get(col) == "minimize":
            norm_df[col] = normalize_minimize(raw_df[col])
        else:
            norm_df[col] = normalize_maximize(raw_df[col])
    return norm_df, available
