from __future__ import annotations
import pandas as pd
from .normalize import normalize_matrix
from .weights import generate_loi_permutations, loi_to_weights, CRITERIA_ORDER


def compute_scores(
    raw_df: pd.DataFrame,
    directions: dict[str, str],
    tie_tolerance: float = 1e-9,
) -> tuple[pd.DataFrame, list[str]]:
    norm_df, available = normalize_matrix(raw_df, directions)
    missing = [c for c in CRITERIA_ORDER if c not in available]
    permutations = generate_loi_permutations()

    results = []
    for combo in permutations:
        full_weights = loi_to_weights(combo)

        # Redistribute weight of missing criteria across available ones
        if missing:
            missing_weight = sum(full_weights[m] for m in missing)
            available_weights = {k: v for k, v in full_weights.items() if k in available}
            total_avail = sum(available_weights.values())
            if total_avail > 0:
                adjusted = {k: v + missing_weight * (v / total_avail) for k, v in available_weights.items()}
            else:
                adjusted = {k: 1.0 / len(available) for k in available}
        else:
            adjusted = full_weights

        weight_series = pd.Series(adjusted)
        weighted = norm_df.mul(weight_series, axis=1)
        scores = weighted.sum(axis=1)

        sorted_scores = scores.sort_values(ascending=False)
        winner = sorted_scores.index[0]
        winner_score = sorted_scores.iloc[0]

        tied = False
        for other in sorted_scores.index[1:]:
            if abs(sorted_scores[other] - winner_score) < tie_tolerance:
                tied = True
                winner = f"tied: {winner} & {other}"
                break

        margin = sorted_scores.iloc[0] - sorted_scores.iloc[-1] if len(sorted_scores) > 1 else 0
        ranks = scores.rank(ascending=False, method="min")

        row = {"loi_combo": str(combo)}
        for crit in CRITERIA_ORDER:
            row[f"weight_{crit}"] = full_weights[crit]
        for did in scores.index:
            row[f"score_{did}"] = round(scores[did], 6)
            row[f"rank_{did}"] = int(ranks[did])
        row["winner"] = winner
        row["winning_margin"] = round(margin, 6)
        row["tied"] = tied
        results.append(row)

    return pd.DataFrame(results), missing
