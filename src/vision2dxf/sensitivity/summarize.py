from __future__ import annotations
import pandas as pd


def summarize_robustness(scenario_df: pd.DataFrame, design_ids: list[str]) -> pd.DataFrame:
    summaries = []
    for did in design_ids:
        score_cols = [c for c in scenario_df.columns if c.startswith("score_") and did in c]
        rank_cols = [c for c in scenario_df.columns if c.startswith("rank_") and did in c]
        if not score_cols:
            continue
        scores = scenario_df[score_cols[0]]
        ranks = scenario_df[rank_cols[0]]

        wins = (scenario_df["winner"] == did).sum()
        tied_wins = scenario_df["winner"].str.contains(did, na=False).sum() - wins
        total = len(scenario_df)

        summaries.append({
            "design_id": did,
            "outright_wins": int(wins),
            "tied_wins": int(tied_wins),
            "win_pct": round(wins / total * 100, 2) if total > 0 else 0,
            "mean_score": round(scores.mean(), 4),
            "score_std": round(scores.std(), 4),
            "mean_rank": round(ranks.mean(), 4),
            "worst_rank": int(ranks.max()),
            "mean_margin": round(scenario_df["winning_margin"].mean(), 4),
        })

    return pd.DataFrame(summaries)


def generate_recommendation(
    robustness_df: pd.DataFrame,
    raw_values: dict,
    required_criteria: list[str],
) -> dict | None:
    for did, vals in raw_values.items():
        for crit in required_criteria:
            if vals.get(crit) is None:
                return None

    best = robustness_df.sort_values("outright_wins", ascending=False).iloc[0]
    return {
        "recommended_design": best["design_id"],
        "outright_wins": int(best["outright_wins"]),
        "win_pct": best["win_pct"],
        "mean_score": best["mean_score"],
        "mean_rank": best["mean_rank"],
        "note": "Results depend on criterion priorities if margins are close.",
    }
