from __future__ import annotations
from vision2dxf.sensitivity.normalize import normalize_minimize, normalize_maximize, normalize_matrix
from vision2dxf.sensitivity.weights import generate_loi_permutations, loi_to_weights, CRITERIA_ORDER
from vision2dxf.sensitivity.score import compute_scores
from vision2dxf.sensitivity.summarize import summarize_robustness, generate_recommendation
import pandas as pd
import pytest


FIXTURE_RAW = pd.DataFrame(
    {
        "economic": [100.0, 150.0, 120.0],
        "usability": [70.0, 65.0, 75.0],
        "performance": [5.0, 3.0, 4.0],
        "reliability": [2.0, 1.0, 3.0],
        "maintainability": [60.0, 70.0, 65.0],
    },
    index=["design_a", "design_b", "design_c"],
)

DIRECTIONS = {
    "economic": "minimize",
    "usability": "maximize",
    "performance": "minimize",
    "reliability": "minimize",
    "maintainability": "maximize",
}


class TestNormalization:
    def test_minimize_direction(self):
        s = pd.Series([10.0, 20.0, 30.0])
        norm = normalize_minimize(s)
        assert norm.iloc[0] == 10.0
        assert norm.iloc[2] == 1.0

    def test_maximize_direction(self):
        s = pd.Series([10.0, 20.0, 30.0])
        norm = normalize_maximize(s)
        assert norm.iloc[0] == 1.0
        assert norm.iloc[2] == 10.0

    def test_equal_values_no_division_by_zero(self):
        s = pd.Series([5.0, 5.0, 5.0])
        norm = normalize_minimize(s)
        assert all(v == 10.0 for v in norm)

    def test_matrix_normalization(self):
        norm, available = normalize_matrix(FIXTURE_RAW, DIRECTIONS)
        for col in norm.columns:
            assert norm[col].between(1, 10).all()


class TestWeights:
    def test_120_permutations(self):
        perms = generate_loi_permutations()
        assert len(perms) == 120

    def test_unique_permutations(self):
        perms = generate_loi_permutations()
        assert len(set(perms)) == 120

    def test_weights_sum_to_one(self):
        for combo in generate_loi_permutations()[:5]:
            w = loi_to_weights(combo)
            assert abs(sum(w.values()) - 1.0) < 1e-9


class TestScores:
    def test_scores_in_range(self):
        df, missing = compute_scores(FIXTURE_RAW, DIRECTIONS)
        for col in df.columns:
            if col.startswith("score_"):
                assert df[col].between(1, 10).all()

    def test_fixed_order_criteria(self):
        df, missing = compute_scores(FIXTURE_RAW, DIRECTIONS)
        assert len(df) == 120

    def test_deterministic(self):
        df1, _ = compute_scores(FIXTURE_RAW, DIRECTIONS)
        df2, _ = compute_scores(FIXTURE_RAW, DIRECTIONS)
        pd.testing.assert_frame_equal(df1, df2)

    def test_missing_criteria_flagged(self):
        raw = FIXTURE_RAW.copy()
        raw["usability"] = None
        df, missing = compute_scores(raw, DIRECTIONS)
        assert "usability" in missing
        for col in df.columns:
            if col.startswith("score_"):
                assert df[col].between(1, 10).all()


class TestSummarize:
    def test_robustness_summary(self):
        df, _ = compute_scores(FIXTURE_RAW, DIRECTIONS)
        robust = summarize_robustness(df, ["design_a", "design_b", "design_c"])
        assert len(robust) == 3
        assert robust["outright_wins"].sum() <= 120

    def test_missing_criterion_blocks_recommendation(self):
        raw = {"a": {"economic": 1, "usability": None}, "b": {"economic": 2, "usability": 3}}
        rec = generate_recommendation(pd.DataFrame(), raw, ["economic", "usability"])
        assert rec is None

    def test_complete_recommendation(self):
        robust = pd.DataFrame([
            {"design_id": "a", "outright_wins": 60, "win_pct": 50, "mean_score": 7.5, "mean_rank": 1.5, "worst_rank": 3, "tied_wins": 5, "mean_margin": 1.2},
            {"design_id": "b", "outright_wins": 40, "win_pct": 33, "mean_score": 6.8, "mean_rank": 2.0, "worst_rank": 3, "tied_wins": 3, "mean_margin": 0.8},
        ])
        raw = {"a": {"economic": 1, "usability": 80}, "b": {"economic": 2, "usability": 70}}
        rec = generate_recommendation(robust, raw, ["economic", "usability"])
        assert rec is not None
        assert rec["recommended_design"] == "a"
