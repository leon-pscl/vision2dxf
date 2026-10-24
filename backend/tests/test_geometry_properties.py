"""Property and fixture-schema tests for the drafting geometry."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.config import drafting_config, measurement_index
from app.drafting.ease import Settings
from app.drafting.measurements import parse_source
from app.drafting.pipeline import draft
from app.drafting.semantics import resolve

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
TOL = 1e-6


@pytest.mark.parametrize("name", [
    "body_measurements_small.yaml",
    "body_measurements_medium.yaml",
    "body_measurements_large.yaml",
])
def test_valid_fixture_passes_schema(name, default_confirmations):
    profile = parse_source((FIXTURES / name).read_text(encoding="utf-8"), name)
    resolution = resolve(profile.values_cm, default_confirmations)
    assert resolution.can_generate
    result = draft(resolution, Settings())
    assert result.report.status == "pass"


def test_achieved_bust_waist_shoulder_match_requested(basic_resolution):
    result = draft(basic_resolution, Settings())
    v = result.values
    achieved_bust = 2 * (v["der.front_bust_half_width"] + v["der.back_bust_half_width"])
    requested_bust = v["der.bust_with_ease"]
    achieved_waist = 2 * (v["der.front_waist_half_width"] + v["der.back_waist_half_width"])
    requested_waist = v["der.waist_with_ease"]
    assert achieved_bust == pytest.approx(requested_bust, abs=TOL)
    assert achieved_waist == pytest.approx(requested_waist, abs=TOL)


def test_achieved_shoulder_seam_matches_requested(basic_resolution):
    result = draft(basic_resolution, Settings())
    v = result.values
    assert v["der.shoulder_seam_length"] == pytest.approx(
        v["der.shoulder_half_width"] - drafting_config()["constants"]["shoulder_neck_offset_mm"],
        abs=TOL,
    )


def test_outer_contours_are_closed(basic_result):
    tol = drafting_config()["tolerances"]["panel_closure_mm"]
    for panel in basic_result.document.panels:
        contour = panel.outer_contour()
        assert contour[0].dist(panel.edges[-1].end) <= tol


@pytest.mark.parametrize("name", [
    "body_measurements_small.yaml",
    "body_measurements_medium.yaml",
    "body_measurements_large.yaml",
])
def test_fixture_values_are_valid(name):
    profile = parse_source((FIXTURES / name).read_text(encoding="utf-8"), name)
    assert set(profile.values_cm) <= set(measurement_index())