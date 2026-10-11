"""Rule registry: equations, dependency tracing, landmark coordinates."""
from __future__ import annotations

import math
from math import cos, radians, sin

import pytest

from app.config import drafting_config
from app.drafting.derived_dimensions import EXTERNAL_INPUTS, build_registry, registry_inputs
from app.drafting.ease import Settings, resolve_ease
from app.drafting.rules import RuleRegistry
from app.drafting.semantics import DraftingMeasurements


@pytest.fixture(scope="module")
def registry():
    return build_registry()


@pytest.fixture(scope="module")
def resolved(registry, basic_resolution):
    inputs = registry_inputs(basic_resolution.measurements, resolve_ease(Settings()), Settings())
    values, traces = registry.evaluate(inputs)
    return inputs, values, traces


def test_every_equation_is_registered(registry):
    assert len(list(registry)) >= 50
    assert all(r.explanation and r.source_note for r in registry)


def test_registry_inputs_are_drafting_variables_not_source_names():
    for name in EXTERNAL_INPUTS:
        assert name not in {"bust", "waist", "shoulder_w", "vert_bust_line", "bust_line",
                            "waist_line", "back_width", "neck_w", "bust_points",
                            "armscye_depth", "shoulder_incl"}, name


def test_registry_accepts_only_drafting_measurements(registry, basic_resolution):
    inputs = registry_inputs(basic_resolution.measurements, resolve_ease(Settings()), Settings())
    declared = set(EXTERNAL_INPUTS)
    assert set(inputs) <= declared | {t for t in DraftingMeasurements.model_fields}


def test_dependencies_are_traced_and_resolvable(registry, resolved):
    inputs, values, traces = resolved
    produced = {t.output_name for t in traces}
    for trace in traces:
        rule = registry.get(trace.rule_id)
        for dep in rule.required_inputs:
            assert dep in inputs or dep in produced, f"{rule.id} depends on unknown {dep}"
        # no dependency means the value is a declared config constant, not a hidden default
        assert rule.required_inputs or "config" in rule.source_note, rule.id


def test_trace_reports_its_inputs(registry, resolved):
    _inputs, _values, traces = resolved
    trace = next(t for t in traces if t.rule_id == "der.front_bust_half_width")
    assert set(trace.inputs) == {"der.bust_with_ease", "back_width_full"}
    expected = (trace.inputs["der.bust_with_ease"] - trace.inputs["back_width_full"]) / 2
    assert trace.value == pytest.approx(expected)


def test_front_and_back_widths_close_on_the_girth(resolved):
    inputs, values, _ = resolved
    total = values["der.front_bust_half_width"] + values["der.back_bust_half_width"]
    assert total == pytest.approx((inputs["bust_circumference"] + inputs["ease_bust"]) / 2, abs=1e-9)


def test_shoulder_run_follows_the_slope(resolved):
    _inputs, values, _ = resolved
    length = values["der.shoulder_seam_length"]
    slope = radians(values["der.shoulder_slope_deg"])
    assert values["der.shoulder_tip_dx"] == pytest.approx(length * cos(slope))
    assert values["der.shoulder_tip_dy"] == pytest.approx(length * sin(slope))
    assert math.hypot(values["der.shoulder_tip_dx"], values["der.shoulder_tip_dy"]) == pytest.approx(length)


def test_half_shoulder_is_derived_from_the_full_width(resolved):
    inputs, values, _ = resolved
    assert values["der.shoulder_half_width"] == pytest.approx(inputs["shoulder_width_full"] / 2)


def test_shoulder_slope_is_clamped_and_reported(resolved):
    inputs, values, _ = resolved
    lo, hi = drafting_config()["ranges"]["shoulder_slope_deg"]
    assert inputs["shoulder_slope_degrees"] == pytest.approx(45.0)
    assert values["der.shoulder_slope_deg"] == pytest.approx(hi)


def test_neckline_geometry_follows_its_constants(resolved):
    _inputs, values, _ = resolved
    c = drafting_config()["constants"]
    assert values["der.neck_shoulder_x"] == pytest.approx(values["neck_width_half"] * c["neck_shoulder_share"])
    assert values["der.front_neck_depth"] == pytest.approx(
        values["der.neck_shoulder_x"] * c["front_neck_depth_ratio"])


def test_dart_intake_follows_the_apex_spacing(resolved):
    _inputs, values, _ = resolved
    c = drafting_config()["constants"]
    assert values["der.dart_intake"] == pytest.approx(values["bust_apex_spacing_full"] * c["dart_intake_ratio"])
    assert values["lm.front.dart_half_intake"] == pytest.approx(values["der.dart_intake"] / 2)


def test_dart_intake_override_wins(basic_resolution):
    settings = Settings(dart_intake_mm=42.0)
    inputs = registry_inputs(basic_resolution.measurements, resolve_ease(settings), settings)
    values, _traces = build_registry().evaluate(inputs)
    assert values["der.dart_intake"] == pytest.approx(42.0)


def test_shoulder_slope_override_wins(basic_resolution):
    settings = Settings(shoulder_slope_deg=18.0)
    inputs = registry_inputs(basic_resolution.measurements, resolve_ease(settings), settings)
    values, _traces = build_registry().evaluate(inputs)
    assert values["der.shoulder_slope_deg"] == pytest.approx(18.0)


def test_bust_apex_is_held_between_the_dart_legs(resolved):
    _inputs, values, _ = resolved
    assert values["lm.front.dart_top_y"] < values["lm.front.bust_apex_y"] < values["lm.front.dart_bottom_y"]


def test_underarm_sits_on_the_shoulder_tip_plus_depth(resolved):
    _inputs, values, _ = resolved
    for side in ("front", "back"):
        assert values[f"lm.{side}.underarm_y"] == pytest.approx(
            values[f"lm.{side}.shoulder_tip_y"] + values["der.armscye_depth_mm"])


def test_waist_level_is_never_a_raw_source_value(registry, resolved):
    inputs, values, _ = resolved
    # the rule chain reaches the waist through the confirmed pattern-frame level
    assert inputs["neck_to_waist_level"] == pytest.approx(411.0)
    assert values["lm.front.waist_side_y"] == pytest.approx(inputs["neck_to_waist_level"])


def test_registry_rejects_out_of_order_rules():
    from app.drafting.rules import DraftingRule
    first = DraftingRule("a", "a", "a.out", ["b.out"], lambda v: 0.0, "x", "y")
    second = DraftingRule("b", "b", "b.out", [], lambda v: 0.0, "x", "y")
    with pytest.raises(ValueError, match="not produced by an earlier rule"):
        RuleRegistry([first, second])


def test_registry_is_deterministic(basic_resolution):
    settings = Settings()
    inputs = registry_inputs(basic_resolution.measurements, resolve_ease(settings), settings)
    first = build_registry().evaluate(inputs)[0]
    second = build_registry().evaluate(inputs)[0]
    assert first == second
