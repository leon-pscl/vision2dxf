from __future__ import annotations
from vision2dxf.core.models import MeasurementProfile, PatternGenerator
from vision2dxf.designs.formula_based.generator import FormulaBasedGenerator
from vision2dxf.designs.parametric_block.generator import ParametricBlockGenerator
from vision2dxf.designs.template_grading.generator import TemplateGradingGenerator
from vision2dxf.core.validation import validate_pattern

import pytest

ALL_GENERATORS = [FormulaBasedGenerator(), ParametricBlockGenerator(), TemplateGradingGenerator()]

SAMPLE = MeasurementProfile(
    profile_id="P001", chest=92.0, waist=76.0, hip=98.0,
    shoulder=40.0, torso_length=43.0, armhole_depth=22.0,
    sleeve_length=59.0, ease=4.0,
)


class TestContract:
    def test_all_implement_interface(self):
        for gen in ALL_GENERATORS:
            assert hasattr(gen, "design_id")
            assert hasattr(gen, "generate")
            assert callable(gen.generate)

    def test_all_accept_same_measurement(self):
        for gen in ALL_GENERATORS:
            result = gen.generate(SAMPLE)
            assert result.design_id == gen.design_id
            assert isinstance(result.points, dict)
            assert isinstance(result.segments, list)

    def test_output_schema(self):
        for gen in ALL_GENERATORS:
            result = gen.generate(SAMPLE)
            assert "design_id" in result.__dict__
            assert "points" in result.__dict__
            assert "segments" in result.__dict__
            assert "valid" in result.__dict__
            assert "validation_errors" in result.__dict__
            assert "metadata" in result.__dict__


class TestValidation:
    def test_required_landmarks_present(self):
        required = {"center_front_neck", "side_neck", "shoulder_tip",
                     "armhole_bottom", "side_waist", "center_front_waist"}
        for gen in ALL_GENERATORS:
            result = gen.generate(SAMPLE)
            assert required.issubset(result.points.keys()), f"{gen.design_id} missing landmarks"

    def test_coordinates_finite(self):
        for gen in ALL_GENERATORS:
            result = gen.generate(SAMPLE)
            for name, (x, y) in result.points.items():
                assert isinstance(x, (int, float))
                assert isinstance(y, (int, float))

    def test_validation_runs(self):
        for gen in ALL_GENERATORS:
            result = gen.generate(SAMPLE)
            result = validate_pattern(result)
            assert isinstance(result.valid, bool)
            assert isinstance(result.validation_errors, list)
