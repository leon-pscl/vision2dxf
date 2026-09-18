from __future__ import annotations
from vision2dxf.core.models import MeasurementProfile
from vision2dxf.designs.formula_based.generator import FormulaBasedGenerator
from vision2dxf.designs.parametric_block.generator import ParametricBlockGenerator
from vision2dxf.designs.template_grading.generator import TemplateGradingGenerator

import pytest

SAMPLE = MeasurementProfile(
    profile_id="P001", chest=92.0, waist=76.0, hip=98.0,
    shoulder=40.0, torso_length=43.0, armhole_depth=22.0,
    sleeve_length=59.0, ease=4.0,
)

SMALL = MeasurementProfile(
    profile_id="P010", chest=82.0, waist=66.0, hip=88.0,
    shoulder=35.0, torso_length=38.0, armhole_depth=19.5,
    sleeve_length=54.0, ease=4.0,
)


class TestDesignA:
    def test_generates_bodice(self):
        gen = FormulaBasedGenerator()
        result = gen.generate(SAMPLE)
        assert len(result.points) == 9
        assert len(result.segments) == 7

    def test_formula_driven(self):
        gen = FormulaBasedGenerator()
        result = gen.generate(SAMPLE)
        assert result.metadata["status"] == "prototype_only"

    def test_no_template_loaded(self):
        gen = FormulaBasedGenerator()
        result = gen.generate(SAMPLE)
        assert "base_block" not in result.metadata
        assert "selected_template" not in result.metadata


class TestDesignB:
    def test_transforms_block(self):
        gen = ParametricBlockGenerator()
        result = gen.generate(SAMPLE)
        assert len(result.points) == 9
        assert result.metadata["base_block"] == "medium"

    def test_uses_master_block(self):
        gen = ParametricBlockGenerator()
        r1 = gen.generate(SAMPLE)
        r2 = gen.generate(SMALL)
        assert r1.points != r2.points


class TestDesignC:
    def test_selects_template(self):
        gen = TemplateGradingGenerator()
        result = gen.generate(SAMPLE)
        assert "selected_template" in result.metadata
        assert result.metadata["selected_template"] in ("small", "medium", "large")

    def test_records_grading(self):
        gen = TemplateGradingGenerator()
        result = gen.generate(SAMPLE)
        assert "grading_info" in result.metadata

    def test_smaller_profile_selects_smaller_template(self):
        gen = TemplateGradingGenerator()
        result = gen.generate(SMALL)
        assert result.metadata["selected_template"] == "small"
