from __future__ import annotations
from vision2dxf.core.models import PatternResult
from vision2dxf.core.validation import validate_pattern
import pytest


def _valid_result():
    return PatternResult(
        design_id="test",
        points={
            "center_front_neck": (0.0, 0.0),
            "side_neck": (7.4, 0.0),
            "shoulder_tip": (19.4, 3.5),
            "armhole_bottom": (24.0, 22.0),
            "side_waist": (24.0, 43.0),
            "center_front_waist": (0.0, 43.0),
        },
        segments=[
            {"type": "line", "from": "center_front_neck", "to": "side_neck"},
            {"type": "line", "from": "side_neck", "to": "shoulder_tip"},
        ],
        valid=False,
        validation_errors=[],
    )


class TestValidation:
    def test_valid_pattern(self):
        r = validate_pattern(_valid_result())
        assert r.valid

    def test_missing_landmark(self):
        r = _valid_result()
        del r.points["shoulder_tip"]
        r = validate_pattern(r)
        assert not r.valid
        assert any("shoulder_tip" in e for e in r.validation_errors)

    def test_zero_width(self):
        r = _valid_result()
        r.points["side_waist"] = (0.0, 43.0)
        r = validate_pattern(r)
        assert not r.valid

    def test_zero_height(self):
        r = _valid_result()
        r.points["center_front_waist"] = (0.0, 0.0)
        r = validate_pattern(r)
        assert not r.valid

    def test_bounds_check(self):
        r = _valid_result()
        r = validate_pattern(r, bounds={"x": [-10, 10], "y": [-10, 10]})
        assert not r.valid
