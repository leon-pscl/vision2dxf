from __future__ import annotations
import copy
import json
from pathlib import Path
from ...core.models import MeasurementProfile, PatternResult
from .transformer import transform_block

_BLOCK_PATH = Path(__file__).resolve().parents[4] / "data" / "base_blocks" / "bodice_medium.json"


def _load_block() -> dict:
    with open(_BLOCK_PATH) as f:
        return json.load(f)


class ParametricBlockGenerator:
    design_id = "design_b_parametric"

    def generate(self, m: MeasurementProfile) -> PatternResult:
        block = _load_block()
        ref = block["reference_measurements"]
        ease = m.ease

        points = transform_block(block["points"], ref, m, ease)
        segments = copy.deepcopy(block["segments"])

        return PatternResult(
            design_id=self.design_id,
            points=points,
            segments=segments,
            valid=False,
            validation_errors=[],
            metadata={
                "base_block": block["pattern_id"],
                "ease_applied": ease,
                "units": "cm",
                "status": "prototype_only",
            },
        )
