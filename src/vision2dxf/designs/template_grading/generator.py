from __future__ import annotations
import json
from pathlib import Path
from ...core.models import MeasurementProfile, PatternResult
from .selector import select_template
from .grader import apply_grading

_TEMPLATES_DIR = Path(__file__).resolve().parents[4] / "data" / "templates"
_INDEX_PATH = _TEMPLATES_DIR / "template_index.json"


def _load_index() -> dict:
    with open(_INDEX_PATH) as f:
        return json.load(f)


class TemplateGradingGenerator:
    design_id = "design_c_template"

    def generate(self, m: MeasurementProfile) -> PatternResult:
        index = _load_index()
        template_info, distance = select_template(index, m)

        tpl_path = _TEMPLATES_DIR / template_info["file"]
        with open(tpl_path) as f:
            template = json.load(f)

        ref = template["reference_measurements"]
        ease = m.ease

        adjusted_points, grading_info = apply_grading(
            template["points"], template.get("grading_limits", {}),
            ref, m, ease,
        )

        segments = template["segments"]

        return PatternResult(
            design_id=self.design_id,
            points=adjusted_points,
            segments=segments,
            valid=False,
            validation_errors=[],
            metadata={
                "selected_template": template_info["id"],
                "distance": round(distance, 6),
                "grading_info": grading_info,
                "ease_applied": ease,
                "units": "cm",
                "status": "prototype_only",
            },
        )
