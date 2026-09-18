from __future__ import annotations
import math
from .models import PatternResult


def validate_pattern(result: PatternResult, bounds: dict | None = None) -> PatternResult:
    errors: list[str] = []
    required_landmarks = [
        "center_front_neck", "side_neck", "shoulder_tip",
        "armhole_bottom", "side_waist", "center_front_waist",
    ]

    for lm in required_landmarks:
        if lm not in result.points:
            errors.append(f"Missing required landmark: {lm}")

    for name, (x, y) in result.points.items():
        if not (math.isfinite(x) and math.isfinite(y)):
            errors.append(f"Non-finite coordinate at {name}: ({x}, {y})")

    if required_landmarks[0] in result.points and required_landmarks[4] in result.points:
        cw = result.points[required_landmarks[4]]
        cfw = result.points[required_landmarks[5]]
        width = abs(cw[0] - cfw[0])
        if width <= 0:
            errors.append("Width is not positive")

    if required_landmarks[0] in result.points and required_landmarks[5] in result.points:
        top = result.points[required_landmarks[0]]
        bot = result.points[required_landmarks[5]]
        height = abs(bot[1] - top[1])
        if height <= 0:
            errors.append("Height is not positive")

    for seg in result.segments:
        fr = seg.get("from", "")
        to = seg.get("to", "")
        if fr in result.points and to in result.points:
            p1 = result.points[fr]
            p2 = result.points[to]
            if math.isclose(p1[0], p2[0]) and math.isclose(p1[1], p2[1]):
                errors.append(f"Zero-length segment: {fr} -> {to}")

    if bounds:
        for name, (x, y) in result.points.items():
            bx = bounds.get("x", [-100, 200])
            by = bounds.get("y", [-100, 200])
            if not (bx[0] <= x <= bx[1] and by[0] <= y <= by[1]):
                errors.append(f"Point {name} out of bounds: ({x}, {y})")

    result.validation_errors = errors
    result.valid = len(errors) == 0
    return result
