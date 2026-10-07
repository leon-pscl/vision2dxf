"""Step 2 mock: capture quality checks.

Pretends to judge distance, pose, framing and clothing from the image bytes.
Thresholds come from config.yaml so the phase-2 pose adapter reads the same
bands.
"""

from __future__ import annotations

from contracts import CaptureInput, CaptureResult

from ._common import cfg, image_size, seeded

_CHECKS = [
    ("distance_ok", "distance", "Move closer or further until you fill 55-95% of the frame height."),
    ("pose_ok", "pose", "Stand square with arms relaxed at 10-20 degrees off the body."),
    ("full_body_visible", "framing", "Head to feet must be inside the frame, feet visible."),
    ("clothing_ok", "clothing", "Wear close-fitting clothing so the silhouette is readable."),
]


class MockCapture:
    def run(self, inp: CaptureInput) -> CaptureResult:
        conf = cfg()["capture"]
        guidance: list[str] = []

        # deterministic pseudo-verdicts, biased to pass so the demo clicks through
        rnd = seeded("capture", len(inp.front_image), len(inp.side_image), inp.height_cm)
        verdicts = {name: rnd.random() > 0.15 for name, _, _ in _CHECKS}
        verdicts["full_body_visible"] = rnd.random() > 0.08

        if not verdicts["distance_ok"]:
            guidance.append(_guidance(conf, "bbox_height_fraction"))

        for key, label, text in _CHECKS:
            if not verdicts[key]:
                guidance.append(f"{label.upper()}: {text}")

        bmi = None
        if inp.weight_kg:
            bmi = inp.weight_kg / (inp.height_cm / 100) ** 2
            if bmi > 30:
                guidance.append("BMI over 30: expect larger ease recommendations.")

        return CaptureResult(
            is_mock=True,
            distance_ok=verdicts["distance_ok"],
            pose_ok=verdicts["pose_ok"],
            full_body_visible=verdicts["full_body_visible"],
            clothing_ok=verdicts["clothing_ok"],
            guidance=guidance,
            warnings=[
                "mock capture check, no image was analysed"
                + (f"; BMI {bmi:.1f}" if bmi else "")
            ],
        )


def _guidance(conf: dict, key: str) -> str:
    lo, hi = conf[key]
    return f"DISTANCE: you should fill {lo:.0%}-{hi:.0%} of the frame height."
