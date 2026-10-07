"""Step 4 mock: calibration.

The real adapter derives px_per_cm from mask top/bottom rows over height_cm.
This mock does the same arithmetic on assumed image geometry, so the numbers
land in a believable range and the uncertainty maths is real.
"""

from __future__ import annotations

from contracts import CalibMethod, CalibrationResult

from ._common import cfg, seeded


class MockCalibration:
    def run(self, image_sizes: dict[str, int], height_cm: float) -> CalibrationResult:
        conf = cfg()["calibration"]
        method = CalibMethod(conf["method"])
        px_err = float(conf["pixel_error_px"])

        px_per_cm: dict[str, float] = {}
        for view, (w, h) in image_sizes.items():
            rnd = seeded("calib", view, w, h, height_cm)
            # assume the figure fills 55-95% of the frame (config capture band)
            lo, hi = cfg()["capture"]["bbox_height_fraction"]
            fill = lo + rnd.random() * (hi - lo)
            px_per_cm[view] = round((h * fill) / height_cm, 3)

        # relative uncertainty: two end errors over the pixel span between them
        span_cm = height_cm
        rel = (2 * px_err) / (span_cm * min(px_per_cm.values()))
        rel = round(min(rel, 0.25), 4)

        return CalibrationResult(
            is_mock=True,
            method=method,
            px_per_cm=px_per_cm,
            relative_uncertainty=rel,
            warnings=[f"mock calibration, assumes ±{px_err:.0f} px endpoint error"],
        )
