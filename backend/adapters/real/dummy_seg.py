"""A second mock segmenter, used only to prove the registry swap works.

Not part of the app. The acceptance criterion is that pointing adapters.yaml at
this class changes step 3's output with no other edit, so it deliberately
returns an obviously different model_name and quality.
"""

from __future__ import annotations

from contracts import SegmentationInput, SegmentationResult

from ..mock._common import image_size, mask_png, seeded

MODEL_NAME = "dummy-swap-check"


class DummySwapSegmenter:
    def run(self, inp: SegmentationInput) -> SegmentationResult:
        w, h = image_size(inp.image)
        mw = min(w, 512)
        mh = max(1, int(h * mw / max(w, 1)))
        rnd = seeded("dummy", MODEL_NAME, w, h)
        return SegmentationResult(
            is_mock=True,
            mask_png=mask_png(mw, mh, MODEL_NAME, cx_frac=0.5),
            boundary_quality=round(0.5 + rnd.random() * 0.1, 3),
            model_name=MODEL_NAME,
            warnings=["dummy adapter, exists to prove the registry swap"],
        )
