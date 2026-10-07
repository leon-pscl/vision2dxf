"""Step 3 mock: segmentation."""

from __future__ import annotations

from contracts import SegmentationInput, SegmentationResult

from ._common import image_size, mask_png, seeded

MODEL_NAME = "mock-silhouette-v1"


class MockSegmentation:
    def run(self, inp: SegmentationInput) -> SegmentationResult:
        w, h = image_size(inp.image)

        # If the UI sent a prompt, honour it: the prompt shifts the mask so the
        # user can see the prompts are wired through, not ignored.
        offset = 0.0
        if inp.box:
            box_cx = (inp.box[0] + inp.box[2]) / 2
            offset = (box_cx - w / 2) / max(w, 1)
        elif inp.points:
            offset = (inp.points[0][0] - w / 2) / max(w, 1)

        rnd = seeded("seg", MODEL_NAME, w, h, round(offset, 4))
        quality = round(0.82 + rnd.random() * 0.15, 3)

        # downscale for the mask; the UI scales it back over the photo
        mw = min(w, 512)
        mh = max(1, int(h * mw / max(w, 1)))

        cx = min(0.85, max(0.15, 0.5 + offset))

        return SegmentationResult(
            is_mock=True,
            mask_png=mask_png(mw, mh, f"{MODEL_NAME}:{offset:.3f}", cx_frac=cx),
            boundary_quality=quality,
            model_name=MODEL_NAME,
            warnings=["mock mask, generated geometrically, not from the photo"],
        )
