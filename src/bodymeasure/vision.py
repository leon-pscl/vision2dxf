"""YOLO26 segmentation and pose, loaded once and reused (brief 5.3, M3).

Model paths come from ``config/app.yaml`` so fine-tuned weights can replace the
pretrained ones without touching code. Both models are loaded on the first call
and kept, because loading them per request dominates the runtime on a CPU-only
station.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from bodymeasure.config import ROOT, load_app_config
from bodymeasure.mask_clean import clean_mask

if TYPE_CHECKING:  # pragma: no cover - import cost only matters at runtime
    from ultralytics import YOLO
    from ultralytics.engine.results import Results

PERSON_CLASS = 0
_MASK_BINARY_THRESHOLD = 0.5


@dataclass(frozen=True)
class ViewAnalysis:
    """One view's cleaned mask, keypoints and the CPU time each model took."""

    mask: np.ndarray
    keypoints: np.ndarray
    seg_ms: float
    pose_ms: float

    @property
    def total_ms(self) -> float:
        """Return the combined model time in milliseconds."""
        return self.seg_ms + self.pose_ms


def model_paths() -> tuple[Path, Path]:
    """Return the configured segmentation and pose weight paths."""
    models = load_app_config().models
    return ROOT / models.seg, ROOT / models.pose


def weights_available() -> bool:
    """Return whether both weight files are present on disk."""
    return all(path.exists() for path in model_paths())


@lru_cache(maxsize=1)
def load_models() -> tuple[YOLO, YOLO]:
    """Load both YOLO26 models once and cache them.

    Raises:
        FileNotFoundError: if either weight file is missing.

    """
    from ultralytics import YOLO

    seg_path, pose_path = model_paths()
    for path in (seg_path, pose_path):
        if not path.exists():
            raise FileNotFoundError(
                f"model weights missing: {path}. Fetch them with "
                f"\"python -c 'from ultralytics import YOLO; YOLO(\\\"{path.name}\\\")'\" "
                f"and move the file into {path.parent}"
            )
    return YOLO(str(seg_path)), YOLO(str(pose_path))


def _largest_person_mask(
    result: Results, shape: tuple[int, int], min_conf: float
) -> np.ndarray:
    """Return the largest person instance mask, resized to ``shape``.

    Raises:
        ValueError: if no person passes ``min_conf``.

    """
    masks = getattr(result, "masks", None)
    if masks is None or masks.data is None or len(masks.data) == 0:
        raise ValueError("No person found in the frame")

    classes = result.boxes.cls.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    data = masks.data.cpu().numpy()
    candidates = [
        index
        for index in range(len(data))
        if int(classes[index]) == PERSON_CLASS and float(confidences[index]) >= min_conf
    ]
    if not candidates:
        raise ValueError("No person found in the frame")

    best = max(candidates, key=lambda index: float(data[index].sum()))
    mask = data[best]
    if mask.shape != shape:
        import cv2

        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask > _MASK_BINARY_THRESHOLD


def _best_keypoints(result: Results) -> np.ndarray:
    """Return the most confident person's COCO-17 keypoints as ``(17, 3)``.

    Raises:
        ValueError: if the pose model found nobody.

    """
    keypoints = getattr(result, "keypoints", None)
    if keypoints is None or keypoints.data is None or len(keypoints.data) == 0:
        raise ValueError("No person found in the frame")
    data = keypoints.data.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    return data[int(np.argmax(confidences))].astype(float)


def analyse(image: np.ndarray) -> ViewAnalysis:
    """Segment and pose one prepared frame, and clean the mask.

    Raises:
        ValueError: if no person is found by either model.
        FileNotFoundError: if the weights are missing.

    """
    config = load_app_config()
    seg_model, pose_model = load_models()
    shape = (image.shape[0], image.shape[1])

    started = time.perf_counter()
    seg_result = seg_model.predict(
        image,
        imgsz=config.models.imgsz,
        retina_masks=config.models.retina_masks,
        conf=config.thresholds["mask_conf"],
        verbose=False,
    )[0]
    seg_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    pose_result = pose_model.predict(
        image, imgsz=config.models.imgsz, conf=config.thresholds["mask_conf"], verbose=False
    )[0]
    pose_ms = (time.perf_counter() - started) * 1000

    raw = _largest_person_mask(seg_result, shape, config.thresholds["mask_conf"])
    return ViewAnalysis(
        mask=clean_mask(
            raw,
            int(config.thresholds["morph_kernel_px"]),
            int(config.thresholds["hole_fill_max_area_px"]),
        ),
        keypoints=_best_keypoints(pose_result),
        seg_ms=seg_ms,
        pose_ms=pose_ms,
    )
