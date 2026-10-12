"""YOLO26 segmentation and pose, loaded once and reused (brief 5.3, M3).

Model paths come from ``config/app.yaml`` so fine-tuned weights can replace the
pretrained ones without touching code. Both models are loaded on the first call
and kept, because loading them per request dominates the runtime on a CPU-only
station.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from bodymeasure.config import ROOT, load_app_config
from bodymeasure.mask_clean import clean_mask

if TYPE_CHECKING:  # pragma: no cover - import cost only matters at runtime
    from ultralytics import YOLO
    from ultralytics.engine.results import Results

PERSON_CLASS = 0
_MASK_BINARY_THRESHOLD = 0.5
# One body is often detected twice - a torso, and the bare legs below it - and the
# second half scores far lower (0.26 against 0.83 on a real capture). Detections
# are therefore filtered low and then by size: a region within
# MERGE_AREA_FRACTION of the biggest one is part of the same body, and anything
# smaller is not, which is what keeps a second person out of the subject's mask.
MERGE_CONF = 0.15
MERGE_AREA_FRACTION = 0.3


@dataclass(frozen=True)
class ViewAnalysis:
    """One view's masks, detection, keypoints and the CPU time each model took.

    ``raw_mask`` is the model's own output and ``mask`` the cleaned one; both are
    kept because the operator has to be able to see what cleanup changed
    (brief 8, stages 06 and 07).
    """

    mask: np.ndarray
    keypoints: np.ndarray
    seg_ms: float
    pose_ms: float
    raw_mask: np.ndarray | None = None
    box: tuple[int, int, int, int] = (0, 0, 0, 0)
    confidence: float = 0.0
    instances: int = 0

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


def file_sha256(path: Path) -> str:
    """Return the SHA-256 of a file, read in blocks so a big weight file fits."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_weights() -> dict[str, str]:
    """Return each weight file's hash, raising if a configured one does not match.

    Raises:
        FileNotFoundError: if a weight file is missing. Nothing is ever
            downloaded: a station is provisioned before it is used (brief 5).
        ValueError: if a configured hash does not match the file on disk.

    """
    models = load_app_config().models
    digests = {}
    for name, path, expected in (
        ("seg", ROOT / models.seg, models.seg_sha256),
        ("pose", ROOT / models.pose, models.pose_sha256),
    ):
        if not path.exists():
            raise FileNotFoundError(f"model weights missing: {path}")
        digests[name] = file_sha256(path)
        if expected and digests[name] != expected:
            raise ValueError(
                f"{path.name} is not the configured weight file: "
                f"expected {expected}, found {digests[name]}. "
                f"Provision the station with tools/provision_models.py."
            )
    return digests


@lru_cache(maxsize=1)
def load_models() -> tuple[YOLO, YOLO]:
    """Load both YOLO26 models once and cache them.

    Raises:
        FileNotFoundError: if either weight file is missing.
        ValueError: if a configured hash does not match the file on disk.

    """
    from ultralytics import YOLO

    seg_path, pose_path = model_paths()
    for path in (seg_path, pose_path):
        if not path.exists():
            raise FileNotFoundError(
                f"model weights missing: {path}. Provision the station with "
                f"\"python tools/provision_models.py\" and move the file into {path.parent}"
            )
    verify_weights()
    return YOLO(str(seg_path)), YOLO(str(pose_path))


@dataclass(frozen=True)
class Detection:
    """The chosen person: the union mask, its box, its confidence, its rivals."""

    mask: np.ndarray
    box: tuple[int, int, int, int]
    confidence: float
    instances: int


def _largest_person_mask(
    result: Results, shape: tuple[int, int], min_conf: float
) -> Detection:
    """Return the subject's mask, rejoining a body the model split in two.

    Every person instance within ``MERGE_AREA_FRACTION`` of the biggest one is
    unioned in, so a torso and the legs detected separately make one silhouette.
    A smaller person-shaped region is left out, which is what keeps a second
    person, a poster or a dog out of the subject's mask.

    Raises:
        ValueError: if no person-shaped region passes ``min_conf``.

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

    floor = max(float(data[index].sum()) for index in candidates) * MERGE_AREA_FRACTION
    kept = [index for index in candidates if float(data[index].sum()) >= floor]
    mask = np.zeros(shape, dtype=bool)
    for index in kept:
        part = data[index]
        if part.shape != shape:
            part = cv2.resize(part, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        mask |= part > _MASK_BINARY_THRESHOLD

    rows = np.flatnonzero(mask.any(axis=1))
    columns = np.flatnonzero(mask.any(axis=0))
    box = (
        (int(columns.min()), int(rows.min()), int(columns.max()), int(rows.max()))
        if rows.size
        else (0, 0, 0, 0)
    )
    return Detection(
        mask=mask,
        box=box,
        confidence=max(float(confidences[index]) for index in kept),
        instances=len(candidates),
    )


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
        conf=MERGE_CONF,
        verbose=False,
    )[0]
    seg_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    pose_result = pose_model.predict(
        image, imgsz=config.models.imgsz, conf=config.thresholds["mask_conf"], verbose=False
    )[0]
    pose_ms = (time.perf_counter() - started) * 1000

    detection = _largest_person_mask(seg_result, shape, MERGE_CONF)
    return ViewAnalysis(
        mask=clean_mask(
            detection.mask,
            int(config.thresholds["morph_kernel_px"]),
            int(config.thresholds["hole_fill_max_area_px"]),
        ),
        keypoints=_best_keypoints(pose_result),
        seg_ms=seg_ms,
        pose_ms=pose_ms,
        raw_mask=detection.mask,
        box=detection.box,
        confidence=detection.confidence,
        instances=detection.instances,
    )
