"""Vision wrappers. The model calls are skipped when the weights are absent."""

from pathlib import Path

import numpy as np
import pytest
from tests.synthetic.people import make_person

from bodymeasure import overlay, vision

needs_weights = pytest.mark.skipif(
    not vision.weights_available(), reason="YOLO26 weights not in models/"
)


def test_model_paths_come_from_the_config() -> None:
    seg_path, pose_path = vision.model_paths()
    assert seg_path.name == "yolo26s-seg.pt"
    assert pose_path.name == "yolo26s-pose.pt"
    assert seg_path.is_absolute()


def test_missing_weights_say_how_to_fetch_them() -> None:
    if vision.weights_available():
        pytest.skip("weights are present")
    vision.load_models.cache_clear()
    with pytest.raises(FileNotFoundError, match="model weights missing"):
        vision.load_models()


def test_total_time_adds_the_two_models() -> None:
    analysis = vision.ViewAnalysis(
        mask=np.zeros((4, 4), dtype=bool), keypoints=np.zeros((17, 3)), seg_ms=120.0, pose_ms=80.0
    )
    assert analysis.total_ms == 200.0


def test_mask_outline_marks_the_edge_without_touching_the_input() -> None:
    person = make_person()
    image = np.zeros((*person.mask.shape, 3), dtype=np.uint8)
    traced = overlay.draw_mask_outline(image, person.mask)

    assert traced.shape == image.shape
    assert image.sum() == 0, "the input image must not be modified"
    assert traced.sum() > 0


def test_keypoints_are_drawn_hollow_below_the_threshold() -> None:
    person = make_person()
    image = np.zeros((*person.mask.shape, 3), dtype=np.uint8)
    weak = person.keypoints.copy()
    weak[:, 2] = 0.1

    strong_ink = overlay.draw_keypoints(image, person.keypoints).sum()
    weak_ink = overlay.draw_keypoints(image, weak).sum()

    assert 0 < weak_ink < strong_ink


def test_view_overlay_draws_the_floor_line() -> None:
    person = make_person()
    image = np.zeros((*person.mask.shape, 3), dtype=np.uint8)
    without = overlay.view_overlay(image, person.mask, person.keypoints)
    with_floor = overlay.view_overlay(image, person.mask, person.keypoints, person.floor_row)

    assert with_floor.sum() > without.sum()
    assert with_floor[person.floor_row, 10].any()


def test_save_writes_a_png_and_creates_the_folder(tmp_path: Path) -> None:
    import cv2

    path = overlay.save(np.full((20, 20, 3), 128, dtype=np.uint8), tmp_path / "debug" / "x.png")
    assert path.exists()
    assert cv2.imread(str(path)).shape == (20, 20, 3)


@needs_weights
def test_the_models_run_and_report_cpu_time_per_image() -> None:
    """Timing on a station-sized frame. Content barely affects inference cost."""
    import time

    from bodymeasure.config import load_app_config

    seg_model, pose_model = vision.load_models()
    imgsz = load_app_config().models.imgsz
    image = np.full((1800, 900, 3), 200, dtype=np.uint8)

    timings = {}
    for name, model in (("seg", seg_model), ("pose", pose_model)):
        started = time.perf_counter()
        model.predict(image, imgsz=imgsz, verbose=False)
        timings[name] = (time.perf_counter() - started) * 1000

    assert timings["seg"] > 0 and timings["pose"] > 0
    print(
        f"\nCPU time per image: seg {timings['seg']:.0f} ms, "
        f"pose {timings['pose']:.0f} ms"
    )


@needs_weights
def test_models_are_loaded_only_once() -> None:
    first = vision.load_models()
    assert vision.load_models() is first


class _Tensor:
    """Smallest stand-in for the torch tensors ultralytics returns."""

    def __init__(self, array: np.ndarray) -> None:
        self._array = array

    def cpu(self) -> "_Tensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._array

    def __len__(self) -> int:
        return len(self._array)


class _Result:
    """A stand-in for ultralytics' Results, so the parsing can be tested offline."""

    def __init__(
        self,
        masks: np.ndarray | None = None,
        classes: np.ndarray | None = None,
        confidences: np.ndarray | None = None,
        keypoints: np.ndarray | None = None,
    ) -> None:
        self.masks = type("M", (), {"data": _Tensor(masks)})() if masks is not None else None
        self.keypoints = (
            type("K", (), {"data": _Tensor(keypoints)})() if keypoints is not None else None
        )
        self.boxes = type(
            "B",
            (),
            {
                "cls": _Tensor(classes if classes is not None else np.zeros(0)),
                "conf": _Tensor(confidences if confidences is not None else np.zeros(0)),
            },
        )()


def _two_people() -> np.ndarray:
    masks = np.zeros((3, 40, 40), dtype=float)
    masks[0, 0:10, 0:10] = 1.0  # small person
    masks[1, 0:30, 0:30] = 1.0  # large person
    masks[2, 0:35, 0:35] = 1.0  # largest, but not a person
    return masks


def test_the_largest_person_wins_and_other_classes_are_ignored() -> None:
    result = _Result(_two_people(), np.array([0, 0, 16]), np.array([0.9, 0.8, 0.99]))
    mask = vision._largest_person_mask(result, (40, 40), 0.5)
    assert mask[20, 20], "should pick the larger of the two people"
    assert not mask[32, 32], "the dog is not a person"
    assert mask.sum() == 900, "the smaller person must not be merged in"


def _split_body() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One body detected twice, as a real capture does: a torso and bare legs."""
    masks = np.zeros((2, 40, 40), dtype=float)
    masks[0, 0:20, 0:20] = 1.0  # head to mid-thigh, confident
    masks[1, 20:40, 0:20] = 1.0  # hips to feet, scored low
    return masks, np.array([0, 0]), np.array([0.83, 0.26])


def test_a_body_split_into_two_instances_is_rejoined() -> None:
    """Half a mask would scale every measurement off the wrong floor row."""
    masks, classes, confidences = _split_body()
    mask = vision._largest_person_mask(_Result(masks, classes, confidences), (40, 40), 0.15)
    assert mask.sum() == 800, "both halves are one body"
    assert mask[5, 5] and mask[35, 5]


def test_people_below_the_confidence_threshold_are_dropped() -> None:
    result = _Result(_two_people(), np.array([0, 0, 0]), np.array([0.9, 0.3, 0.2]))
    mask = vision._largest_person_mask(result, (40, 40), 0.5)
    assert mask.sum() == 100, "only the small, confident person survives"


def test_a_mask_at_model_resolution_is_resized_to_the_frame() -> None:
    result = _Result(_two_people(), np.array([0, 0, 0]), np.array([0.9, 0.9, 0.9]))
    mask = vision._largest_person_mask(result, (80, 80), 0.5)
    assert mask.shape == (80, 80)
    assert mask[40, 40]


@pytest.mark.parametrize(
    "result",
    [_Result(), _Result(np.zeros((0, 40, 40))), _Result(_two_people(), np.array([16, 16, 16]),
     np.array([0.9, 0.9, 0.9]))],
)
def test_no_person_is_reported_as_such(result: _Result) -> None:
    with pytest.raises(ValueError, match="No person found"):
        vision._largest_person_mask(result, (40, 40), 0.5)


def test_the_most_confident_pose_is_taken() -> None:
    keypoints = np.zeros((2, 17, 3))
    keypoints[1, :, 0] = 7.0
    result = _Result(keypoints=keypoints, confidences=np.array([0.4, 0.95]))
    assert vision._best_keypoints(result)[0, 0] == 7.0


def test_a_pose_result_with_nobody_is_reported() -> None:
    with pytest.raises(ValueError, match="No person found"):
        vision._best_keypoints(_Result())
