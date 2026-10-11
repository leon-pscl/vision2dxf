"""Run the measurement engine on the GarmentCode reference bodies (brief M4b).

Each body ships with a YAML of its true measurements, so this reports the error
of the engine against a known answer. The thresholds in ``config/app.yaml`` are
**not** tuned to these five bodies: the brief forbids it, and five bodies cannot
justify a threshold anyway.

Usage::

    PYTHONPATH=src python bench/run_bench.py vendor/GarmentCode/assets/bodies
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from bench.render_silhouettes import render_body
from bodymeasure.config import load_schema
from bodymeasure.landmarks import as_dict, find_front_landmarks, find_side_landmarks, view_scale
from bodymeasure.measure import MeasurementSet, SideView, measure
from bodymeasure.vision import load_models

BODIES = (
    "mean_all",
    "mean_female",
    "mean_male",
    "f_smpl_average_A40",
    "m_smpl_average_A40",
)
OUT = Path(__file__).resolve().parent / "out"


@dataclass(frozen=True)
class BodyResult:
    """One body's measurements beside its published truth."""

    name: str
    measured: dict[str, float]
    truth: dict[str, float]
    export_path: Path


def keypoints_for(shaded: np.ndarray) -> np.ndarray:
    """Return COCO-17 keypoints from the pose model on a shaded rendering.

    Raises:
        ValueError: if the model finds nobody, which the bench reports rather
            than papering over with invented joints.

    """
    _, pose_model = load_models()
    result = pose_model.predict(shaded, verbose=False)[0]
    data = result.keypoints.data.cpu().numpy() if result.keypoints is not None else []
    if len(data) == 0:
        raise ValueError("the pose model found no person in the rendering")
    confidences = result.boxes.conf.cpu().numpy()
    return data[int(np.argmax(confidences))].astype(float)


def measure_body(path: Path, height_cm: float) -> MeasurementSet:
    """Render one mesh and run the whole engine on it.

    ``height_cm`` is the body's published height: the engine's scale, so the bench
    must not assume one height for five different bodies.

    Raises:
        ValueError: if rendering or landmarking cannot proceed.

    """
    views = render_body(path)
    front, side = views["front"], views["side"]

    front_kp = keypoints_for(front.shaded)
    side_kp = keypoints_for(side.shaded)

    front_marks = find_front_landmarks(front.mask, front_kp, height_cm)
    front_rows = as_dict(front_marks)
    front_scale = view_scale(front_marks, height_cm)
    front_heights = {
        name: front_scale.height_mm(front_rows[f"{name}_row"].y)
        for name in ("armpit", "waist", "crotch")
    }
    side_marks = find_side_landmarks(side.mask, side_kp, "side_left", front_heights, height_cm)
    side_view = SideView(
        name="side_left", mask=side.mask, landmarks=as_dict(side_marks), height_cm=height_cm
    )
    return measure(front.mask, front_marks, [side_view], front_kp, height_cm)


def load_truth(path: Path) -> dict[str, float]:
    """Read a body's published measurements, dropping the derived underscore keys."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    body = data.get("body", data)
    return {key: float(value) for key, value in body.items() if not key.startswith("_")}


def _body_parameters(clone: Path) -> type | None:
    """Return GarmentCode's BodyParameters class from the cloned repository.

    The PyPI ``pygarment`` wheel installs its modules under different top-level
    names and does not ship this subclass, so the clone is the source of truth.
    """
    params_file = clone / "assets" / "bodies" / "body_params.py"
    if not params_file.exists():
        return None
    if str(clone) not in sys.path:
        sys.path.insert(0, str(clone))
    spec = importlib.util.spec_from_file_location("garmentcode_body_params", params_file)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BodyParameters


def check_pygarment(export_path: Path, clone: Path) -> str:
    """Return whether GarmentCode's BodyParameters loads the export."""
    try:
        body_parameters = _body_parameters(clone)
    except Exception as error:  # noqa: BLE001 - a missing optional dependency is a result
        return f"unavailable: {type(error).__name__}: {error}"
    if body_parameters is None:
        return "unavailable: no GarmentCode clone"
    try:
        body = body_parameters(str(export_path))
    except Exception as error:  # noqa: BLE001 - the point is to report any failure
        return f"FAILED: {type(error).__name__}: {error}"
    derived = sorted(key for key in body.params if key.startswith("_"))
    return f"loaded, recomputed {len(derived)} derived keys"


def write_results(results: list[BodyResult], path: Path) -> Path:
    """Write ``bench/results.csv``: one row per body and measurement."""
    schema = load_schema()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["body", "key", "unit", "true", "measured", "error", "error_pct"])
        for result in results:
            for key in sorted(result.measured):
                measured = result.measured[key]
                truth = result.truth.get(key)
                if truth is None:
                    writer.writerow([result.name, key, schema[key]["unit"], "", measured, "", ""])
                    continue
                error = measured - truth
                writer.writerow(
                    [
                        result.name,
                        key,
                        schema[key]["unit"],
                        f"{truth:.1f}",
                        f"{measured:.1f}",
                        f"{error:.1f}",
                        f"{100 * error / truth:.1f}" if truth else "",
                    ]
                )
    return path


def summarise(results: list[BodyResult]) -> None:
    """Print the error spread per measurement, worst first."""
    per_key: dict[str, list[float]] = {}
    for result in results:
        for key, measured in result.measured.items():
            truth = result.truth.get(key)
            if truth:
                per_key.setdefault(key, []).append(abs(measured - truth))

    print(f"\n{'key':24s} {'n':>2s} {'mean |err|':>10s} {'max |err|':>9s}")
    for key, errors in sorted(per_key.items(), key=lambda item: -float(np.mean(item[1]))):
        print(f"{key:24s} {len(errors):2d} {np.mean(errors):10.1f} {max(errors):9.1f}")


def main(argv: list[str]) -> int:
    """Measure every reference body and write the results."""
    if not argv:
        print(__doc__)
        return 1
    bodies_dir = Path(argv[0])
    OUT.mkdir(parents=True, exist_ok=True)

    results: list[BodyResult] = []
    for name in BODIES:
        mesh_path = bodies_dir / f"{name}.obj"
        truth_path = bodies_dir / f"{name}.yaml"
        if not mesh_path.exists() or not truth_path.exists():
            print(f"{name}: SKIPPED, no {mesh_path.name} or {truth_path.name}")
            continue
        try:
            truth = load_truth(truth_path)
            measurements = measure_body(mesh_path, truth["height"])
        except Exception as error:  # noqa: BLE001 - a bench reports failures, it does not raise
            print(f"{name}: FAILED, {type(error).__name__}: {error}")
            continue

        export_path = OUT / f"{name}_body_measurements.yaml"
        export_path.write_text(
            yaml.safe_dump({"body": measurements.body()}, sort_keys=True), encoding="utf-8"
        )
        results.append(BodyResult(name, measurements.body(), truth, export_path))
        status = check_pygarment(export_path, bodies_dir.parent.parent)
        print(f"{name}: measured, pygarment {status}")

    if not results:
        print("no bodies measured")
        return 1

    path = write_results(results, Path(__file__).resolve().parent / "results.csv")
    summarise(results)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
