"""Loading and validation of the YAML configuration files in ``config/``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CameraConfig(_Strict):
    """Webcam index, the rotation applied to every raw frame, and its border."""

    index: int
    rotation: int
    pad_frac: float = 0.08


class ModelsConfig(_Strict):
    """Paths to the YOLO26 weights, their hashes, and the inference image size.

    A hash is optional so a station can run fine-tuned weights that have not been
    published yet, but when one is configured it is enforced: the app refuses to
    measure with a weight file that is not the one the results were recorded
    against (brief 5).
    """

    seg: str
    pose: str
    imgsz: int
    retina_masks: bool
    seg_sha256: str | None = None
    pose_sha256: str | None = None


class PathsConfig(_Strict):
    """Data locations, relative to the repository root."""

    sessions: str


class StorageConfig(_Strict):
    """Retention policy and the consent text version (brief 10.5)."""

    data_dir: str
    retention_days_images: int
    purge_on_startup: bool
    keep_debug_after_confirm: bool
    unconfirmed_days: int
    consent_version: str


class UploadsConfig(_Strict):
    """What an upload may be, so a hostile file cannot reach the decoder (brief 17)."""

    max_bytes: int
    max_pixels: int
    allowed_types: list[str]


class GarmentCodeConfig(_Strict):
    """The pinned GarmentCode revision the export is written for (brief 3.2)."""

    commit: str
    repository: str


class AppConfig(_Strict):
    """Contents of ``config/app.yaml``."""

    camera: CameraConfig
    models: ModelsConfig
    thresholds: dict[str, float]
    defaults: dict[str, float]
    enhance: dict[str, float]
    uploads: UploadsConfig
    garmentcode: GarmentCodeConfig
    profiles: dict[str, list[str]]
    paths: PathsConfig
    storage: StorageConfig


class Calibration(_Strict):
    """Contents of the optional ``config/calibration.yaml``.

    The lens model only: there is no scale here, because the millimetres per
    pixel come from the subject's measured height on each view
    (``landmarks.ViewScale``). ``camera_matrix`` is optional because a station
    may run uncorrected; without it frames are measured as they arrive.
    """

    camera_matrix: list[list[float]] | None = None
    dist_coeffs: list[float] | None = None
    rotation: int = 0

    @property
    def has_lens_model(self) -> bool:
        """Return whether frames can be undistorted."""
        return self.camera_matrix is not None and self.dist_coeffs is not None


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@lru_cache(maxsize=1)
def load_app_config(path: Path | None = None) -> AppConfig:
    """Return the validated application configuration."""
    return AppConfig(**_read_yaml(path or CONFIG_DIR / "app.yaml"))


def required_fields(profile: str) -> list[str]:
    """Return the body keys the named garment profile must have before export.

    Raises:
        KeyError: if the profile is not configured.

    """
    return list(load_app_config().profiles[profile])


@lru_cache(maxsize=1)
def load_schema(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Return the measurement schema: key -> {unit, views, confidence}."""
    return _read_yaml(path or CONFIG_DIR / "schema.yaml")["measurements"]


def load_calibration(path: Path | None = None) -> Calibration | None:
    """Return the stored calibration, or ``None`` if the station is not calibrated yet."""
    path = path or CONFIG_DIR / "calibration.yaml"
    if not path.exists():
        return None
    return Calibration(**_read_yaml(path))
