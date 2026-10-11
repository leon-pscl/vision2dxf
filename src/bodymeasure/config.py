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
    """Webcam index and the rotation applied to every raw frame."""

    index: int
    rotation: int


class ModelsConfig(_Strict):
    """Paths to the YOLO26 weights and the inference image size."""

    seg: str
    pose: str
    imgsz: int
    retina_masks: bool


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


class AppConfig(_Strict):
    """Contents of ``config/app.yaml``."""

    camera: CameraConfig
    models: ModelsConfig
    thresholds: dict[str, float]
    defaults: dict[str, float]
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
