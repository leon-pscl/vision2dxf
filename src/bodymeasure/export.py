"""The three export files and the gate in front of the pattern-ready one (brief 15).

``body_measurements.yaml`` is the only file a pattern engine reads, so it holds
the primary GarmentCode keys and nothing else: an extra key would make it fail to
load, and a plausible-looking stand-in for a missing one would make it load and
lie. Everything else the pipeline measures goes to
``extended_measurements.yaml``, and the whole provenance trail goes to
``measurement_report.yaml``, which may be written at any time, including for a
session that failed.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path

import yaml

from bodymeasure import __version__, store
from bodymeasure.config import Calibration, load_app_config, load_schema
from bodymeasure.measure import MeasurementSet
from bodymeasure.store import write_atomic

BODY_FILE = "body_measurements.yaml"
EXTENDED_FILE = "extended_measurements.yaml"
REPORT_FILE = "measurement_report.yaml"
UNVALIDATED = "unvalidated_against_manual_anthropometry"
NO_DISTORTION_WARNING = "lens distortion not corrected"


@dataclass(frozen=True)
class Gate:
    """Whether the pattern-ready export may be written, and what blocks it."""

    ok: bool
    profile: str
    blocked: dict[str, str]

    @property
    def reasons(self) -> list[str]:
        """Return one human-readable line per blocked field."""
        return [f"{name}: {why}" for name, why in sorted(self.blocked.items())]


def export_gate(measurements: MeasurementSet, profile: str) -> Gate:
    """Return whether every field the garment profile needs is usable (brief 15.3).

    Raises:
        KeyError: if the profile is not configured.

    """
    required = load_app_config().profiles[profile]
    blocked = measurements.missing(list(required))
    return Gate(ok=not blocked, profile=profile, blocked=blocked)


def body_document(measurements: MeasurementSet, session_id: str) -> dict[str, object]:
    """Return the ``body_measurements.yaml`` document for the pinned revision."""
    primary = load_schema()
    pinned = load_app_config().garmentcode
    return {
        "metadata": {
            "session_id": session_id,
            "garmentcode_commit": pinned.commit,
            "unit": "cm",
            "validation_status": UNVALIDATED,
        },
        "body": {
            name: value
            for name, value in measurements.body().items()
            if name in primary
        },
    }


def extended_document(measurements: MeasurementSet) -> dict[str, object]:
    """Return everything measured that is not a primary GarmentCode key (brief 3.3)."""
    primary = load_schema()
    return {
        "metadata": {"unit": "cm", "validation_status": UNVALIDATED},
        "measurements": {
            name: {
                "value": entry.value_cm,
                "unit": "cm" if entry.unit == "mm" else "deg",
                "status": entry.status,
                "source": entry.source,
                "method": entry.method,
            }
            for name, entry in sorted(measurements.values.items())
            if name not in primary
        },
    }


def report_document(
    session_id: str,
    measurements: MeasurementSet,
    gate: Gate,
    calibration: Calibration | None,
) -> dict[str, object]:
    """Return the full provenance report (brief 15.2).

    It names the models, the scale method, every measurement record, every
    landmark the operator moved and every artifact on disk, and it says plainly
    that nothing here has been checked against a manual measurement.
    """
    from bodymeasure import artifacts  # local: the report is the only caller

    config = load_app_config()
    session = store.get_session(session_id) or {}
    warnings = list(measurements.warnings)
    if calibration is None or not calibration.has_lens_model:
        warnings.append(NO_DISTORTION_WARNING)

    return {
        "metadata": {
            "session_id": session_id,
            "app_version": __version__,
            "garmentcode_commit": config.garmentcode.commit,
            "validated_against_manual": False,
            "validation_status": UNVALIDATED,
            "status": session.get("status"),
        },
        "models": {
            "segmentation": config.models.seg,
            "pose": config.models.pose,
            "imgsz": config.models.imgsz,
            "retina_masks": config.models.retina_masks,
            "thresholds": dict(config.thresholds),
        },
        "lens_calibration": {
            "applied": bool(calibration and calibration.has_lens_model),
            "rotation_deg": calibration.rotation if calibration else config.camera.rotation,
        },
        "scale": {
            "mode": "stature",
            "stature_cm": session.get("height_cm"),
            "note": (
                "A measured stature converts pixels to millimetres. It is a "
                "calibration input, not evidence that any measurement is accurate."
            ),
        },
        "enhancement": dict(config.enhance),
        "export_gate": {
            "profile": gate.profile,
            "ok": gate.ok,
            "blocked": gate.blocked,
        },
        "measurements": [
            entry.record(name) for name, entry in sorted(measurements.values.items())
        ],
        "edits": store.list_edits(session_id),
        "artifacts": [
            {
                "stage_id": entry["stage_id"],
                "view": entry["view"],
                "path": entry["path"],
                "algorithm": entry["algorithm"],
                "parameters": entry["parameters"],
                "source_files": entry["source_files"],
            }
            for entry in artifacts.manifest(session_id)
        ],
        "warnings": warnings,
    }


def measurements_csv(measurements: MeasurementSet) -> str:
    """Return one row per measurement, for a spreadsheet (brief 16.4)."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        ["name", "value", "unit", "status", "source", "method", "formula",
         "confidence", "views", "edited", "validated_against_manual", "warnings"]
    )
    for name, entry in sorted(measurements.values.items()):
        writer.writerow([
            name,
            "" if entry.value_cm is None else entry.value_cm,
            "cm" if entry.unit == "mm" else "deg",
            entry.status,
            entry.source,
            entry.method,
            entry.formula,
            entry.confidence,
            " ".join(entry.views),
            int(entry.edited),
            "false",
            "; ".join(entry.warnings),
        ])
    return buffer.getvalue()


def write_all(
    folder: Path,
    session_id: str,
    measurements: MeasurementSet,
    profile: str,
    calibration: Calibration | None,
) -> Gate:
    """Write the report and extended files always, the body file only if it may be.

    A blocked pattern-ready export leaves no stale ``body_measurements.yaml``
    behind: the file is removed, so a pattern engine cannot pick up the previous
    run's numbers.

    Raises:
        KeyError: if the profile is not configured.

    """
    gate = export_gate(measurements, profile)
    write_atomic(
        folder / REPORT_FILE,
        yaml.safe_dump(
            report_document(session_id, measurements, gate, calibration), sort_keys=False
        ),
    )
    write_atomic(
        folder / EXTENDED_FILE,
        yaml.safe_dump(extended_document(measurements), sort_keys=False),
    )
    if gate.ok:
        write_atomic(
            folder / BODY_FILE,
            yaml.safe_dump(body_document(measurements, session_id), sort_keys=False),
        )
    else:
        (folder / BODY_FILE).unlink(missing_ok=True)
    return gate
