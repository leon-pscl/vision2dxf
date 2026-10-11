"""High level orchestration: YAML in, JSON + PDF out.

Kept in the package (rather than in the CLI) so it can be imported and tested
without invoking the command line.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .drafting import draft_pattern, extract_pattern_data
from .export import export_json, export_pdf
from .measurements import (
    MeasurementError,
    load_measurement_file,
    validate_measurement_file,
)

__all__ = ["ProcessedPattern", "iter_yaml_files", "process_file", "summary"]

DEFAULT_FORMATS = ("json", "pdf")


@dataclass
class ProcessedPattern:
    """Result of drafting and exporting a single measurement file."""

    config: Dict[str, Any]
    data: Dict[str, Any]
    json_path: Optional[Path] = None
    pdf_path: Optional[Path] = None
    written: List[Path] = field(default_factory=list)


def iter_yaml_files(targets: Sequence[str]) -> List[Path]:
    """Expand files/directories into a de-duplicated list of YAML paths."""
    files: List[Path] = []
    for target in targets:
        path = Path(target)
        if path.is_dir():
            files.extend(sorted(path.glob("*.yaml")))
            files.extend(sorted(path.glob("*.yml")))
        elif path.exists():
            files.append(path)
        else:
            raise MeasurementError("no such file or directory: %s" % path)

    seen = set()
    unique: List[Path] = []
    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def process_file(
    yaml_path: str | Path,
    out_dir: str | Path,
    formats: Sequence[str] = DEFAULT_FORMATS,
) -> ProcessedPattern:
    """Load, validate, draft and export one measurement file.

    Raises :class:`~pattern_drafter.measurements.MeasurementError` for invalid
    input and :class:`~pattern_drafter.drafting.OpenPatternMissing` when the
    drafting library is unavailable.
    """
    formats = [f.strip().lower() for f in formats]
    out_dir = Path(out_dir)

    config = load_measurement_file(yaml_path)
    validate_measurement_file(config)
    pattern = draft_pattern(config)
    data = extract_pattern_data(pattern, config)

    result = ProcessedPattern(config=config, data=data)

    if "json" in formats:
        result.json_path = Path(
            export_json(
                data, out_dir / ("%s.json" % config["profile_name"])
            )
        )
        result.written.append(result.json_path)

    if "pdf" in formats:
        info = {
            "Pattern": "%s block" % config["template"],
            "Profile": config["profile_name"],
        }
        result.pdf_path = Path(
            export_pdf(
                pattern,
                data,
                out_dir / ("%s.pdf" % config["profile_name"]),
                info,
            )
        )
        result.written.append(result.pdf_path)

    return result


def summary(result: ProcessedPattern) -> str:
    """One-line human readable summary of a processed pattern."""
    data = result.data
    return "%s (template=%s, style=%s, panels=%d)" % (
        result.config["profile_name"],
        result.config["template"],
        result.config["style"],
        len(data["panels"]),
    )
