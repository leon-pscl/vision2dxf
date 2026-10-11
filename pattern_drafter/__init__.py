"""pattern_drafter — draft sewing patterns with OpenPattern from YAML measurements.

The package turns a simple YAML measurement file into:

* a JSON description of the drafted pattern (panels, vertices, points, bounds),
* a 1:1 PDF rendering of the pattern for visual inspection / printing.

It relies on the third party ``OpenPattern`` library for the actual drafting
math unless only measurement parsing / validation is used.
"""

from .measurements import (
    MeasurementError,
    TEMPLATES,
    load_measurement_file,
    validate_measurement_file,
)
from .drafting import draft_pattern, extract_pattern_data
from .export import export_json, export_pdf
from .pipeline import ProcessedPattern, iter_yaml_files, process_file
from .vision import build_config, parse_body_measurements

__all__ = [
    "MeasurementError",
    "TEMPLATES",
    "load_measurement_file",
    "validate_measurement_file",
    "draft_pattern",
    "extract_pattern_data",
    "export_json",
    "export_pdf",
    "ProcessedPattern",
    "iter_yaml_files",
    "process_file",
    "build_config",
    "parse_body_measurements",
]

__version__ = "1.0.0"
