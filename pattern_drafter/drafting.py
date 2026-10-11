"""OpenPattern integration: instantiate a pattern and extract its geometry."""

from __future__ import annotations

import datetime as _dt
import importlib.util as _importlib
import numbers
import os
import sys
import tempfile
import types as _types
from pathlib import Path
from typing import Any, Dict, List, Mapping

from .measurements import TEMPLATES, write_measurement_db

__all__ = ["draft_pattern", "extract_pattern_data", "OpenPatternMissing"]


class OpenPatternMissing(RuntimeError):
    """Raised when the OpenPattern library cannot be imported."""


#: Attributes holding ordered point lists (polygon outlines) per group.
_PANEL_ATTRS = (
    ("front", "Front_vertices"),
    ("back", "Back_vertices"),
    ("sleeve", "Sleeve_vertices"),
    ("bodice", "Bodice_vertices"),
)

#: Dictionaries holding named construction points / curves per group.
_POINT_ATTRS = (
    ("front", "Front_dic"),
    ("back", "Back_dic"),
    ("sleeve", "Sleeve_points_dic"),
    ("bodice", "Bodice_points_dic"),
    ("points", "points_dic"),
)


def _install_pkg_resources_shim() -> None:
    """Provide a minimal ``pkg_resources`` when setuptools >= 82 removed it.

    OpenPattern imports ``pkg_resources`` at module load and only ever calls
    ``resource_filename`` to resolve its bundled measurements database, a path
    we always override through the ``dbPATH`` keyword argument.
    """
    try:
        import pkg_resources  # noqa: F401
    except ModuleNotFoundError:
        pass
    else:
        return

    if "pkg_resources" in sys.modules:
        return

    def resource_filename(
        package_or_requirement: str, resource_name: str
    ) -> str:
        spec = _importlib.find_spec(package_or_requirement)
        if spec is None:
            raise ModuleNotFoundError(package_or_requirement)
        locations = spec.submodule_search_locations
        if locations:
            base = next(iter(locations))
        elif spec.origin:
            base = os.path.dirname(spec.origin)
        else:
            raise FileNotFoundError(package_or_requirement)
        return os.path.join(base, resource_name)

    shim = _types.ModuleType("pkg_resources")
    shim.resource_filename = resource_filename
    sys.modules["pkg_resources"] = shim


def _import_openpattern():
    _install_pkg_resources_shim()
    try:
        import OpenPattern  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on the environment
        raise OpenPatternMissing(
            "OpenPattern is required to draft patterns. Install it with:\n"
            "    pip install git+https://github.com/fmetivier/OpenPattern.git\n\n"
            "Underlying error: %s: %s" % (type(exc).__name__, exc)
        ) from exc
    return OpenPattern


def _is_number(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _is_pair(value: Any) -> bool:
    try:
        return len(value) == 2 and _is_number(value[0]) and _is_number(value[1])
    except (TypeError, KeyError):
        return False


def _round_pair(x: Any, y: Any) -> List[float]:
    return [round(float(x), 4), round(float(y), 4)]


def _point_to_pair(point: Any) -> List[float]:
    return _round_pair(point.x, point.y)


def _normalise_panels(raw: Any) -> List[List[List[float]]]:
    """Return a list of panels (each a list of ``[x, y]`` vertices)."""
    if raw is None:
        return []
    try:
        first = raw[0]
    except (TypeError, IndexError, KeyError):
        return []

    if _is_pair(first):
        return [[_round_pair(p[0], p[1]) for p in raw]]

    panels: List[List[List[float]]] = []
    for sub in raw:
        try:
            if sub and _is_pair(sub[0]):
                panels.append([_round_pair(p[0], p[1]) for p in sub])
        except (TypeError, IndexError, KeyError):
            continue
    return panels


def _split_point_dict(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Split a pattern dictionary into named points and named curves."""
    points: Dict[str, List[float]] = {}
    curves: Dict[str, List[List[float]]] = {}
    if not isinstance(raw, Mapping):
        return {"points": points, "curves": curves}

    for name, value in raw.items():
        if hasattr(value, "x") and hasattr(value, "y"):
            points[str(name)] = _point_to_pair(value)
        else:
            try:
                if value and _is_pair(value[0]):
                    curves[str(name)] = [
                        _round_pair(p[0], p[1]) for p in value
                    ]
            except (TypeError, IndexError, KeyError):
                continue
    return {"points": points, "curves": curves}


def _compute_bounds(
    panels: List[Dict[str, Any]], points: Mapping[str, Any]
) -> Dict[str, float]:
    xs: List[float] = []
    ys: List[float] = []

    for panel in panels:
        for x, y in panel["vertices"]:
            xs.append(x)
            ys.append(y)

    for group in points.values():
        for x, y in group.get("points", {}).values():
            xs.append(x)
            ys.append(y)

    if not xs:
        return {"xmin": 0.0, "xmax": 0.0, "ymin": 0.0, "ymax": 0.0}

    return {
        "xmin": round(min(xs), 4),
        "xmax": round(max(xs), 4),
        "ymin": round(min(ys), 4),
        "ymax": round(max(ys), 4),
    }


def draft_pattern(config: Mapping[str, Any]):
    """Instantiate the OpenPattern class described by ``config``.

    A throw-away sqlite3 database is created from the YAML measurements and
    handed to OpenPattern through its own ``dbPATH`` mechanism, so the library
    reads the bespoke values exactly as it would read its bundled sizes.
    """
    OP = _import_openpattern()
    template = config["template"]
    spec = TEMPLATES[template]

    cls = getattr(OP, spec["op_class"], None)
    if cls is None:  # pragma: no cover - depends on library version
        raise OpenPatternMissing(
            "OpenPattern does not expose the class %r required for the %r "
            "template." % (spec["op_class"], template)
        )

    with tempfile.TemporaryDirectory(prefix="pattern_drafter_") as tmp:
        db_path = os.path.join(tmp, "measurements.db")
        write_measurement_db(
            config["measurements"], config["profile_name"], db_path
        )

        kwargs: Dict[str, Any] = {
            "pname": config["profile_name"],
            "gender": config["gender"],
            "style": config["style"],
            "dbPATH": db_path,
            "figPATH": tmp,
            "frmt": "pdf",
        }
        if template == "bodice":
            kwargs["ease"] = float(config.get("ease", 8))
            kwargs["Back_Front_space"] = float(
                config.get("back_front_space", 4)
            )
        elif template == "skirt":
            kwargs["ease"] = float(config.get("ease", 8))
            kwargs["curves"] = bool(config.get("curves", False))
        elif template == "trousers":
            kwargs["darts"] = bool(config.get("darts", False))

        pattern = cls(**kwargs)

    if template == "bodice" and config.get("darts"):
        pattern.add_bust_dart()
        pattern.add_waist_dart()

    return pattern


def extract_pattern_data(
    pattern: Any, config: Mapping[str, Any]
) -> Dict[str, Any]:
    """Turn a drafted OpenPattern instance into a JSON-serialisable dict."""
    panels: List[Dict[str, Any]] = []
    for name, attr in _PANEL_ATTRS:
        raw = getattr(pattern, attr, None)
        sub_panels = _normalise_panels(raw)
        if len(sub_panels) == 1:
            panels.append(
                {"name": name, "closed": True, "vertices": sub_panels[0]}
            )
        else:
            for index, vertices in enumerate(sub_panels):
                panels.append(
                    {
                        "name": "%s_%d" % (name, index),
                        "closed": True,
                        "vertices": vertices,
                    }
                )

    point_groups: Dict[str, Any] = {}
    for name, attr in _POINT_ATTRS:
        raw = getattr(pattern, attr, None)
        if isinstance(raw, Mapping) and raw:
            point_groups[name] = _split_point_dict(raw)

    bounds = _compute_bounds(panels, point_groups)

    resolved = getattr(pattern, "m", None)
    if isinstance(resolved, Mapping):
        resolved = {str(k): round(float(v), 4) for k, v in resolved.items()}

    op_version = ""
    try:
        import OpenPattern  # type: ignore

        op_version = str(getattr(OpenPattern, "__version__", ""))
    except Exception:  # pragma: no cover
        pass

    return {
        "schema_version": "1.0",
        "template": config["template"],
        "style": config["style"],
        "gender": config["gender"],
        "units": "cm",
        "profile_name": config["profile_name"],
        "input_measurements": {
            str(k): round(float(v), 4)
            for k, v in config["measurements"].items()
        },
        "resolved_measurements": resolved,
        "panels": panels,
        "points": point_groups,
        "bounds": bounds,
        "metadata": {
            "generator": "pattern_drafter",
            "openpattern_class": TEMPLATES[config["template"]]["op_class"],
            "openpattern_version": op_version,
            "source_yaml": config.get("source", ""),
            "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(
                timespec="seconds"
            ),
        },
    }
