"""Vision body measurements -> draftable OpenPattern measurements.

The vision pipeline writes a flat ``body:`` mapping with English keys and only
covers a handful of girths and depths (bust, waist, hips, arm length, ...).
OpenPattern instead reads many native (French) keys, so:

* the measurements that mean the same thing in both vocabularies are copied
  over (:data:`BODY_TO_OPENPATTERN`),
* everything else is taken from the OpenPattern reference profile that is both
  complete for the requested template and closest to the subject's girths,
  rescaled to those girths,
* the body file wins for every key it provides.

:func:`build_config` reports which keys came from the body file and which are
estimates, so the caller can show the user what was measured and what was not.
"""

from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import yaml

from .drafting import OpenPatternMissing
from .measurements import (
    MeasurementError,
    TEMPLATES,
    missing_measurements,
    validate_measurement_file,
)

__all__ = [
    "BODY_TO_OPENPATTERN",
    "build_config",
    "parse_body_measurements",
    "reference_profiles",
]


#: Vision key -> OpenPattern key, for the values that mean the same thing.
#: Vision keys with no OpenPattern counterpart (leg_circ, neck_w, the *_back_width
#: and *_line families, the *_incl / *_angle poses) are deliberately left to the
#: reference profile rather than guessed at.
BODY_TO_OPENPATTERN: Dict[str, str] = {
    "bust": "tour_poitrine",
    "waist": "tour_taille",
    "hips": "tour_bassin",
    "arm_length": "longueur_manche",
    "armscye_depth": "profondeur_emmanchure",
    "wrist": "tour_poignet",
    "height": "stature",
}

#: Girths used to pick the reference profile and to scale it.
ANCHORS = ("tour_poitrine", "tour_taille", "tour_bassin")

#: Sections a measurement file may hide its values under, in priority order.
_SECTIONS = ("body", "measurements")

_PROFILES: Optional[Dict[str, Dict[str, float]]] = None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_body_measurements(source: Any) -> Dict[str, float]:
    """Read vision measurements from YAML text or an already parsed mapping.

    Accepts ``{"body": {...}}`` (vision output), ``{"measurements": {...}}``
    (pattern_drafter input file) or a flat ``{name: value}`` mapping.  Values
    must be strictly positive numbers.
    """
    raw = yaml.safe_load(source) if isinstance(source, (str, bytes)) else source

    if isinstance(raw, Mapping):
        for section in _SECTIONS:
            nested = raw.get(section)
            if isinstance(nested, Mapping):
                raw = nested
                break

    if not isinstance(raw, Mapping):
        raise MeasurementError("expected a mapping of measurement name: value")

    out: Dict[str, float] = {}
    for key, value in raw.items():
        name = str(key).strip().lower().replace(" ", "_")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise MeasurementError(
                "measurement %r has a non numeric value %r" % (key, value)
            )
        if value <= 0:
            raise MeasurementError(
                "measurement %r must be positive, got %r" % (key, value)
            )
        out[name] = float(value)
    return out


def _translate(body: Mapping[str, float]) -> Dict[str, float]:
    """Add the OpenPattern spelling of every known vision key."""
    out = dict(body)
    for key, value in body.items():
        op_key = BODY_TO_OPENPATTERN.get(key)
        if op_key:
            out[op_key] = value
    return out


# ---------------------------------------------------------------------------
# Reference profiles bundled with OpenPattern
# ---------------------------------------------------------------------------


def _openpattern_db() -> Path:
    spec = importlib.util.find_spec("OpenPattern")
    if spec is None or not spec.submodule_search_locations:
        raise OpenPatternMissing(
            "OpenPattern is required to resolve reference measurements. "
            "Install it with:\n"
            "    pip install git+https://github.com/fmetivier/OpenPattern.git"
        )
    db = Path(next(iter(spec.submodule_search_locations))) / "data" / "measurements.db"
    if not db.is_file():
        raise OpenPatternMissing("OpenPattern measurement database missing: %s" % db)
    return db


def reference_profiles() -> Dict[str, Dict[str, float]]:
    """Every measurement profile shipped with OpenPattern, keyed by name.

    Placeholder zeros are dropped: a bundled profile carrying one (e.g.
    ``M44D.longueur_epaule``) would draft a degenerate pattern.
    """
    global _PROFILES
    if _PROFILES is None:
        conn = sqlite3.connect(str(_openpattern_db()))
        try:
            names = [
                row[0]
                for row in conn.execute("select distinct wkey from measurements")
            ]
            _PROFILES = {
                name: {
                    str(mtype): float(mval)
                    for mtype, mval in conn.execute(
                        "select mtype, mval from measurements where wkey = ?",
                        (name,),
                    )
                    if float(mval) > 0
                }
                for name in names
            }
        finally:
            conn.close()
    return _PROFILES


def _select_reference(
    measured: Mapping[str, float],
    template: str,
    style: str,
    gender: str,
) -> Tuple[str, Dict[str, float]]:
    """Nearest reference profile holding every measurement the draft needs.

    Darts are ignored here: they add optional requirements that some bundled
    profiles never recorded, and :func:`build_config` validates the merged
    result afterwards rather than hiding them here.
    """
    complete = [
        (name, profile)
        for name, profile in sorted(reference_profiles().items())
        if not missing_measurements(template, style, gender, profile)
    ]
    if not complete:
        raise MeasurementError(
            "no OpenPattern reference profile covers %s / %s / %s"
            % (template, style, gender)
        )

    def distance(item: Tuple[str, Mapping[str, float]]) -> float:
        _, profile = item
        return sum(
            (measured[key] - profile[key]) ** 2
            for key in ANCHORS
            if key in measured and key in profile
        )

    return min(complete, key=distance)


# ---------------------------------------------------------------------------
# Config assembly
# ---------------------------------------------------------------------------


def build_config(
    body: Any,
    template: str,
    style: str,
    gender: str,
    ease: float = 8.0,
    darts: bool = False,
    profile_name: str = "body",
) -> Dict[str, Any]:
    """Build a pattern_drafter config from vision body measurements.

    The returned mapping is a regular config (usable with
    :func:`~pattern_drafter.drafting.draft_pattern`) plus three extra keys:
    ``reference`` (the OpenPattern profile used), ``scale`` (the factor applied
    to it) and ``from_body`` (the keys the body file supplied).  The config is
    validated before it is returned, so anything it cannot fill in from the
    body file *and* the reference block is reported instead of drafted.
    """
    template = str(template).strip().lower()
    style = str(style).strip()
    gender = str(gender).strip().lower()
    if template not in TEMPLATES:
        raise MeasurementError(
            "unknown garment %r; expected one of %s"
            % (template, ", ".join(sorted(TEMPLATES)))
        )
    spec = TEMPLATES[template]
    if (style, gender) not in spec["requirements"]:
        raise MeasurementError(
            "%s cannot be drafted as %s / %s; available: %s"
            % (
                template,
                style,
                gender,
                ", ".join(
                    "%s %s" % pair for pair in sorted(spec["requirements"])
                ),
            )
        )

    measured = _translate(parse_body_measurements(body))
    if not measured:
        raise MeasurementError("no measurements found in the input")
    if not any(key in measured for key in ANCHORS):
        raise MeasurementError(
            "the body file must contain at least one girth "
            "(bust, waist or hips) to size the block"
        )

    name, profile = _select_reference(measured, template, style, gender)

    anchors: Sequence[str] = [k for k in ANCHORS if k in measured and k in profile]
    scale = (
        sum(measured[k] / profile[k] for k in anchors) / len(anchors)
        if anchors
        else 1.0
    )

    scaled = {key: round(value * scale, 3) for key, value in profile.items()}
    # A measured value always wins, even for a key the reference block lacks.
    overrides = {
        key: value
        for key, value in measured.items()
        if key in scaled or key in BODY_TO_OPENPATTERN.values()
    }
    scaled.update(overrides)

    config = {
        "template": template,
        "style": style,
        "gender": gender,
        "profile_name": profile_name,
        "ease": float(ease),
        "darts": bool(darts),
        "measurements": scaled,
        "source": "body_measurements",
        "reference": name,
        "scale": round(scale, 4),
        "from_body": sorted(overrides),
    }
    validate_measurement_file(config)
    return config