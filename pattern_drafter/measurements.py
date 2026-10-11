"""YAML measurement handling for :mod:`pattern_drafter`.

This module is intentionally free of any heavy dependency (no OpenPattern,
numpy, matplotlib or scipy) so that measurement files can be parsed and
validated quickly and on machines where OpenPattern is not installed yet.

OpenPattern stores its anthropometric data in a small sqlite3 database using
French measurement keys (``tour_poitrine`` = bust circumference, ...).  The
functions here prepare an equivalent throw-away database from the values found
in a YAML file, so the drafting classes can be instantiated with bespoke
measurements.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import yaml

__all__ = [
    "MeasurementError",
    "TEMPLATES",
    "MEASUREMENT_GLOSSARY",
    "load_measurement_file",
    "validate_measurement_file",
    "missing_measurements",
    "write_measurement_db",
]


class MeasurementError(ValueError):
    """Raised when a measurement file is incomplete or malformed."""


# ---------------------------------------------------------------------------
# Template catalogue
# ---------------------------------------------------------------------------
# Each template maps to an OpenPattern class and knows which measurement keys
# the underlying drafting method reads from ``pattern.m``.  ``requirements``
# is keyed by ``(style, gender)``; when an exact key is missing the union of
# all requirements declared for that style (any gender) is used, and finally
# the union of every requirement for the template.
#
# ``one_of`` lists groups of alternative keys from which at least one must be
# present (e.g. total length can be given in several ways).

TEMPLATES: Dict[str, Dict[str, Any]] = {
    "bodice": {
        "op_class": "Basic_Bodice",
        "description": "Basic bodice block (front + back + sleeve).",
        "default_style": "Gilewska",
        "default_gender": "w",
        "supports_darts": True,
        "requirements": {
            ("Gilewska", "w"): [
                "carrure_devant",
                "carrure_dos",
                "hauteur_carrure",
                "hauteur_emmanchure",
                "largeur_encolure",
                "longueur_col_devant",
                "longueur_col_dos",
                "longueur_devant",
                "longueur_dos",
                "longueur_emmanchure_devant",
                "longueur_emmanchure_dos",
                "longueur_epaule",
                "profondeur_emmanchure",
                "profondeur_encolure_devant",
                "profondeur_encolure_dos",
                "tour_encolure",
                "tour_poitrine",
                "hauteur_coude",
                "longueur_manche",
                "tour_poignet",
            ],
            ("Gilewska", "m"): [
                "carrure_devant",
                "carrure_dos",
                "hauteur_bassin",
                "largeur_epaule",
                "longueur_col_devant",
                "longueur_col_dos",
                "longueur_devant",
                "longueur_dos",
                "longueur_emmanchure_devant",
                "longueur_emmanchure_dos",
                "longueur_epaule",
                "profondeur_emmanchure",
                "tour_bassin",
                "tour_encolure",
                "tour_poitrine",
                "longueur_manche",
                "tour_poignet",
            ],
            ("Donnanno", "w"): [
                "carrure_dos",
                "longueur_devant",
                "longueur_dos",
                "longueur_emmanchure_devant",
                "longueur_emmanchure_dos",
                "longueur_epaule",
                "tour_cou",
                "tour_poitrine",
                "longueur_manche",
            ],
            ("Donnanno", "m"): [
                "hauteur_bassin",
                "largeur_epaule",
                "longueur_devant",
                "longueur_dos",
                "longueur_emmanchure_devant",
                "longueur_emmanchure_dos",
                "longueur_epaule",
                "tour_poitrine",
            ],
            ("Chiappetta", "w"): [
                "carrure_devant",
                "carrure_dos",
                "longueur_col_devant",
                "longueur_col_dos",
                "longueur_dos",
                "longueur_emmanchure_devant",
                "longueur_emmanchure_dos",
                "longueur_epaule",
                "profondeur_emmanchure",
                "tour_encolure",
                "tour_poitrine",
                "longueur_manche",
                "tour_poignet",
            ],
            ("Chiappetta", "m"): [
                "carrure_devant",
                "carrure_dos",
                "longueur_col_devant",
                "longueur_col_dos",
                "longueur_dos",
                "longueur_emmanchure_devant",
                "longueur_emmanchure_dos",
                "longueur_epaule",
                "profondeur_emmanchure",
                "tour_encolure",
                "tour_poitrine",
                "longueur_manche",
                "tour_poignet",
            ],
        },
        # Extra keys needed when the optional bust/waist darts are requested.
        "dart_requirements": [
            "ecart_poitrine",
            "hauteur_poitrine",
            "tour_poitrine",
            "tour_taille",
            "carrure_dos",
        ],
        "one_of": [],
    },
    "skirt": {
        "op_class": "Basic_Skirt",
        "description": "Basic (pencil) skirt block.",
        "default_style": "Gilewska",
        "default_gender": "w",
        "supports_darts": False,
        "requirements": {
            ("Gilewska", "w"): [
                "ecart_poitrine",
                "hauteur_bassin",
                "hauteur_taille_genou",
                "tour_bassin",
                "tour_taille",
            ],
            ("Gilewska", "m"): [
                "hauteur_bassin",
                "hauteur_taille_genou",
                "tour_bassin",
                "tour_taille",
            ],
            ("Donnanno", "w"): [
                "ecart_poitrine",
                "hauteur_bassin",
                "hauteur_taille_genou",
                "tour_bassin",
                "tour_taille",
            ],
            ("Donnanno", "m"): [
                "hauteur_bassin",
                "hauteur_taille_genou",
                "tour_bassin",
                "tour_taille",
            ],
            ("Chiappetta", "w"): [
                "hauteur_bassin",
                "hauteur_taille_genou",
                "tour_bassin",
                "tour_taille",
            ],
            ("Chiappetta", "m"): [
                "hauteur_bassin",
                "hauteur_taille_genou",
                "tour_bassin",
                "tour_taille",
            ],
        },
        "dart_requirements": [],
        "one_of": [],
    },
    "trousers": {
        "op_class": "Basic_Trousers",
        "description": "Basic trousers block (only Donnanno is implemented upstream).",
        "default_style": "Donnanno",
        "default_gender": "m",
        "supports_darts": True,
        "requirements": {
            ("Donnanno", "m"): [
                "hauteur_bassin",
                "hauteur_taille_genou",
                "montant",
                "tour_bassin",
                "tour_taille",
            ],
            ("Donnanno", "w"): [
                "hauteur_bassin",
                "hauteur_taille_genou",
                "montant",
                "tour_bassin",
                "tour_taille",
            ],
        },
        "dart_requirements": [],
        "one_of": [
            [
                "longueur_tot",
                "hauteur_taille_terre",
                "longueur_taille_terre",
                "entrejambe_terre",
            ],
        ],
    },
}


#: English glosses for the native French keys (documentation only).
MEASUREMENT_GLOSSARY: Dict[str, str] = {
    "tour_poitrine": "bust circumference",
    "tour_taille": "waist circumference",
    "tour_bassin": "hip circumference",
    "tour_petites_hanches": "high-hip circumference",
    "tour_encolure": "neck circumference",
    "tour_cou": "neck circumference (Donnanno)",
    "tour_bras": "upper-arm circumference",
    "tour_poignet": "wrist circumference",
    "tour_cuisse": "thigh circumference",
    "tour_genou": "knee circumference",
    "tour_cheville": "ankle circumference",
    "carrure_dos": "back shoulder width",
    "carrure_devant": "front shoulder width",
    "largeur_epaule": "shoulder width",
    "largeur_encolure": "neck width",
    "ecart_poitrine": "bust point separation",
    "hauteur_poitrine": "bust height",
    "hauteur_carrure": "shoulder slope drop",
    "hauteur_emmanchure": "armhole height",
    "hauteur_bassin": "hip depth",
    "hauteur_taille_genou": "waist to knee length",
    "hauteur_taille_terre": "waist to floor length",
    "longueur_taille_terre": "waist to floor length",
    "longueur_tot": "total (outseam) length",
    "longueur_dos": "back length",
    "longueur_devant": "front length",
    "longueur_epaule": "shoulder length",
    "longueur_manche": "sleeve length",
    "longueur_col_dos": "back neck length",
    "longueur_col_devant": "front neck length",
    "longueur_emmanchure_dos": "back armhole length",
    "longueur_emmanchure_devant": "front armhole length",
    "profondeur_emmanchure": "armhole depth",
    "profondeur_encolure_dos": "back neck depth",
    "profondeur_encolure_devant": "front neck depth",
    "hauteur_coude": "elbow height",
    "montant": "rise / crotch depth",
    "entrejambe_terre": "inseam length",
    "fourche": "crotch length",
}


# ---------------------------------------------------------------------------
# Requirement resolution / validation
# ---------------------------------------------------------------------------


def _resolve_requirements(
    template: str, style: str, gender: str
) -> Tuple[List[str], List[Sequence[str]]]:
    """Return (required_keys, one_of_groups) for a template/style/gender."""
    if template not in TEMPLATES:
        raise MeasurementError(
            "unknown template %r; expected one of %s"
            % (template, ", ".join(sorted(TEMPLATES)))
        )

    spec = TEMPLATES[template]
    reqs: Dict[Tuple[str, str], List[str]] = spec["requirements"]

    if (style, gender) in reqs:
        required = list(reqs[(style, gender)])
    else:
        same_style = [v for (s, _), v in reqs.items() if s == style]
        if same_style:
            required = sorted({k for group in same_style for k in group})
        else:
            required = sorted(
                {k for group in reqs.values() for k in group}
            )

    one_of = list(spec.get("one_of", []))
    return required, one_of


def missing_measurements(
    template: str,
    style: str,
    gender: str,
    provided: Mapping[str, Any],
    darts: bool = False,
) -> List[str]:
    """Return the list of measurements required but not provided."""
    required, one_of = _resolve_requirements(template, style, gender)

    if darts and TEMPLATES.get(template, {}).get("supports_darts"):
        required = sorted(set(required) | set(TEMPLATES[template]["dart_requirements"]))

    missing = [key for key in required if key not in provided]

    for group in one_of:
        if not any(key in provided for key in group):
            missing.append("one of {%s}" % ", ".join(group))

    return missing


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

_REQUIRED_TOP_LEVEL = ("template", "measurements")


def _coerce_measurements(raw: Any) -> Dict[str, float]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise MeasurementError(
            "the 'measurements' section must be a mapping of name: value"
        )
    out: Dict[str, float] = {}
    for key, value in raw.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError) as exc:
            raise MeasurementError(
                "measurement %r has a non numeric value %r" % (key, value)
            ) from exc
    return out


def load_measurement_file(path: str | Path) -> Dict[str, Any]:
    """Load and normalise a YAML measurement file.

    Returns a dictionary with the keys ``template``, ``style``, ``gender``,
    ``profile_name``, ``ease``, ``darts``, ``measurements`` and ``source``.
    Template defaults are applied but no validation is performed here.
    """
    path = Path(path)
    if not path.exists():
        raise MeasurementError("measurement file not found: %s" % path)

    with path.open("r", encoding="utf-8") as handle:
        try:
            raw = yaml.safe_load(handle) or {}
        except yaml.YAMLError as exc:
            raise MeasurementError("invalid YAML in %s: %s" % (path, exc)) from exc

    if not isinstance(raw, Mapping):
        raise MeasurementError("top level of %s must be a mapping" % path)

    for key in _REQUIRED_TOP_LEVEL:
        if key not in raw:
            raise MeasurementError("missing required key %r in %s" % (key, path))

    template = str(raw["template"]).strip().lower()
    if template not in TEMPLATES:
        raise MeasurementError(
            "unknown template %r in %s; expected one of %s"
            % (template, path, ", ".join(sorted(TEMPLATES)))
        )

    spec = TEMPLATES[template]
    style = str(raw.get("style", spec["default_style"]))
    gender = str(raw.get("gender", spec["default_gender"])).strip().lower()
    profile_name = str(
        raw.get("profile_name")
        or path.stem
    )
    ease = float(raw.get("ease", 8))
    darts = bool(raw.get("darts", False))

    config = {
        "template": template,
        "style": style,
        "gender": gender,
        "profile_name": profile_name,
        "ease": ease,
        "darts": darts,
        "measurements": _coerce_measurements(raw.get("measurements")),
        "source": str(path),
    }
    return config


def validate_measurement_file(config: Mapping[str, Any]) -> List[str]:
    """Validate a config from :func:`load_measurement_file`.

    Returns the list of missing measurements (empty when valid).
    """
    missing = missing_measurements(
        config["template"],
        config["style"],
        config["gender"],
        config["measurements"],
        darts=bool(config.get("darts", False)),
    )
    if missing:
        raise MeasurementError(
            "profile %r (%s / %s) is missing measurements: %s"
            % (
                config.get("profile_name", "?"),
                config["template"],
                config["style"],
                ", ".join(missing),
            )
        )
    return missing


# ---------------------------------------------------------------------------
# sqlite3 measurement database
# ---------------------------------------------------------------------------


def write_measurement_db(
    measurements: Mapping[str, Any], wkey: str, db_path: str | Path
) -> str:
    """Write ``measurements`` into a fresh sqlite3 database.

    The schema mirrors the database shipped with OpenPattern so that the
    drafting classes can read the bespoke values through their normal
    ``get_measurements_sql`` code path.
    """
    db_path = str(db_path)
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS measurements")
        cursor.execute(
            "CREATE TABLE measurements ("
            "wkey varchar(255), mtype varchar(255), mval float)"
        )
        for key, value in measurements.items():
            cursor.execute(
                "INSERT INTO measurements VALUES (?, ?, ?)",
                (wkey, str(key), float(value)),
            )
        conn.commit()
    finally:
        conn.close()

    return db_path
