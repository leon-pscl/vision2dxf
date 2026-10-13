"""Measurement Semantics Layer.

Raw YAML values never reach a drafting equation. They pass through:

    source value -> semantic definition -> unit + representation normalization
                 -> consistency rules -> operator confirmation -> DraftingMeasurements

`DraftingMeasurements` is the only thing the rule registry sees. Its field names are
descriptive, so `body["vert_bust_line"]` has no route into an equation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict

from ..config import semantics_config

CM_TO_MM = 10.0

CONFIRMED = "CONFIRMED"
DERIVED = "DERIVED"
ASSUMED = "ASSUMED"
OPTIONAL = "OPTIONAL"
MISSING = "MISSING"
AMBIGUOUS = "AMBIGUOUS"
INVALID = "INVALID"

BLOCKING_STATUSES = {MISSING, AMBIGUOUS, INVALID}
ORDER = (CONFIRMED, DERIVED, ASSUMED, OPTIONAL, MISSING, AMBIGUOUS, INVALID)


class SemanticError(Exception):
    """Raised for a broken registry definition (developer error, not operator error)."""

    def __init__(self, message: str, field_name: str | None = None) -> None:
        super().__init__(message)
        self.field_name = field_name


@dataclass
class Candidate:
    id: str
    meaning: str
    representation: str | None
    start_landmark: str | None
    end_landmark: str | None
    origin: str | None
    reference_axis: str | None
    sign_convention: str | None
    axis: str | None
    drafting_variable: str
    scale: float
    transform: str | None
    add_variable: str | None
    subtract_variable: str | None

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "meaning": " ".join(self.meaning.split()),
            "representation": self.representation,
            "startLandmark": self.start_landmark,
            "endLandmark": self.end_landmark,
            "origin": self.origin,
            "referenceAxis": self.reference_axis,
            "signConvention": self.sign_convention,
            "axis": self.axis,
            "draftingVariable": self.drafting_variable,
            "scale": self.scale,
        }


@dataclass
class FieldDefinition:
    name: str
    label: str
    category: str
    quantity_type: str
    required: bool
    status: str  # registry status: confirmed | ambiguous | optional
    unit: str
    range: tuple[float, float] | None
    definition: str
    candidates: list[Candidate]
    origin: str | None
    drafting_variable: str | None
    landmarks: tuple[str | None, str | None]

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "category": self.category,
            "quantityType": self.quantity_type,
            "required": self.required,
            "registryStatus": self.status,
            "unit": self.unit,
            "range": list(self.range) if self.range else None,
            "definition": " ".join(self.definition.split()),
            "origin": self.origin,
            "startLandmark": self.landmarks[0],
            "endLandmark": self.landmarks[1],
            "draftingVariable": self.drafting_variable,
            "candidates": [c.as_dict() for c in self.candidates],
        }


def _candidate(raw: dict) -> Candidate:
    return Candidate(
        id=raw["id"],
        meaning=raw.get("meaning", ""),
        representation=raw.get("representation"),
        start_landmark=raw.get("start_landmark"),
        end_landmark=raw.get("end_landmark"),
        origin=raw.get("origin"),
        reference_axis=raw.get("reference_axis"),
        sign_convention=raw.get("sign_convention"),
        axis=raw.get("axis"),
        drafting_variable=raw["drafting_variable"],
        scale=float(raw.get("scale", 1.0)),
        transform=raw.get("transform"),
        add_variable=raw.get("add_drafting_variable"),
        subtract_variable=raw.get("subtract_drafting_variable"),
    )


class SemanticsRegistry:
    """The developer-owned schema configuration."""

    def __init__(self, raw: dict) -> None:
        self.raw = raw
        self.schema_version = raw["schema_version"]
        self.definition_version = raw["definition_version"]
        self.coordinate_system = raw["coordinate_system"]
        self.consistency_rules = raw.get("consistency_rules", [])
        self.fields: dict[str, FieldDefinition] = {}
        for name, spec in raw["fields"].items():
            candidates = [_candidate(c) for c in spec.get("candidates", [])]
            if not candidates and not spec.get("drafting_variable"):
                raise SemanticError(f"field {name} has no candidate and no drafting variable", name)
            if not candidates:
                # A single authoritative definition is modelled as one implicit candidate,
                # so the downstream resolution path stays uniform.
                candidates = [_candidate({
                    "id": "schema_definition",
                    "meaning": " ".join(spec.get("definition", "").split()),
                    "representation": spec.get("representation"),
                    "start_landmark": spec.get("start_landmark"),
                    "end_landmark": spec.get("end_landmark"),
                    "origin": spec.get("origin"),
                    "axis": spec.get("axis"),
                    "reference_axis": spec.get("reference_axis"),
                    "sign_convention": spec.get("sign_convention"),
                    "drafting_variable": spec["drafting_variable"],
                    "scale": 1.0,
                })]
            if spec.get("status") == "ambiguous" and not spec.get("candidates"):
                raise SemanticError(f"ambiguous field {name} offers no candidate", name)
            required_meta = self.raw["categories"][spec["category"]]["required_metadata"]
            missing = {
                meta for meta in required_meta
                if not spec.get(meta) and not any(getattr(c, meta, None) for c in candidates)
            }
            if missing:
                raise SemanticError(f"field {name} misses required metadata {sorted(missing)}", name)
            self.fields[name] = FieldDefinition(
                name=name,
                label=spec["label"],
                category=spec["category"],
                quantity_type=spec["quantity_type"],
                required=bool(spec.get("required", False)),
                status=spec["status"],
                unit=spec.get("unit", self.coordinate_system["unit"]),
                range=(tuple(spec["range"]) if spec.get("range") else None),
                definition=spec.get("definition", ""),
                candidates=candidates,
                origin=spec.get("origin"),
                drafting_variable=spec.get("drafting_variable"),
                landmarks=(spec.get("start_landmark"), spec.get("end_landmark")),
            )

    def origin_of(self, name: str) -> str | None:
        return self.fields[name].origin

    def as_dict(self) -> dict:
        return {
            "schemaVersion": self.schema_version,
            "definitionVersion": self.definition_version,
            "coordinateSystem": {
                "verticalOrigin": self.coordinate_system["vertical_origin"],
                "verticalDirection": self.coordinate_system["vertical_direction"],
                "unit": self.coordinate_system["unit"],
                "patternUnit": self.coordinate_system["pattern_unit"],
                "patternFrame": self.coordinate_system["pattern_frame"],
                "angles": self.coordinate_system["angles"],
            },
            "categories": self.raw["categories"],
            "consistencyRules": self.consistency_rules,
            "fields": [f.as_dict() for f in self.fields.values()],
            "statuses": list(ORDER),
            "blockingStatuses": sorted(BLOCKING_STATUSES),
        }


_registry: SemanticsRegistry | None = None


def registry() -> SemanticsRegistry:
    global _registry
    if _registry is None:
        _registry = SemanticsRegistry(semantics_config())
    return _registry


@dataclass
class MeasurementWarning:
    """A drafting-time note about a confirmed value; never changes it."""

    code: str
    field: str | None
    message: str
    original_value: Any = None
    action: str = "review"


@dataclass
class FieldResolution:
    """One source field after semantics resolution and normalization."""

    name: str
    label: str
    category: str
    quantity_type: str
    status: str
    source_value: float | None
    original_value: float | None
    normalized_mm: float | None
    unit_source: str
    unit_pattern: str
    conversion: str
    meaning: str
    representation: str | None
    start_landmark: str | None
    end_landmark: str | None
    origin: str | None
    drafting_variable: str | None
    candidate_id: str | None
    confirmed_by: str | None
    definition_version: str
    schema_version: str
    corrected: bool
    required: bool
    registry_status: str
    message: str | None = None
    suggested_action: str | None = None

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "category": self.category,
            "quantityType": self.quantity_type,
            "status": self.status,
            "required": self.required,
            "registryStatus": self.registry_status,
            "sourceValue": self.source_value,
            "originalValue": self.original_value,
            "normalizedMm": self.normalized_mm,
            "unitSource": self.unit_source,
            "unitPattern": self.unit_pattern,
            "conversion": self.conversion,
            "meaning": self.meaning,
            "representation": self.representation,
            "startLandmark": self.start_landmark,
            "endLandmark": self.end_landmark,
            "origin": self.origin,
            "draftingVariable": self.drafting_variable,
            "candidateId": self.candidate_id,
            "confirmedBy": self.confirmed_by,
            "definitionVersion": self.definition_version,
            "schemaVersion": self.schema_version,
            "corrected": self.corrected,
            "message": self.message,
            "suggestedAction": self.suggested_action,
        }


class DraftingMeasurements(BaseModel):
    """Confirmed, normalized, drafting-ready values in millimetres.

    Drafting equations address these fields, never the camera subsystem's names.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # circumferences
    bust_circumference: float
    waist_circumference: float
    hip_circumference: float | None = None
    underbust_circumference: float | None = None
    # widths
    shoulder_width_full: float
    back_width_full: float
    waist_back_width_full: float
    hip_back_width_full: float | None = None
    neck_width_half: float
    # vertical levels (pattern frame: y = 0 at the neck reference point, +y down)
    neck_to_bust_level: float
    neck_to_waist_level: float
    neck_to_bust_apex_vertical: float
    # relative vertical distances
    bust_to_waist_vertical: float
    waist_to_hip_vertical: float | None = None
    armscye_depth_vertical: float
    # landmark spacing
    bust_apex_spacing_full: float
    buttock_spacing_full: float | None = None
    # angles
    shoulder_slope_degrees: float
    # out of bodice scope, carried for the manifest only
    stature: float | None = None
    arm_length: float | None = None


@dataclass
class Resolution:
    fields: dict[str, FieldResolution]
    derived: dict[str, dict]
    assumptions: list[dict]
    warnings: list[dict]
    blocking: list[dict]
    measurements: DraftingMeasurements | None
    schema_version: str
    definition_version: str
    coordinate_origin: str = "neck_reference_point"

    @property
    def can_generate(self) -> bool:
        return not self.blocking

    def as_dict(self) -> dict:
        return {
            "schemaVersion": self.schema_version,
            "definitionVersion": self.definition_version,
            "canGenerate": self.can_generate,
            "fields": [self.fields[name].as_dict() for name in self.fields],
            "derivedMeasurements": self.derived,
            "assumptions": self.assumptions,
            "warnings": self.warnings,
            "blocking": self.blocking,
        }


def resolve(
    values: dict[str, float],
    confirmations: dict[str, str] | None = None,
    corrections: dict[str, float] | None = None,
) -> Resolution:
    reg = registry()
    confirmations = confirmations or {}
    corrections = corrections or {}

    fields: dict[str, FieldResolution] = {}
    blocking: list[dict] = []
    warnings: list[dict] = []
    assumptions: list[dict] = []
    resolved: dict[str, float] = {}  # drafting variable -> mm
    origins: dict[str, str] = {}

    for name, fdef in reg.fields.items():
        original = values.get(name)
        corrected = name in corrections
        value = float(corrections[name]) if corrected else original

        if value is None:
            status = MISSING if fdef.required else OPTIONAL
            message = f"{fdef.label} is required but absent." if fdef.required else None
            if fdef.required:
                blocking.append(_block(name, "MISSING", message,
                                       "Return to measurement capture or supply the field."))
            fields[name] = _blank(fdef, reg, status, message)
            continue

        lo, hi = fdef.range or (-1e9, 1e9)
        if not (lo <= value <= hi):
            blocking.append(_block(name, "INVALID",
                                   f"{fdef.label} is {value} {fdef.unit}, outside {lo}-{hi}.",
                                   "Correct the value or fix the schema range."))
            fields[name] = _blank(fdef, reg, INVALID, f"{fdef.label} is outside its schema range.")
            fields[name].original_value = original
            fields[name].source_value = value
            fields[name].corrected = corrected
            continue

        candidate_id = confirmations.get(name)
        pinned = fdef.status in ("confirmed", "optional") and fdef.candidates[0].id == "schema_definition"
        if candidate_id is None and pinned:
            candidate_id = fdef.candidates[0].id
            confirmed_by = "schema"
        elif candidate_id is not None:
            confirmed_by = "operator"
        else:
            confirmed_by = None

        if candidate_id is None:
            status = MISSING if fdef.required else OPTIONAL
            if fdef.required:
                blocking.append(_block(name, "AMBIGUOUS",
                                       f"{fdef.label} has no confirmed definition.",
                                       "Confirm one of the offered interpretations."))
                fields[name] = _blank(fdef, reg, AMBIGUOUS, "Definition not confirmed.")
                fields[name].original_value = original
                fields[name].source_value = value
                fields[name].corrected = corrected
                continue
            fields[name] = _blank(fdef, reg, OPTIONAL, "Not required by the selected garment.")
            fields[name].original_value = original
            fields[name].source_value = value
            fields[name].corrected = corrected
            continue

        candidate = next((c for c in fdef.candidates if c.id == candidate_id), None)
        if candidate is None:
            # Bad operator input, not a broken schema: stay blocked and say why.
            offered = ", ".join(c.id for c in fdef.candidates)
            message = (f"Confirmation '{candidate_id}' is not a candidate of {fdef.label}. "
                       f"Offered: {offered}.")
            if fdef.required:
                blocking.append(_block(name, "AMBIGUOUS", message,
                                       "Pick one of the offered interpretations."))
            fields[name] = _blank(fdef, reg, AMBIGUOUS, message)
            fields[name].original_value = original
            fields[name].source_value = value
            fields[name].corrected = corrected
            continue

        # Angles are already in degrees and must not be scaled by the length conversion.
        is_angle = fdef.category == "angle"
        if is_angle:
            normalized = (90.0 - value) if candidate.transform == "subtract_from_ninety" else value
            conversion = "deg -> deg (identity)"
            if candidate.transform:
                conversion += f"; transform={candidate.transform}"
        else:
            normalized = value * candidate.scale * CM_TO_MM
            conversion = f"cm -> mm (x{CM_TO_MM:g})"
            if candidate.scale != 1.0:
                conversion += f"; {candidate.representation} -> canonical (x{candidate.scale:g})"
            if candidate.transform:
                conversion += f"; transform={candidate.transform}"

        status = ASSUMED if confirmed_by == "operator" and fdef.status != "confirmed" else CONFIRMED
        if status == ASSUMED:
            assumptions.append({
                "field": name,
                "kind": "operator_confirmed_definition",
                "meaning": candidate.meaning,
                "candidate": candidate.id,
                "message": (f"{fdef.label} is not pinned by the schema (schema status "
                            f"{fdef.status}); the operator confirmed '{candidate.id}'."),
            })
        if corrected:
            assumptions.append({
                "field": name,
                "kind": "corrected_value",
                "original": original,
                "corrected": value,
                "message": f"{fdef.label} corrected from {original} to {value} {fdef.unit}.",
            })

        res = FieldResolution(
            name=name, label=fdef.label, category=fdef.category, quantity_type=fdef.quantity_type,
            status=status, source_value=value, original_value=original, normalized_mm=normalized,
            unit_source=fdef.unit, unit_pattern="mm", conversion=conversion,
            meaning=candidate.meaning or " ".join(fdef.definition.split()),
            representation=candidate.representation,
            start_landmark=candidate.start_landmark or fdef.landmarks[0],
            end_landmark=candidate.end_landmark or fdef.landmarks[1],
            origin=candidate.origin or fdef.origin,
            drafting_variable=candidate.drafting_variable, candidate_id=candidate.id,
            confirmed_by=confirmed_by, definition_version=reg.definition_version,
            schema_version=reg.schema_version, corrected=corrected,
            required=fdef.required, registry_status=fdef.status,
        )
        fields[name] = res
        resolved[candidate.drafting_variable] = normalized
        origins[name] = res.origin or ""

    # ---- candidate cross-references (e.g. bust_line offset + vert_bust_line) --
    for name, res in fields.items():
        if res.status not in (CONFIRMED, ASSUMED):
            continue
        fdef = reg.fields[name]
        candidate = next(c for c in fdef.candidates if c.id == res.candidate_id)
        if not candidate.add_variable and not candidate.subtract_variable:
            continue  # nothing to adjust: keep the scaled normalization from the first pass
        value = res.source_value if fdef.category == "angle" else res.source_value * candidate.scale * CM_TO_MM
        if candidate.add_variable and candidate.add_variable in resolved:
            value += resolved[candidate.add_variable]
            res.conversion += f"; + {candidate.add_variable}"
        if candidate.subtract_variable:
            drop = _shoulder_tip_drop(resolved)
            if drop is None:
                blocking.append(_block(name, "AMBIGUOUS",
                                       f"{name} = '{candidate.id}' needs the drafted shoulder drop, "
                                       f"which depends on shoulder_w and shoulder_incl.",
                                       "Confirm shoulder_w and shoulder_incl first."))
                res.status = AMBIGUOUS
                continue
            value -= drop
            res.conversion += f"; - shoulder_tip_drop_below_neck_reference ({drop:.1f} mm)"
        res.normalized_mm = value
        resolved[candidate.drafting_variable] = value

    # ---- derived drafting variables ------------------------------------------
    derived: dict[str, dict] = {}
    measurements = None
    for rule in reg.consistency_rules:
        _run_rule(rule, reg, fields, resolved, warnings, blocking)
    if not blocking:
        measurements, derived = _build_measurements(reg, resolved, blocking)

    return Resolution(
        fields=fields, derived=derived, assumptions=assumptions, warnings=warnings,
        blocking=blocking, measurements=measurements,
        schema_version=reg.schema_version, definition_version=reg.definition_version,
        coordinate_origin=reg.coordinate_system["vertical_origin"],
    )


def _shoulder_tip_drop(resolved: dict[str, float]) -> float | None:
    """Shoulder-tip drop below the neck reference point, needed by one armscye candidate."""
    shoulder = resolved.get("shoulder_width_full")
    slope = resolved.get("shoulder_slope_degrees")
    if shoulder is None or slope is None:
        return None
    from math import radians, sin

    from ..config import drafting_config
    seam = shoulder / 2.0 - drafting_config()["constants"]["shoulder_neck_offset_mm"]
    return seam * sin(radians(slope))


def _block(field_name: str, code: str, message: str, action: str) -> dict:
    return {"field": field_name, "code": code, "message": message, "suggestedAction": action,
            "severity": "blocking"}


def _blank(fdef: FieldDefinition, reg: SemanticsRegistry, status: str, message: str | None) -> FieldResolution:
    first = fdef.candidates[0] if fdef.candidates else None
    return FieldResolution(
        name=fdef.name, label=fdef.label, category=fdef.category, quantity_type=fdef.quantity_type,
        status=status, source_value=None, original_value=None, normalized_mm=None,
        unit_source=fdef.unit, unit_pattern="mm", conversion="none",
        meaning=first.meaning if first else " ".join(fdef.definition.split()),
        representation=first.representation if first else None,
        start_landmark=first.start_landmark if first else fdef.landmarks[0],
        end_landmark=first.end_landmark if first else fdef.landmarks[1],
        origin=(first.origin if first else fdef.origin),
        drafting_variable=first.drafting_variable if first else fdef.drafting_variable,
        candidate_id=None, confirmed_by=None, definition_version=reg.definition_version,
        schema_version=reg.schema_version, corrected=False, required=fdef.required,
        registry_status=fdef.status, message=message,
        suggested_action="Confirm interpretation" if status == AMBIGUOUS else None,
    )


def _run_rule(rule, reg, fields, resolved, warnings, blocking) -> None:
    """Cross-field consistency. A blocking failure is a schema-validation message."""
    rid = rule["id"]
    if rid == "shared_vertical_origin":
        group = rule["members"]
        chosen = {}
        for name in group:
            res = fields.get(name)
            if res is None or res.status in BLOCKING_STATUSES:
                continue
            if res.origin:
                chosen[name] = res.origin
        if len(set(chosen.values())) > 1:
            detail = ", ".join(f"{k} -> {v}" for k, v in chosen.items())
            blocking.append(_block(
                ",".join(group), "SHARED_VERTICAL_ORIGIN",
                f"Vertical fields used in a difference declare different origins ({detail}). "
                f"Confirm a single origin before drafting.",
                "Review the vertical field definitions in the schema configuration."))
        return
    if rid == "neck_reference_level_available":
        for name in rule["fields"]:
            res = fields.get(name)
            if res and res.status not in BLOCKING_STATUSES and res.origin == rule["forbidden_origin"]:
                blocking.append(_block(
                    name, "NECK_REFERENCE_LEVEL_UNAVAILABLE",
                    f"{name} is measured from {rule['forbidden_origin']}, but the pattern frame "
                    f"starts at the {reg.coordinate_system['pattern_frame']['y_zero_landmark']} and no "
                    f"level for it has been confirmed.",
                    "Confirm the neck-reference level or change the schema origin."))
        return
    if rid == "bust_apex_below_bust_line":
        a, b = resolved.get("neck_to_bust_level"), resolved.get("neck_to_bust_apex_vertical")
        if a is not None and b is not None and b <= a:
            blocking.append(_block(
                "vert_bust_line", "BUST_APEX_NOT_BELOW_BUST_LINE",
                f"The bust apex ({b:.0f} mm) does not sit below the bust line ({a:.0f} mm) "
                f"under the confirmed interpretations.",
                "Re-check the origin chosen for bust_line and vert_bust_line."))
        return
    if rid == "waist_below_bust":
        a, b = resolved.get("neck_to_bust_level"), resolved.get("neck_to_waist_level")
        if a is not None and b is not None:
            gap = (b - a) / CM_TO_MM
            if gap < rule["minimum_gap_source_cm"]:
                blocking.append(_block(
                    "waist_line", "WAIST_NOT_BELOW_BUST",
                    f"The waist line sits {gap:.1f} cm below the bust line; at least "
                    f"{rule['minimum_gap_source_cm']:.0f} cm is required.",
                    "Confirm the vertical origins, or correct the measured values."))
        return
    if rid == "back_share_of_girth":
        girth, back = resolved.get("bust_circumference"), resolved.get("back_width_full")
        if girth and back:
            share = back / girth
            if not 0.30 <= share <= 0.60:
                warnings.append({
                    "field": "back_width", "code": "BACK_SHARE_OF_GIRTH", "severity": "warning",
                    "message": (f"The confirmed back width takes {share:.0%} of the bust girth; "
                                f"30-60% is expected."),
                })
        return
    if rid == "underbust_below_bust":
        bust, under = resolved.get("bust_circumference"), resolved.get("underbust_circumference")
        if bust and under and bust - under < 10.0:
            warnings.append({
                "field": "underbust", "code": "UNDERBUST_NOT_BELOW_BUST", "severity": "warning",
                "message": "Underbust is not clearly smaller than bust; check the scan.",
            })


def _build_measurements(reg, resolved, blocking):
    """Derived drafting variables and the adapter model the drafting engine consumes."""
    derived: dict[str, dict] = {}

    def record(name: str, value: float, equation: str, sources: list[str]) -> None:
        derived[name] = {"value": round(value, 4), "unit": "mm", "equation": equation,
                         "dependencies": sources, "status": DERIVED}

    required = [
        "bust_circumference", "waist_circumference", "shoulder_width_full", "back_width_full",
        "waist_back_width_full", "neck_width_half", "neck_to_bust_level", "neck_to_waist_level",
        "neck_to_bust_apex_vertical", "armscye_depth_vertical", "bust_apex_spacing_full",
        "shoulder_slope_degrees",
    ]
    missing = [name for name in required if name not in resolved]
    if missing:
        for name in missing:
            blocking.append(_block(name, "MISSING",
                                   f"Drafting variable {name} has no confirmed source field.",
                                   "Confirm the field that provides it."))
        return None, derived

    bust_level = resolved["neck_to_bust_level"]
    waist_level = resolved["neck_to_waist_level"]
    bust_to_waist = abs(waist_level - bust_level)
    record("bust_to_waist_vertical", bust_to_waist,
           "bust_to_waist_vertical = |neck_to_waist_level - neck_to_bust_level|",
           ["neck_to_bust_level", "neck_to_waist_level"])

    # hips_line is measured from the crotch point, not the neck reference point, so it
    # cannot be placed in the pattern frame without a confirmed crotch level.
    derived["waist_to_hip_vertical"] = {
        "value": None, "unit": "mm", "status": "OPTIONAL",
        "equation": "not derivable: hips_line uses the crotch origin, not the neck reference point",
        "dependencies": ["crotch_to_hip_level"], "message": "Outside the bodice scope.",
    }

    data: dict[str, Any] = {k: v for k, v in resolved.items() if k in DraftingMeasurements.model_fields}
    data["bust_to_waist_vertical"] = bust_to_waist
    try:
        measurements = DraftingMeasurements(**data)
    except Exception as exc:
        blocking.append(_block(None, "INVALID", f"Drafting measurements are inconsistent: {exc}",
                               "Re-check the confirmed interpretations."))
        return None, derived
    return measurements, derived


# The interpretation the prototype assumes when an operator has not chosen one. Each entry
# is a real choice between the candidates the registry offers, not a default the schema
# declares, so the resolution reports every one of them as ASSUMED rather than CONFIRMED.
DEFAULT_CONFIRMATIONS = {
    "back_width": "across_back_side_to_side",
    "waist_back_width": "across_back_side_to_side",
    "hip_back_width": "across_back_side_to_side",
    "neck_w": "half_neck_girth",
    "shoulder_w": "half_neck_to_shoulder_tip",
    "shoulder_incl": "below_horizontal",
    "bust_line": "level_from_neck_reference_point",
    "waist_line": "level_from_neck_reference_point",
    "waist_over_bust_line": "level_from_neck_reference_point",
    "vert_bust_line": "neck_reference_point_to_bust_apex",
    # The default interpretation of the source measurement is the direct vertical drop from
    # the shoulder-tip level to the underarm point. The neck-reference variant is still
    # supported as an explicit operator choice when that is how a given dataset defines it.
    "armscye_depth": "shoulder_tip_level_to_underarm",
    "bust_points": "chest_wall_to_apex",
    "bum_points": "lower_back_to_apex",
    "hips_line": "level_from_crotch_point",
    "hip_inclination": "below_horizontal",
    "arm_pose_angle": "from_vertical",
}


def default_confirmations() -> dict[str, str]:
    """A copy, so a caller cannot mutate the table."""
    return dict(DEFAULT_CONFIRMATIONS)
