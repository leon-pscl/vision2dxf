"""The generation pipeline.

    YAML -> validation and confirmation -> anatomical levels -> cross-sections
         -> proxy torso lofting -> constraint-preserving smoothing
         -> ease-adjusted garment surface -> predetermined seam placement
         -> panel segmentation -> constrained triangulation
         -> distortion-minimising flattening -> boundary extraction and curve fitting
         -> seam balancing -> geometric validation -> GarmageSet-like JSON

Every stage can refuse. A refusal is a structured list of field errors with a suggested
action, never a stack trace and never a silently repaired geometry.

No two-dimensional block is read, and no drafting formula is evaluated: the only route
from a measurement to a pattern edge runs through the reconstructed surface.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from .assumptions import AssumptionRegistry
from .config import drafting_config, reconstruction_config
from .drafting.cutting import CuttingLine, build_cutting_lines
from .drafting.ease import Settings
from .drafting.geometry import (
    CubicBezier,
    PatternDocument,
    PatternEdge,
    PatternPanel,
    Point2D,
    StitchPair,
    StitchRecord,
)
from .drafting.measurements import MeasurementError, SourceProfile, parse_source
from .drafting.semantics import Resolution, resolve
from .flattening.boundary_refit import EdgeFit, RefitError, refit_panel
from .flattening.distortion import heat_map, measure
from .flattening.parameterization import (
    Flattening,
    align_grain,
    arap,
    ensure_positive_orientation,
    lscm,
)
from .flattening.seam_balance import balance_pairs, surface_seam_lengths
from .flattening.triangulation import mesh_quality
from .garment_surface.ease import Ease, resolve_ease, underarm_drop
from .garment_surface.offset import build_garment_surface
from .garment_surface.seams import SeamError, SeamNetwork, build_seam_network
from .garment_surface.segmentation import SegmentationError, SurfacePanel, segment
from .reconstruction.anatomical_levels import LevelError, build_levels
from .reconstruction.cross_sections import build_sections
from .reconstruction.lofting import MeshError, RingSurface, loft
from .reconstruction.perimeter_solver import SectionError
from .reconstruction.smoothing import smooth
from .validation.geometry_checks import ValidationReport, validate

SOFTWARE_VERSION = "3.0.0"

# The seams two panels share, as (panel key, seam role) pairs. Predetermined, like the
# rest of the network.
STITCH_PAIRS = (
    (("front", "side_seam"), ("back", "side_seam")),
    (("front", "shoulder"), ("back", "shoulder")),
)
# The side whose boundary is walked against the other panel's direction.
REVERSED_PANEL = "back"


class GenerationBlocked(Exception):
    """A stage refused. Carries the structured reason and the HTTP-facing code."""

    def __init__(self, code: str, message: str, details: list[dict], stage: str) -> None:
        self.code = code
        self.message = message
        self.details = details
        self.stage = stage
        super().__init__(message)


@dataclass
class PanelResult:
    panel: SurfacePanel
    flattening: Flattening
    uv: np.ndarray
    quality: dict
    distortion: dict
    fits: list[EdgeFit]
    pattern_panel: PatternPanel


@dataclass
class GenerationResult:
    profile: SourceProfile
    resolution: Resolution
    settings: Settings
    ease: Ease
    levels: dict
    sections: dict
    proxy: RingSurface
    garment: RingSurface
    network: SeamNetwork
    panels: list[PanelResult]
    document: PatternDocument
    cutting: dict[str, CuttingLine]
    report: ValidationReport
    assumptions: AssumptionRegistry
    smoothing: dict
    garment_diagnostics: dict
    seam_balance: dict
    surface_seams: list[dict]
    progress: list[dict] = field(default_factory=list)

    # ---- the exported documents -------------------------------------------

    def cross_sections_json(self) -> dict:
        return {
            "levels": [self.levels[n].as_dict() for n in sorted(self.levels, key=lambda n: self.levels[n].y_mm)],
            "sections": [self.sections[n].as_dict() for n in self.sections],
            "model": self.settings.section_model,
            "residualToleranceMm": reconstruction_config()["cross_section"]["solver"]["residual_tolerance_mm"],
            "maxResidualMm": round(max((s.residual_mm for s in self.sections.values()), default=0.0), 6),
        }

    def flattening_diagnostics(self) -> dict:
        return {
            "method": self.settings.flattening_method,
            "weights": self.settings.weights(),
            "configVersion": reconstruction_config()["version"],
            "panels": [
                {
                    "panel": p.panel.key,
                    "mesh": p.quality,
                    "solver": p.flattening.as_dict(),
                    "distortion": {k: v for k, v in p.distortion.items()
                                   if k != "perTriangleAngularDeg"},
                    "boundaryFits": [f.as_dict() for f in p.fits],
                }
                for p in self.panels
            ],
            "crossSectionResiduals": {
                name: round(section.residual_mm, 6) for name, section in self.sections.items()
            },
            "garmentEaseResiduals": {
                name: entry["easeResidualMm"]
                for name, entry in self.garment_diagnostics["levels"].items()
            },
            "surfaceSeamLengths": self.surface_seams,
            "seamBalance": self.seam_balance,
            "smoothing": self.smoothing,
            "maxFlippedTriangles": max((p.distortion["flippedTriangles"] for p in self.panels),
                                       default=0),
            "maxMeanAngularDistortionDeg": max(
                (p.distortion["meanAngularDistortionDeg"] for p in self.panels), default=0.0),
        }

    def heat_maps(self) -> list[dict]:
        return [
            {
                "panel": p.panel.key,
                "label": p.panel.label,
                "center": (p.pattern_panel.center or Point2D(0, 0)).as_triple(),
                **heat_map(p.uv, p.panel.faces, p.distortion["perTriangleAngularDeg"]),
            }
            for p in self.panels
        ]

    def flattened_mesh(self) -> dict:
        return {
            "panels": [
                {
                    "panel": p.panel.key,
                    "vertices2d": [[round(float(x), 4), round(float(y), 4)] for x, y in p.uv],
                    "vertices3d": [[round(float(v), 4) for v in row] for row in p.panel.vertices_3d],
                    "faces": [[int(v) for v in f] for f in p.panel.faces],
                    "boundary": [int(i) for i in p.panel.boundary()],
                    "semanticBoundaries": {
                        role: [int(i) for i in ids]
                        for role, ids in p.panel.semantic_boundaries.items()
                    },
                }
                for p in self.panels
            ]
        }

    def manifest(self, filename: str) -> dict:
        cfg = reconstruction_config()
        return {
            "method": "anthropometric surface reconstruction and mesh flattening",
            "software_version": SOFTWARE_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source_file": filename,
            "units": {"source": self.settings.unit, "pattern": "mm"},
            "coordinate_convention": {
                "vertical_origin": "neck reference point",
                "vertical_direction": "+y downward",
                "ring_angle_zero": "centre front",
                "pattern_frame": "panel-local millimetres, placed by panel center",
            },
            "measurement_schema_version": self.resolution.schema_version,
            "definition_version": self.resolution.definition_version,
            "reconstruction_config": {
                "version": cfg["version"],
                "crossSection": cfg["cross_section"],
                "levels": cfg["levels"],
                "mesh": cfg["mesh"],
                "garment": cfg["garment"],
                "seams": cfg["seams"],
                "flattening": cfg["flattening"],
                "boundaryRefit": cfg["boundary_refit"],
            },
            "field_interpretations": {
                name: {
                    "meaning": f.meaning, "status": f.status,
                    "confirmed_by": f.confirmed_by, "candidate": f.candidate_id,
                }
                for name, f in self.resolution.fields.items()
            },
            "drafting_measurements_mm": (
                self.resolution.measurements.model_dump() if self.resolution.measurements else {}
            ),
            "derived_measurements": self.resolution.derived,
            "anatomical_levels": [
                self.levels[n].as_dict()
                for n in sorted(self.levels, key=lambda n: self.levels[n].y_mm)
            ],
            "cross_sections": [s.as_dict() for s in self.sections.values()],
            "proxy_surface": self.proxy.as_dict(),
            "garment_surface": self.garment_diagnostics,
            "seam_network": self.network.as_dict(),
            "assumptions": self.assumptions.as_list(),
            "semantic_assumptions": self.resolution.assumptions,
            "warnings": self.resolution.warnings,
            "settings": self.settings.as_dict(),
            "ease": self.ease.as_dict(),
            "tolerances": {**drafting_config()["tolerances"], **cfg["tolerances"]},
            "fixture_defaults": drafting_config()["fixture_defaults"],
            "validation": self.report.as_dict()["summary"],
            "unmapped_fields": UNMAPPED_FIELDS,
            "progress": self.progress,
        }


# Fields the reconstruction reads nowhere, with the reason. Published so no one assumes a
# measurement reached the geometry just because it was imported.
UNMAPPED_FIELDS = {
    "height": "no section or level is derived from stature in a torso garment",
    "head_l": "out of scope",
    "arm_length": "no sleeve in the first release",
    "wrist": "no sleeve in the first release",
    "leg_circ": "out of scope",
    "arm_pose_angle": "the proxy surface carries no arms, so the pose cannot be applied",
    "bust_points": "an apex protrusion, not a half-depth; used only to place the apex landmark",
    "bum_points": "out of scope while the hip section is off",
    "crotch_hip_diff": "crotch-referenced, and the hip section is off by default",
    "hips_line": "crotch-referenced, so it gives no level in the neck-referenced frame",
    "hip_inclination": "the sections are horizontal; a tilted hip section is not modelled",
    "waist_over_bust_line": "no seam or section in this preset uses an over-bust level",
    "underbust": "used for the underbust section only when the level assumption is accepted",
}


def resolve_semantics(
    text: str, filename: str, confirmations: dict | None = None, corrections: dict | None = None
) -> tuple[SourceProfile, Resolution]:
    """Import and resolve. Raises MeasurementError for an unusable file."""
    profile = parse_source(text, filename)
    return profile, resolve(profile.values_cm, confirmations, corrections)


def generate(resolution: Resolution, profile: SourceProfile, settings: Settings) -> GenerationResult:
    progress: list[dict] = []
    clock = time.perf_counter()

    def done(stage: str) -> None:
        nonlocal clock
        now = time.perf_counter()
        progress.append({"stage": stage, "status": "done",
                         "elapsedMs": round((now - clock) * 1000.0, 2)})
        clock = now

    if not settings.confirm_units:
        raise GenerationBlocked(
            "units_unconfirmed",
            "The assumed unit has to be confirmed before anything is converted.",
            [{"field": "unit", "code": "UNIT_UNCONFIRMED",
              "message": f"The file is read as {settings.unit}. Confirm it on the "
                         f"measurement review step.",
              "suggestedAction": "Confirm the unit.", "severity": "blocking"}],
            "measurements",
        )
    if resolution.measurements is None:
        raise GenerationBlocked(
            "semantics_unresolved",
            "Every required field needs a confirmed definition before a surface is built.",
            resolution.blocking, "measurements",
        )
    m = resolution.measurements
    registry = AssumptionRegistry()
    done("measurements")

    try:
        levels = build_levels(m, registry)
    except LevelError as exc:
        raise GenerationBlocked("anatomical_levels_inconsistent",
                                "The anatomical levels are inconsistent.", exc.errors,
                                "anatomical-levels") from exc
    done("anatomical-levels")

    try:
        sections = build_sections(m, levels, settings.section_model, registry,
                                  settings.exponent())
    except SectionError as exc:
        raise GenerationBlocked("cross_section_unsolvable",
                                "A cross-section cannot be reconstructed.", exc.errors,
                                "cross-sections") from exc
    done("cross-sections")

    try:
        proxy = loft(sections, levels, settings.resolution(), settings.segments())
        proxy, smoothing = smooth(proxy)
    except MeshError as exc:
        raise GenerationBlocked("proxy_surface_invalid",
                                "The proxy surface failed its mesh checks.", exc.errors,
                                "proxy-surface") from exc
    done("proxy-surface")

    ease = resolve_ease(settings)
    try:
        garment, garment_diagnostics = build_garment_surface(proxy, levels, ease, registry)
    except (MeshError, ValueError) as exc:
        details = getattr(exc, "errors", [{"field": "ease", "code": "EASE_INVALID",
                                           "message": str(exc),
                                           "suggestedAction": "Reduce the ease.",
                                           "severity": "blocking"}])
        raise GenerationBlocked("garment_surface_invalid",
                                "The garment surface could not be built.", details,
                                "garment-surface") from exc
    done("garment-surface")

    try:
        network = build_seam_network(garment, sections, levels, m, settings,
                                     underarm_drop(ease), registry)
    except SeamError as exc:
        raise GenerationBlocked("seam_network_invalid",
                                "The seam network cannot be placed on this surface.",
                                exc.errors, "seam-network") from exc
    done("seam-network")

    try:
        surface_panels = segment(garment, network)
    except SegmentationError as exc:
        raise GenerationBlocked("panel_segmentation_invalid",
                                "The surface could not be segmented into panels.",
                                exc.errors, "segmentation") from exc
    done("segmentation")

    surface_seams = surface_seam_lengths(surface_panels, STITCH_PAIRS)
    panels = [_flatten_panel(p, settings) for p in surface_panels]
    done("flattening")

    try:
        _refit(panels, settings)
    except RefitError as exc:
        raise GenerationBlocked("boundary_refit_failed",
                                "A panel boundary could not be reduced to pattern edges.",
                                exc.errors, "boundary-refit") from exc
    done("boundary-refit")

    document, seam_balance = _assemble(panels)
    done("seam-balancing")

    cutting = build_cutting_lines(document, settings)
    report = validate(document, cutting)
    done("validation")

    return GenerationResult(
        profile=profile, resolution=resolution, settings=settings, ease=ease, levels=levels,
        sections=sections, proxy=proxy, garment=garment, network=network, panels=panels,
        document=document, cutting=cutting, report=report, assumptions=registry,
        smoothing=smoothing, garment_diagnostics=garment_diagnostics,
        seam_balance=seam_balance, surface_seams=surface_seams, progress=progress,
    )


# ---- flattening -----------------------------------------------------------


def _fold_column(panel: SurfacePanel) -> int:
    """The grid column that lies on the fold: centre front, or centre back."""
    return 0 if panel.key == "front" else panel.cols - 1


def _flatten_panel(panel: SurfacePanel, settings: Settings) -> PanelResult:
    cfg = reconstruction_config()["flattening"]
    if settings.flattening_method not in ("lscm_arap", "", None):
        raise GenerationBlocked(
            "unknown_flattening_method",
            f"Flattening method '{settings.flattening_method}' is not implemented.",
            [{"field": "flattening_method", "code": "UNKNOWN_METHOD",
              "message": "The first release implements lscm_arap only.",
              "suggestedAction": "Select lscm_arap.", "severity": "blocking"}],
            "flattening",
        )

    verts, faces = panel.vertices_3d, panel.faces
    quality = mesh_quality(verts, faces)

    fold = _fold_column(panel)
    top = panel.vertex(0, fold)
    bottom = panel.vertex(panel.rows - 1, fold)

    uv = lscm(verts, faces, top, bottom)

    weights = settings.weights()
    # E_boundary and E_seam: hold every semantic interval corner where the conformal
    # initialisation put it, so a seam end cannot slide along the boundary during the
    # ARAP refinement.
    corners = [int(panel.semantic_boundaries[role][0]) for role in panel.boundary_order]
    soft = float(weights["w_boundary"] + weights["w_seam"])
    pins = {v: (float(uv[v, 0]), float(uv[v, 1])) for v in [*corners, top, bottom]}
    # Rigid-motion removal must not depend on the soft weights, so the two anchors keep a
    # weight of their own however the objective is tuned.
    weighting = {v: (1.0 if v in (top, bottom) else soft) for v in pins}
    flattening = arap(verts, faces, uv, pins, weighting)

    result = ensure_positive_orientation(flattening.uv, faces)
    result = align_grain(result, top, bottom)
    flattening.uv = result

    distortion = measure(result, faces, flattening.local, panel.boundary())

    errors: list[dict] = []
    if distortion["flippedTriangles"] > int(cfg["max_flipped_triangles"]):
        errors.append(_err(panel.key, "FLIPPED_TRIANGLES",
                           f"{distortion['flippedTriangles']} triangle(s) of the "
                           f"{panel.key} panel turned over when flattened."))
    if distortion["boundarySelfIntersections"]:
        errors.append(_err(panel.key, "BOUNDARY_SELF_INTERSECTION",
                           f"The flattened {panel.key} boundary crosses itself."))
    if not flattening.converged:
        errors.append(_err(panel.key, "FLATTENING_NOT_CONVERGED",
                           f"The {panel.key} panel did not converge in "
                           f"{flattening.iterations} iterations."))
    if errors:
        raise GenerationBlocked("flattening_failed", "A panel could not be flattened.",
                                errors, "flattening")

    return PanelResult(panel=panel, flattening=flattening, uv=result, quality=quality,
                       distortion=distortion, fits=[], pattern_panel=None)  # type: ignore[arg-type]


def _refit(panels: list[PanelResult], settings: Settings) -> None:
    offset = 0.0
    gap = float(drafting_config()["paper"]["gap_mm"])
    for result in panels:
        edges, fits = refit_panel(result.panel, result.uv, settings.seam_allowances)
        for edge in edges:
            edge.ensure_id()
        boundary_points = [point for edge in edges for point in edge.polyline()]
        min_x = min(point.x for point in boundary_points)
        max_x = max(point.x for point in boundary_points)
        min_y = min(point.y for point in boundary_points)
        max_y = max(point.y for point in boundary_points)
        center_x, center_y = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0
        for edge in edges:
            edge.start = _translated(edge.start, center_x, center_y)
            edge.end = _translated(edge.end, center_x, center_y)
            if edge.curve is not None:
                edge.curve = CubicBezier(
                    _translated(edge.curve.p0, center_x, center_y),
                    _translated(edge.curve.c1, center_x, center_y),
                    _translated(edge.curve.c2, center_x, center_y),
                    _translated(edge.curve.p3, center_x, center_y),
                )
            edge.points = [_translated(point, center_x, center_y) for point in edge.points]
        result.uv = result.uv - np.array([center_x, center_y])
        width = max_x - min_x
        pattern = PatternPanel(
            key=result.panel.key, label=result.panel.label, edges=edges,
            center=Point2D(offset + center_x, center_y),
            landmarks={}, grainline=(edges[0].start, edges[0].start),
        )
        fold_role = "center_front" if result.panel.key == "front" else "center_back"
        fold = pattern.edge_by_role(fold_role)
        if fold is not None:
            pattern.grainline = (fold.start, fold.end)
        pattern.ensure_id()
        result.fits = fits
        result.pattern_panel = pattern
        offset += width + gap


def _translated(point: Point2D, x: float, y: float) -> Point2D:
    return Point2D(point.x - x, point.y - y)


def _assemble(panels: list[PanelResult]) -> tuple[PatternDocument, dict]:
    index = {p.panel.key: p for p in panels}
    pairs: list[tuple[PatternEdge, PatternEdge]] = []
    for (key_a, role_a), (key_b, role_b) in STITCH_PAIRS:
        edge_a = index[key_a].pattern_panel.edge_by_role(role_a)
        edge_b = index[key_b].pattern_panel.edge_by_role(role_b)
        if edge_a is None or edge_b is None:
            continue
        edge_a.paired_edge_id = edge_b.ensure_id()
        edge_b.paired_edge_id = edge_a.ensure_id()
        pairs.append((edge_a, edge_b))

    balance = balance_pairs(pairs)

    stitches = [
        StitchPair(records=[
            StitchRecord(
                panel_id=index[key].pattern_panel.ensure_id(),
                start_edge_id=index[key].pattern_panel.edge_by_role(role).ensure_id(),
                start_param=0.0,
                end_edge_id=index[key].pattern_panel.edge_by_role(role).ensure_id(),
                end_param=1.0,
                is_counter_clockwise=(key == REVERSED_PANEL),
            )
            for key, role in ((key_a, role_a), (key_b, role_b))
        ])
        for (key_a, role_a), (key_b, role_b) in STITCH_PAIRS
    ]
    document = PatternDocument(
        panels=[p.pattern_panel for p in panels], stitches=stitches,
    )
    return document, balance


def _err(field: str, code: str, message: str) -> dict:
    return {
        "field": field, "code": code, "message": " ".join(message.split()),
        "suggestedAction": "Lower the ease, raise the mesh resolution, or review the seam "
                           "network.",
        "severity": "blocking",
    }
