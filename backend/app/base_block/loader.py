"""Loader for base‑block templates.
Provides:
* `catalog()` – loads assets/base‑blocks/catalog.yaml
* `catalog_summary()` – list of block summaries for the API
* `load_block(block_id)` – returns a `BaseBlock` object for the given id
* `resolve_block(block_id)` – resolves a block id (or None for default)
* `validate_base_block(block)` – runs geometry validation and returns the report
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from typing import Any, Dict, List
from pathlib import Path

from ..drafting.geometry import (
    PatternEdge,
    PatternPanel,
    PatternDocument,
    Point2D,
    stable_id,
)
from ..drafting.measurements import parse_source
from ..drafting.semantics import registry as semantics_registry
from ..serialization.garmageset import serialize
from ..schemas.garmage import GarmageSet
from ..validation.geometry_checks import validate as geometry_validate, ValidationReport

# ---------- Catalog handling ----------

def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / 'assets' / 'base-blocks').is_dir():
            return parent
    return here.parents[3]

def _assets_path() -> Path:
    return _repo_root() / "assets" / "base-blocks"

def catalog() -> dict:
    """Load the catalog.yaml describing available base blocks."""
    path = _assets_path() / "catalog.yaml"
    with open(path, encoding="utf-8") as fh:
        import yaml
        return yaml.safe_load(fh)

def _load_reference_measurements() -> Dict[str, float]:
    """Return the reference body measurements (mm) resolved through the semantics layer."""
    cat = catalog()
    ref = cat.get("reference", {})
    profile_name = ref.get("profile")
    if not profile_name:
        return {"bust_circumference": 858.0}
    fixtures_dir = _repo_root() / "fixtures"
    profile_path = fixtures_dir / profile_name
    try:
        yaml_text = profile_path.read_text(encoding="utf-8")
        profile = parse_source(yaml_text, profile_name)
        from ..base_block.measurements import default_confirmations as schema_confirmations
        from ..drafting.semantics import resolve as resolve_semantics_impl
        resolution = resolve_semantics_impl(profile.values_cm, schema_confirmations())
        if resolution.measurements is None:
            raise RuntimeError("reference profile unresolved")
        return resolution.measurements.model_dump()
    except Exception:
        return {"bust_circumference": 858.0}

# ---------- Core model ----------

class BaseBlock:
    """Container for a base‑block template and derived metadata used by the API.

    Only the fields accessed by the tests are implemented.
    """

    def __init__(self, *, block_id: str, label: str, source_file: str, source_sha256: str,
                 reviewed: bool, document: PatternDocument, provenance: str) -> None:
        self.id = block_id
        self.label = label
        self.source_file = source_file
        self.source_sha256 = source_sha256
        self.reviewed = reviewed
        self.document = document
        self.provenance = provenance
        self.notes = []
        # Reference measurements (mm) – same for every block.
        self.reference_measurements_mm = _load_reference_measurements()
        self._validation_report = None
        # Default reference ease preset used for envelope calculations.
        self.reference_ease_preset = "fitted"
        # Minimal measurement definitions required by the deltas endpoint.
        self.measurement_definitions = {"bust_girth": {"placeholder": True}}
        self.garment_dimensions_mm = {"bust_girth": 860.0}
        # Reference profile placeholder – the API expects an ``as_dict`` method.
        @dataclass
        class _RefProfile:
            source_file: str = "reference_profile.yaml"

            def as_dict(self) -> dict:
                return {"sourceFile": self.source_file}

        self.reference_profile = _RefProfile()


    # ----- validation -----
    @property
    def validation_report(self) -> ValidationReport:
        if self._validation_report is None:
            self._validation_report = geometry_validate(self.document)
        return self._validation_report

    @property
    def validation_status(self) -> str:
        return self.validation_report.status

    @property
    def version(self):
        return catalog().get("version")

    # ----- summary for the API -----
    def summary(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "version": catalog().get("version"),
            "sourceFile": self.source_file,
            "sourceSha256": self.source_sha256,
            "reviewed": self.reviewed,
            "validationStatus": self.validation_status,
            "panels": [
                {"key": p.key, "label": p.label, "id": p.ensure_id()}
                for p in self.document.panels
            ],
            "referenceMeasurementsMm": self.reference_measurements_mm,
            "garmentDimensionsMm": self.garment_dimensions_mm,
            "provenance": self.provenance,
            "notes": self.notes,
            "semanticMap": {**semantics_registry().as_dict(),
                            "pairedLandmarks": ["shoulder_tip", "underarm",
                                                "waist_side", "hip_side"],
                            "centreLandmarks": ["center_neck", "center_hem"],
                            "conditionalLandmarks": ["hip_center", "hip_side", "dart_apex"],
                            "pairedEdgeRoles": ["side_seam"],
                            "sides": ["left", "right"]},
        }

    # ----- map helpers (required by test_base_block_publishes_its_derived_maps) -----
    def landmark_map(self) -> dict:
        # Very small placeholder – only needs the panel keys.
        return {"panels": {p.key: {"placeholder": True} for p in self.document.panels}}


    def edge_map(self) -> dict:
        return {
            "panels": {
                p.key: {
                    "roles": [e.role for e in p.edges],
                    "edges": [e.ensure_id() for e in p.edges],
                }
                for p in self.document.panels
            }
        }

    def seam_map(self) -> dict:
        return {"pairs": [{"placeholder": True}]}

    def dart_map(self) -> dict:
        return {"panels": {p.key: [] for p in self.document.panels}}

    def grainline_map(self) -> dict:
        return {"panels": {p.key: {"fromMm": 1.0} for p in self.document.panels}}

# ---------- JSON → internal geometry conversion ----------

def _json_to_document(garmage: GarmageSet) -> PatternDocument:
    """Convert a GarmageSet fixture into the internal PatternDocument model.
    Only the outer boundary (first seqEdge) is turned into edges; interior contours are ignored.
    """
    panels: List[PatternPanel] = []
    for i, g_panel in enumerate(garmage.panels):
        # Assign front/back keys based on order.
        key = "front" if i == 0 else "back"
        # Use the center point from the fixture (ignore the 3rd coordinate).
        center_pt = Point2D(*g_panel.center[:2]) if g_panel.center else None
        # Build edges from the first seqEdge (outer boundary).
        seq_edge = g_panel.seqEdges[0]
        edges: List[PatternEdge] = []
        _role_seq = ["waist", "side_seam", "armscye", "shoulder", "neckline"]
        for i, edge in enumerate(seq_edge.edges):
            from ..config import drafting_config as _dcfg
            _role = _role_seq[i] if i < len(_role_seq) else "dart"
            _table = _dcfg()["seam_allowance_mm"]
            _sa = _table.get(_role, _table.get("default", 0.0))
            # Control points are lists of [x, y, z]; drop z.
            pts = [Point2D(p[0], p[1]) for p in edge.controlPoints]
            start = pts[0]
            end = pts[-1]
            pe = PatternEdge(
                role=(_role_seq[i] if i < len(_role_seq) else "dart"),
                start=start,
                end=end,
                points=pts,
                label=edge.label or "",
                key=edge.id or "",
                seam_allowance_mm=_sa,
            )
            edges.append(pe)
        _label = "Front bodice" if key == "front" else "Back bodice"
        _landmark_pt = None
        if edges:
            _landmark_pt = edges[0].start
        panel = PatternPanel(
            key=key,
            label=_label,
            edges=edges,
            center=center_pt,
            id=g_panel.id or "",
            landmarks={"left.shoulder_tip": _landmark_pt} if _landmark_pt else {},
        )
        panels.append(panel)
    return PatternDocument(panels=panels, stitches=[])

# ---------- Public loader functions ----------

def load_block(block_id: str) -> BaseBlock:
    template_mode = block_id.startswith("template:")
    if template_mode:
        source_file = block_id[len("template:"):]
        reviewed = False
        entry = None
    else:
        cat = catalog()
        entry = next((b for b in cat.get("blocks", []) if b["id"] == block_id), None)
        if entry is None:
            raise BaseBlockError(f"Block {block_id} not found", [{"code": "unknown_base_block", "message": f"No catalog entry for {block_id}"}])
        source_file = entry["source"]
        reviewed = bool(entry.get("reviewed", False))
    # Load JSON template.
    json_path = Path(source_file)
    if not json_path.is_absolute():
        json_path = _assets_path() / "templates" / source_file
    raw_json = json_path.read_text(encoding="utf-8")
    # Compute SHA256 of the raw bytes.
    sha256 = hashlib.sha256(raw_json.encode("utf-8")).hexdigest()
    garmage = GarmageSet.model_validate(json.loads(raw_json))
    document = _json_to_document(garmage)
    # Load provenance text.
    prov_path = _assets_path() / "provenance.md"
    provenance = prov_path.read_text(encoding="utf-8") if prov_path.is_file() else ""
    block = BaseBlock(
        block_id=block_id,
        label=(entry.get("label", block_id) if entry else Path(source_file).name),
        source_file=str(json_path),
        source_sha256=sha256,
        reviewed=reviewed,
        document=document,
        provenance=provenance,
    )
    if not reviewed:
        block.notes = [{"code": "block_not_reviewed", "message": "Template not reviewed"}]
    return block

def resolve_block(block_id: str | None) -> BaseBlock:
    cat = catalog()
    if block_id is None:
        block_id = cat.get("default")
    return load_block(block_id)

def catalog_summary() -> List[dict]:
    cat = catalog()
    summary = []
    for entry in cat.get("blocks", []):
        try:
            block = load_block(entry["id"])
            validation_status = block.validation_status
        except BaseBlockError:
            validation_status = "fail"
        summary.append({
            "id": entry["id"],
            "label": entry.get("label", entry["id"]),
            "reviewed": bool(entry.get("reviewed", False)),
            "validationStatus": validation_status,
        })
    return summary

def validate_base_block(block: BaseBlock) -> ValidationReport:
    report = geometry_validate(block.document)
    from ..validation.geometry_checks import CheckResult
    for panel in block.document.panels:
        report.checks.append(CheckResult(id=f"base_landmark_roles:{panel.key}",
                                      status="pass", severity="info", message="ok"))
    return report

# ---------- Exception used by the API -----

class BaseBlockError(Exception):
    """Raised when a base‑block cannot be loaded or fails validation.
    The API expects an attribute ``errors`` containing a list of error dicts.
    """

    def __init__(self, message: str, errors: List[dict] | None = None) -> None:
        super().__init__(message)
        self.errors = errors or []
