"""Geometric validation. Every critical check blocks export; the report is written
to validation-report.json with status, metrics, affected ids and suggested actions."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from shapely.geometry import LinearRing, Polygon

from ..config import drafting_config
from ..drafting.cutting import CuttingLine
from ..drafting.geometry import PatternDocument

CRITICAL = "critical"
WARNING = "warning"


@dataclass
class CheckResult:
    id: str
    status: str  # pass | fail
    severity: str  # critical | warning
    message: str
    affected_ids: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    suggested_action: str = ""

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "severity": self.severity,
            "message": self.message,
            "affectedIds": self.affected_ids,
            "metrics": self.metrics,
            "suggestedAction": self.suggested_action,
        }


@dataclass
class ValidationReport:
    status: str
    checks: list[CheckResult]
    metrics: dict[str, Any]
    warnings: list[dict]

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "passed": self.status == "pass",
            "summary": {
                "total": len(self.checks),
                "passed": sum(1 for c in self.checks if c.status == "pass"),
                "failed": sum(1 for c in self.checks if c.status == "fail"),
                "criticalFailures": sum(
                    1 for c in self.checks if c.status == "fail" and c.severity == CRITICAL
                ),
            },
            "metrics": self.metrics,
            "warnings": self.warnings,
            "checks": [c.as_dict() for c in self.checks],
            "tolerances": drafting_config()["tolerances"],
        }


def validate(doc: PatternDocument, cutting: dict[str, CuttingLine] | None = None) -> ValidationReport:
    tol = drafting_config()["tolerances"]
    cont = tol["endpoint_continuity_mm"]
    dup = tol["duplicate_point_mm"]

    checks: list[CheckResult] = []
    warnings: list[dict] = []
    metrics: dict[str, Any] = {}

    panels = doc.panels
    metrics["panelCount"] = len(panels)
    metrics["stitchCount"] = len(doc.stitches)

    # -- non-finite coordinates ---------------------------------------------
    bad: list[str] = []
    for panel in panels:
        for e in panel.edges:
            pts = e.polyline(drafting_config()["sample_spacing_mm"])
            for p in [e.start, e.end] + pts:
                if not (math.isfinite(p.x) and math.isfinite(p.y)):
                    bad.append(e.ensure_id())
                    break
    checks.append(
        CheckResult(
            "finite_coordinates",
            "fail" if bad else "pass",
            CRITICAL,
            f"{len(bad)} edge(s) carry a non-finite coordinate." if bad else "All coordinates are finite.",
            sorted(set(bad)),
            {"count": len(bad)},
            "Re-import the measurements; the draft cannot continue with non-finite values.",
        )
    )

    # -- boundary completeness and closure ----------------------------------
    for panel in panels:
        ids: list[str] = []
        issues: list[str] = []
        gaps: list[float] = []
        contour = panel.outer_contour()
        for i, edge in enumerate(panel.edges):
            edge_id = edge.ensure_id()
            if edge.length_mm <= dup:
                issues.append(f"zero-length edge {edge_id}")
                ids.append(edge_id)
            nxt = panel.edges[(i + 1) % len(panel.edges)]
            gap = edge.end.dist(nxt.start)
            gaps.append(gap)
            if gap > cont:
                issues.append(f"{edge_id} -> {nxt.ensure_id()} gap {gap:.3f} mm")
                ids.extend([edge_id, nxt.ensure_id()])
        if len(contour) < 3:
            issues.append(f"only {len(contour)} boundary vertices")
        checks.append(
            CheckResult(
                f"panel_closure:{panel.key}",
                "fail" if issues else "pass",
                CRITICAL,
                f"{panel.label}: " + ("; ".join(issues) if issues else
                                      f"{len(panel.edges)} edges close within {cont} mm."),
                ids,
                {"edges": len(panel.edges), "vertices": len(contour),
                 "maxEndpointGapMm": max(gaps) if gaps else 0.0},
                "Check the landmark rules for this panel.",
            )
        )
        metrics[f"panel:{panel.key}:maxEndpointGapMm"] = max(gaps) if gaps else 0.0
        metrics[f"panel:{panel.key}:areaMm2"] = _area(panel)

        # -- self-intersecting outer boundary -------------------------------
        ring = LinearRing(_flatten(panel))
        if not ring.is_simple:
            checks.append(
                CheckResult(
                    f"self_intersection:{panel.key}", "fail", CRITICAL,
                    f"{panel.label}: the outer boundary crosses itself.",
                    [panel.ensure_id()], {"selfIntersections": 1},
                    "Adjust the armscye depth, shoulder slope or ease and regenerate.",
                )
            )
        else:
            checks.append(
                CheckResult(
                    f"self_intersection:{panel.key}", "pass", CRITICAL,
                    f"{panel.label}: the outer boundary is simple.", [panel.ensure_id()],
                    {"selfIntersections": 0},
                )
            )

        # -- darts inside their own panel -----------------------------------
        for dart in panel.darts:
            outer = Polygon(_flatten(panel))
            outside = 0.0
            area = 0.0
            for leg in dart.contours:
                poly = Polygon([(p.x, p.y) for p in leg])
                if not poly.is_valid or poly.area == 0:
                    outside = float("inf")
                    break
                area += poly.area
                outside += poly.area - outer.intersection(poly).area
            ok = outside <= max(1e-6, area * 1e-6) + 1e-9
            checks.append(
                CheckResult(
                    f"dart_containment:{dart.id}", "pass" if ok else "fail", CRITICAL,
                    f"Dart {dart.id} lies inside {panel.label}." if ok
                    else f"Dart {dart.id} extends outside {panel.label}.",
                    [dart.id, panel.ensure_id()],
                    {"outsideAreaMm2": outside, "dartAreaMm2": area},
                    "Reduce the dart intake or widen the panel.",
                )
            )

    # -- duplicate identifiers ----------------------------------------------
    panel_ids = [p.ensure_id() for p in panels]
    edge_ids = [e.ensure_id() for p in panels for e in p.edges]
    dupes = _duplicates(panel_ids + edge_ids)
    checks.append(
        CheckResult(
            "unique_identifiers", "fail" if dupes else "pass", CRITICAL,
            f"Duplicate identifiers: {dupes}" if dupes else "All panel and edge ids are unique.",
            dupes, {"duplicates": len(dupes)}, "Regenerate the pattern.",
        )
    )

    # -- stitches ------------------------------------------------------------
    missing_refs: list[str] = []
    bad_params: list[str] = []
    for pair in doc.stitches:
        for rec in pair.records:
            if doc.edge_by_id(rec.edge_id) is None:
                missing_refs.append(rec.edge_id)
            if not (0.0 <= rec.param_start <= 1.0 and 0.0 <= rec.param_end <= 1.0):
                bad_params.append(rec.edge_id)
    checks.append(
        CheckResult(
            "stitch_references", "fail" if missing_refs else "pass", CRITICAL,
            f"Unresolved stitch edges: {missing_refs}" if missing_refs
            else "Every stitch references an existing edge.",
            missing_refs, {"count": len(missing_refs)}, "Regenerate the pattern.",
        )
    )
    checks.append(
        CheckResult(
            "stitch_parameters", "fail" if bad_params else "pass", CRITICAL,
            "Stitch parameters outside [0, 1]." if bad_params else "All stitch parameters are normalised.",
            bad_params, {"count": len(bad_params)}, "Regenerate the pattern.",
        )
    )

    # -- paired seam mismatch ------------------------------------------------
    mismatches = []
    for pair in doc.stitches:
        lengths = []
        for rec in pair.records:
            found = doc.edge_by_id(rec.edge_id)
            if found:
                lengths.append(found[1].length_mm * rec.length_param)
        if len(lengths) == 2:
            diff = abs(lengths[0] - lengths[1])
            allowed = max(tol["seam_mismatch_abs_mm"], tol["seam_mismatch_rel"] * max(lengths))
            mismatches.append((pair.edge_ids(), diff, allowed))
    bad_pairs = [m for m in mismatches if m[1] > m[2] + 1e-9]
    checks.append(
        CheckResult(
            "seam_mismatch", "fail" if bad_pairs else "pass", CRITICAL,
            "Paired seams differ by more than the tolerance." if bad_pairs
            else "Paired seam lengths agree within tolerance.",
            [e for ids, _, _ in bad_pairs for e in ids],
            {"pairs": [{"edges": ids, "deltaMm": d, "allowedMm": a} for ids, d, a in mismatches]},
            "Check that the front and back use the same shoulder seam rule.",
        )
    )
    for ids, diff, allowed in mismatches:
        metrics[f"seamMismatch:{ids[0][:8]}"] = round(diff, 4)

    # -- cutting lines -------------------------------------------------------
    if cutting:
        issues = []
        for key, line in cutting.items():
            if not line.closed or len(line.contour) < 3:
                issues.append(key)
            if any(not math.isfinite(p.x) or not math.isfinite(p.y) for p in line.contour):
                issues.append(key)
        metrics["cuttingContours"] = len(cutting)
        checks.append(
            CheckResult(
                "cutting_lines", "fail" if issues else "pass", CRITICAL,
                f"Invalid cutting contours for panels: {issues}" if issues
                else "Cutting contours are closed and finite.",
                issues, {"contours": len(cutting)},
                "Reduce the seam allowance or switch the join style.",
            )
        )

    failed_critical = [c for c in checks if c.status == "fail" and c.severity == CRITICAL]
    return ValidationReport(
        status="fail" if failed_critical else "pass",
        checks=checks,
        metrics=metrics,
        warnings=warnings,
    )


def _flatten(panel) -> list[tuple[float, float]]:
    """Boundary as a single point list, with each edge's last point dropped so the
    next edge's first point closes the seam exactly once."""
    spacing = drafting_config()["sample_spacing_mm"]
    pts = []
    for edge in panel.edges:
        poly = edge.polyline(spacing)
        pts.extend((p.x, p.y) for p in poly[:-1])
    return pts


def _area(panel) -> float:
    try:
        return round(Polygon(_flatten(panel)).area, 2)
    except Exception:  # pragma: no cover - degenerate geometry only
        return 0.0


def _duplicates(ids: list[str]) -> list[str]:
    seen, dupes = set(), []
    for i in ids:
        if i in seen:
            dupes.append(i)
        seen.add(i)
    return dupes
