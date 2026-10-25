"""Placement and A0 nesting smoke tests."""
from __future__ import annotations

from app.placement import place_document, validate_placement
from app.drafting.geometry import PatternPanel, PatternEdge, Point2D


def _panel(key, w, h):
    a = Point2D(0.0, 0.0)
    b = Point2D(w, 0.0)
    c = Point2D(w, h)
    d = Point2D(0.0, h)
    edges = [
        PatternEdge(role="r0", start=a, end=b, label=f"{key}.e0"),
        PatternEdge(role="r1", start=b, end=c, label=f"{key}.e1"),
        PatternEdge(role="r2", start=c, end=d, label=f"{key}.e2"),
        PatternEdge(role="r3", start=d, end=a, label=f"{key}.e3"),
    ]
    return PatternPanel(key=key, label=key, edges=edges)


def test_a_single_panel_fits_on_a0():
    from app.drafting.geometry import PatternDocument
    doc = PatternDocument(panels=[_panel("front", 500.0, 400.0)], stitches=[])
    report = place_document(doc, paper_size="A0")
    assert report.panel_count == 1
    assert validate_placement(report) == []
    assert report.efficiency > 0


def test_two_small_panels_do_not_overlap():
    from app.drafting.geometry import PatternDocument
    doc = PatternDocument(panels=[_panel("a", 200.0, 200.0), _panel("b", 200.0, 200.0)], stitches=[])
    report = place_document(doc, paper_size="A0")
    assert report.panel_count == 2
    assert validate_placement(report) == []


def test_paper_dimensions_are_a0_by_default():
    from app.drafting.geometry import PatternDocument
    doc = PatternDocument(panels=[_panel("a", 10.0, 10.0)], stitches=[])
    report = place_document(doc)
    assert report.paper_width == 1189.0
    assert report.paper_height == 841.0
    assert report.margin > 0
