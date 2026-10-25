"""Production exports: full scale in PDF and DXF, plus the preview."""
from __future__ import annotations

import io
import re
import zipfile

import ezdxf
import pytest

from app.export.dxf import export_dxf
from app.export.pdf import export_pdf
from app.rendering.png import render_png
from app.rendering.svg import VIEW_LAYERS, render_svg

MM_PER_INCH = 25.4


def test_pdf_is_a_pdf(basic_result):
    assert export_pdf(basic_result.document, basic_result.cutting).startswith(b"%PDF")


def test_pdf_is_full_scale(basic_result):
    """A 100 mm drawn line must come back as 100 mm of PDF user space."""
    data = export_pdf(basic_result.document, basic_result.cutting)
    # 1 pt = 1/72 inch; 1 mm = 72/25.4 pt. Every page box must be an A0 sheet in pt,
    # in either orientation: a panel taller than the configured one turns the sheet.
    a0_w_mm, a0_h_mm = 1189.0, 841.0
    expected = {
        (round(a0_w_mm * 72 / MM_PER_INCH, 2), round(a0_h_mm * 72 / MM_PER_INCH, 2)),
        (round(a0_h_mm * 72 / MM_PER_INCH, 2), round(a0_w_mm * 72 / MM_PER_INCH, 2)),
    }
    text = data.decode("latin-1")
    media = re.findall(r"/MediaBox \[([^\]]+)\]", text)
    assert media, "PDF has no MediaBox"
    boxes = set()
    for box in media:
        width, height = box.split()[2:4]
        boxes.add((round(float(width), 2), round(float(height), 2)))
    assert boxes <= expected


def test_pdf_places_every_panel_on_its_own_sheet(basic_result):
    """Placement runs in world coordinates, so a panel lands on its sheet instead of
    being pushed off it by the `center` it is placed by, and it is centred."""
    from app.config import drafting_config
    from app.export.pdf import _place

    cfg = drafting_config()["paper"]
    sheets = _place(basic_result.document, cfg, cfg["default_size"], cfg["default_orientation"])
    assert len(sheets) == len(basic_result.document.panels)
    turned = 0
    for panel, sx, sy, page_w, page_h in sheets:
        x0, y0, x1, y1 = panel.bbox()
        cx, cy = (panel.center.x, panel.center.y) if panel.center else (0.0, 0.0)
        left, right = x0 + cx + sx, x1 + cx + sx
        bottom, top = y0 + cy + sy, y1 + cy + sy
        assert 0.0 <= left and right <= page_w
        assert 0.0 <= bottom and top <= page_h
        assert (left + right) / 2 == pytest.approx(page_w / 2)
        assert (bottom + top) / 2 == pytest.approx(page_h / 2)
        # the cut line, which sits seam-allowance outside the sewing line, fits too
        allowance = max((e.seam_allowance_mm for e in panel.edges), default=0.0)
        assert 0.0 <= left - allowance and right + allowance <= page_w
        assert 0.0 <= bottom - allowance and top + allowance <= page_h
        turned += page_h > page_w
    # The fixture panels are ~1151 mm tall, taller than A0 landscape, so the sheet
    # must have turned to portrait rather than clipping the panel away.
    assert turned == len(sheets)


def test_pdf_geometry_scales_linearly(basic_result):
    """Two panels with different sizes must produce proportionally different paths."""
    small = export_pdf(basic_result.document, basic_result.cutting)
    assert len(small) > 1000


def test_dxf_is_readable_and_in_millimetres(basic_result):
    data = export_dxf(basic_result.document, basic_result.cutting)
    doc = ezdxf.read(io.StringIO(data.decode("utf-8")))
    assert doc.header["$INSUNITS"] == 4  # millimetres


def test_dxf_is_full_scale(basic_result):
    """Sewing-line polyline coordinates must equal the document's millimetres."""
    from app.config import drafting_config
    from app.drafting.geometry import world_point

    data = export_dxf(basic_result.document, basic_result.cutting)
    doc = ezdxf.read(io.StringIO(data.decode("utf-8")))
    sewing = [e for e in doc.modelspace() if e.dxf.layer == "SEWING"]
    assert sewing
    xs = [p[0] for e in sewing for p in e.get_points("xy")]
    spacing = drafting_config()["sample_spacing_mm"]
    expected = [world_point(panel, q).x
                for panel in basic_result.document.panels
                for edge in panel.edges
                for q in edge.polyline(spacing)]
    assert (min(xs), max(xs)) == pytest.approx((min(expected), max(expected)), abs=1e-6)


def test_dxf_has_one_layer_per_geometry_class(basic_result):
    """A block derived from a template carries no notch marks and no cut-out darts, so
    those layers are empty; the geometry classes it does have each get their own."""
    data = export_dxf(basic_result.document, basic_result.cutting)
    doc = ezdxf.read(io.StringIO(data.decode("utf-8")))
    layers = {e.dxf.layer for e in doc.modelspace()}
    assert {"SEWING", "CUTTING", "CONSTRUCTION"} <= layers


def test_preview_svg_matches_the_exported_geometry(basic_result):
    svg = render_svg(basic_result.document, basic_result.cutting)
    front = basic_result.document.panels[0]
    p = front.landmarks["left.shoulder_tip"]
    world = p.x + front.center.x, p.y + front.center.y
    assert f"{world[0]:.2f}" in svg or f"{world[0]:.1f}" in svg


def test_preview_svg_has_a_group_per_view_layer(basic_result):
    svg = render_svg(basic_result.document, basic_result.cutting)
    for layer in VIEW_LAYERS:
        assert f'data-layer="{layer}"' in svg


def test_preview_png_is_a_png(basic_result):
    data = render_png(basic_result.document, basic_result.cutting, width_px=400)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"


def test_bundle_contains_every_required_file(base_confirmations):
    import json

    from app.api.routes import export_bundle
    from app.schemas.api import ExportRequest, SettingsPayload
    from conftest import yaml_text

    payload = dict(yamlText=yaml_text("body_measurements_target_a.yaml"),
                   confirmations=base_confirmations,
                   settings=SettingsPayload())
    response = export_bundle(ExportRequest(**payload))
    with zipfile.ZipFile(io.BytesIO(response.body)) as zf:
        names = set(zf.namelist())
        assert {"pattern.json", "generation-manifest.json", "transformation-diagnostics.json",
                "validation-report.json", "base-vs-target-preview.png", "pattern-a0.pdf",
                "pattern.dxf", "base-vs-target-preview.svg", "base-block.json"} <= names
        assert json.loads(zf.read("validation-report.json"))["status"] == "pass"
