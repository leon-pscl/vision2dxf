"""HTTP contract for surface reconstruction, flattening, validation, and exports."""
from __future__ import annotations

import json
import io
import zipfile
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.api import routes
from conftest import yaml_text

client = TestClient(app)


@pytest.fixture(scope="module")
def yaml_basic():
    return yaml_text("body_measurements.yaml")


@pytest.fixture(scope="module")
def confirmations(default_confirmations):
    return default_confirmations


@pytest.fixture(scope="module")
def generate_payload(yaml_basic, confirmations):
    return {
        "yamlText": yaml_basic,
        "filename": "body_measurements.yaml",
        "confirmations": confirmations,
        "settings": {"confirm_units": True},
    }


@pytest.fixture(scope="module")
def generated(generate_payload):
    response = client.post("/api/patterns/flatten", json={
        **generate_payload,
        "include_mesh": True,
        "include_surfaces": True,
    })
    assert response.status_code == 200, response.text
    return response.json()


def test_measurement_upload_reports_all_input_rows(yaml_basic):
    response = client.post("/api/measurements/validate", json={
        "filename": "body_measurements.yaml",
        "content": yaml_basic,
    })
    assert response.status_code == 200
    data = response.json()
    assert data["ok"]
    assert len(data["rows"]) == 26
    assert all("interpretation" in row for row in data["rows"])


def test_invalid_yaml_is_a_structured_error():
    response = client.post("/api/measurements/validate", json={
        "filename": "body_measurements.yaml",
        "content": "body: [invalid",
    })
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_yaml"
    assert "Traceback" not in response.text


def test_semantics_require_confirmation(yaml_basic):
    response = client.post("/api/measurements/semantics", json={"yamlText": yaml_basic})
    assert response.status_code == 200
    data = response.json()
    assert data["source"]["canGenerate"] is False
    assert data["draftingMeasurements"] is None
    assert data["source"]["blocking"]


def test_generation_returns_surface_derived_pattern_and_diagnostics(generated):
    assert generated["trace"]["method"] == "anthropometric surface reconstruction and mesh flattening"
    assert len(generated["pattern"]["panels"]) == 2
    assert generated["validation"]["status"] == "pass"
    assert generated["diagnostics"]["method"] == "lscm_arap"
    assert generated["diagnostics"]["maxFlippedTriangles"] == 0
    assert generated["flattenedMesh"]["panels"]
    assert generated["crossSections"]["sections"]
    assert generated["surfaceStages"]["proxy"]["faces"]
    assert generated["surfaceStages"]["garment"]["faces"]
    assert len(generated["surfaceStages"]["panels"]) == 2
    assert generated["assumptions"]
    assert all("base_block" not in key for key in generated["trace"])


def test_generation_is_blocked_until_unit_confirmation(yaml_basic, confirmations):
    response = client.post("/api/patterns/flatten", json={
        "yamlText": yaml_basic,
        "confirmations": confirmations,
    })
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "units_unconfirmed"


def test_surface_stage_endpoints_expose_intermediate_geometry(generate_payload):
    cross_sections = client.post("/api/reconstruction/cross-sections", json=generate_payload)
    assert cross_sections.status_code == 200
    assert cross_sections.json()["sections"]

    proxy = client.post("/api/reconstruction/proxy-surface", json=generate_payload)
    assert proxy.status_code == 200
    assert proxy.json()["mesh"]["faces"]

    garment = client.post("/api/garment/surface", json=generate_payload)
    assert garment.status_code == 200
    assert garment.json()["diagnostics"]["levels"]

    segmented = client.post("/api/garment/segment", json=generate_payload)
    assert segmented.status_code == 200
    assert len(segmented.json()["panels"]) == 2
    assert segmented.json()["seamPaths"]


def test_pattern_validation_accepts_generated_output(generated):
    response = client.post("/api/patterns/validate", json={"pattern": generated["pattern"]})
    assert response.status_code == 200
    assert response.json()["ok"]
    assert response.json()["panelCount"] == 2


def test_render_returns_svg_and_distortion_heat_maps(generate_payload):
    response = client.post("/api/patterns/render", json=generate_payload)
    assert response.status_code == 200
    assert "<svg" in response.json()["svg"]
    assert response.json()["heatMaps"]


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/api/patterns/export/pdf", b"%PDF"),
        ("/api/patterns/export/dxf", b"SECTION"),
        ("/api/patterns/export/svg", b"<svg"),
    ],
)
def test_production_export_endpoints(generate_payload, path, expected):
    response = client.post(path, json=generate_payload)
    assert response.status_code == 200, response.text
    assert expected in response.content


def test_production_export_is_blocked_by_critical_validation_failure(monkeypatch):
    report = {
        "status": "fail",
        "summary": {"criticalFailures": 1},
        "checks": [{
            "id": "panel_closure:front",
            "status": "fail",
            "severity": "critical",
            "message": "The front panel boundary is open.",
            "suggestedAction": "Review the reconstructed panel.",
        }],
    }
    failed_result = SimpleNamespace(
        report=SimpleNamespace(status="fail", as_dict=lambda: report),
    )
    monkeypatch.setattr(routes, "_run", lambda _request: (failed_result, None))
    request = routes.ExportRequest(yamlText="body: {}", settings={"confirm_units": True})

    response = routes.export_pdf_route(request)
    assert response.status_code == 409
    content = json.loads(response.body)
    assert content["error"]["code"] == "validation_failed"
    assert content["stage"] == "validation"
    assert content["validation"]["summary"]["criticalFailures"] == 1


def test_export_bundle_contains_pattern_manifest_and_debug_meshes(generate_payload):
    response = client.post("/api/patterns/export/bundle", json=generate_payload)
    assert response.status_code == 200, response.text
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        names = set(archive.namelist())
        assert {
            "pattern.json",
            "generation-manifest.json",
            "flattening-diagnostics.json",
            "validation-report.json",
            "pattern-preview.svg",
            "pattern-preview.png",
            "pattern-a0.pdf",
            "pattern.dxf",
            "debug/proxy-mesh.obj",
            "debug/garment-mesh.obj",
            "debug/flattened-mesh.json",
        } <= names
        manifest = json.loads(archive.read("generation-manifest.json"))
        assert manifest["method"] == "anthropometric surface reconstruction and mesh flattening"
        assert manifest["assumptions"]
        assert json.loads(archive.read("validation-report.json"))["status"] == "pass"


def test_config_and_health_endpoints():
    config = client.get("/api/config")
    assert config.status_code == 200
    assert config.json()["flattening"]["method"] == "lscm_arap"
    assert "fitted" in config.json()["easePresets"]
    assert client.get("/healthz").json() == {"status": "ok"}


def test_unknown_api_paths_are_structured_404s():
    response = client.get("/api/does-not-exist")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"
