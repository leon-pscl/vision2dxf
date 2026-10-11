"""HTTP contract: every documented endpoint, and structured errors."""
from __future__ import annotations

import io
import json
import zipfile

import pytest
from fastapi.testclient import TestClient

from app.main import app
from conftest import yaml_text

client = TestClient(app)


@pytest.fixture(scope="module")
def yaml_ok():
    return yaml_text("body_measurements.yaml")


@pytest.fixture(scope="module")
def yaml_pascual():
    return yaml_text("body_measurements_pascual.yaml")


@pytest.fixture(scope="module")
def generated(yaml_ok):
    response = client.post("/api/patterns/generate", json={"yamlText": yaml_ok})
    assert response.status_code == 200
    return response.json()


def test_measurements_validate_returns_every_value(yaml_ok):
    data = client.post("/api/measurements/validate",
                       json={"filename": "body.yaml", "content": yaml_ok}).json()
    assert data["ok"] and len(data["rows"]) == 26
    assert data["rows"][0]["interpretation"]


def test_measurements_validate_reports_field_errors():
    response = client.post("/api/measurements/validate",
                           json={"filename": "body.yaml", "content": "body:\n  bust: oops\n"})
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "invalid_measurements"
    assert any(d["field"] == "body.bust" for d in error["details"])


def test_generate_returns_pattern_trace_and_validation(generated):
    assert set(generated) >= {"pattern", "trace", "validation", "preview"}
    assert generated["validation"]["status"] == "pass"
    assert generated["trace"]["method"] == "formula-drafting"
    assert generated["trace"]["rules"]


def test_generate_produces_front_and_back_panels(generated):
    panels = generated["pattern"]["panels"]
    assert len(panels) == 2
    assert [p["label"] for p in panels] == ["Front bodice", "Back bodice"]


def test_generate_preview_has_svg_and_extent(generated):
    assert "<svg" in generated["preview"]["svg"]
    assert len(generated["preview"]["extent"]) == 4


def test_second_fixture_also_generates(yaml_pascual):
    data = client.post("/api/patterns/generate", json={"yamlText": yaml_pascual}).json()
    assert data["validation"]["status"] == "pass"


def test_generate_rejects_bad_yaml():
    response = client.post("/api/patterns/generate", json={"yamlText": "body: 3"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_measurements"


def test_validate_accepts_our_own_output(generated):
    response = client.post("/api/patterns/validate", json={"pattern": generated["pattern"]})
    assert response.status_code == 200
    data = response.json()
    assert data["ok"], data["issues"]
    assert data["panelCount"] == 2


def test_validate_flags_a_broken_endpoint(generated):
    pattern = json.loads(json.dumps(generated["pattern"]))
    seq = pattern["panels"][0]["seqEdges"][0]
    seq["edges"][0]["controlPoints"][-1][0] += 25.0
    data = client.post("/api/patterns/validate", json={"pattern": pattern}).json()
    assert not data["ok"]
    assert any(i["check"] == "endpoint_continuity" for i in data["issues"])


def test_validate_rejects_a_schema_violation(generated):
    pattern = json.loads(json.dumps(generated["pattern"]))
    del pattern["panels"][0]["particleDistance"]
    response = client.post("/api/patterns/validate", json={"pattern": pattern})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_pattern_schema"


def test_render_returns_the_same_geometry_as_generate(yaml_ok, generated):
    data = client.post("/api/patterns/render", json={"yamlText": yaml_ok}).json()
    assert data["svg"] == generated["preview"]["svg"]


def test_export_pdf(yaml_ok):
    response = client.post("/api/patterns/export/pdf", json={"yamlText": yaml_ok})
    assert response.status_code == 200
    assert response.content[:4] == b"%PDF"
    assert response.headers["content-type"] == "application/pdf"


def test_export_dxf(yaml_ok):
    response = client.post("/api/patterns/export/dxf", json={"yamlText": yaml_ok})
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/dxf"
    assert b"SECTION" in response.content


def test_export_bundle(yaml_ok):
    response = client.post("/api/patterns/export/bundle", json={"yamlText": yaml_ok})
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        assert {"pattern.json", "validation-report.json", "pattern-a0.pdf"} <= set(zf.namelist())
        report = json.loads(zf.read("validation-report.json"))
        assert report["status"] == "pass"
        assert report["tolerances"]["endpoint_continuity_mm"] == 0.1


def test_config_endpoint(yaml_ok):
    data = client.get("/api/config").json()
    assert "fitted" in data["easePresets"]
    assert data["paper"]["default_size"] == "A0"
    assert "cutting" in data["viewLayers"]


def test_healthz():
    assert client.get("/healthz").json() == {"status": "ok"}
