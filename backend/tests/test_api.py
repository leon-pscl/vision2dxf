"""HTTP contract: the semantics gate, generation, exports, structured errors."""
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
def confirmations(default_confirmations):
    return default_confirmations


@pytest.fixture(scope="module")
def generated(yaml_ok, confirmations):
    response = client.post("/api/patterns/generate",
                           json={"yamlText": yaml_ok, "confirmations": confirmations})
    assert response.status_code == 200
    return response.json()


# ---- step 1: syntax only ----------------------------------------------------

def test_validate_measurements_returns_every_value(yaml_ok):
    data = client.post("/api/measurements/validate",
                       json={"filename": "body.yaml", "content": yaml_ok}).json()
    assert data["ok"] and len(data["rows"]) == 26
    assert data["rows"][0]["interpretation"]


def test_validate_measurements_reports_field_errors():
    response = client.post("/api/measurements/validate",
                           json={"filename": "body.yaml", "content": "body:\n  bust: oops\n"})
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "invalid_yaml"
    assert any(d["field"] == "body.bust" for d in error["details"])


# ---- step 2: semantics -----------------------------------------------------

def test_semantics_schema_endpoint():
    data = client.get("/api/measurements/semantics/schema").json()
    assert data["schemaVersion"]
    assert len(data["fields"]) == 26
    assert data["statuses"][0] == "CONFIRMED"
    assert set(data["blockingStatuses"]) == {"MISSING", "AMBIGUOUS", "INVALID"}


def test_semantics_resolution_blocks_without_confirmation(yaml_ok):
    data = client.post("/api/measurements/semantics", json={"yamlText": yaml_ok}).json()
    assert data["source"]["canGenerate"] is False
    assert data["draftingMeasurements"] is None
    ambiguous = [f for f in data["source"]["fields"] if f["status"] == "AMBIGUOUS"]
    assert ambiguous and all(f["suggestedAction"] == "Confirm interpretation" for f in ambiguous)


def test_semantics_resolution_after_confirmation(yaml_ok, confirmations):
    data = client.post("/api/measurements/semantics",
                       json={"yamlText": yaml_ok, "confirmations": confirmations}).json()
    assert data["source"]["canGenerate"] is True
    assert data["draftingMeasurements"]["shoulder_width_full"] == 342.0
    assert data["draftingMeasurements"]["shoulder_slope_degrees"] == 45.0


def test_semantics_records_original_and_normalized(yaml_ok, confirmations):
    data = client.post("/api/measurements/semantics", json={
        "yamlText": yaml_ok, "confirmations": confirmations, "corrections": {"bust": 86.0}}).json()
    bust = next(f for f in data["source"]["fields"] if f["name"] == "bust")
    assert bust["originalValue"] == 85.8
    assert bust["sourceValue"] == 86.0
    assert bust["normalizedMm"] == 860.0
    assert bust["conversion"].startswith("cm -> mm")
    assert bust["confirmedBy"] == "schema"
    assert bust["definitionVersion"]


def test_mixed_origins_are_a_blocking_schema_message(yaml_ok, confirmations):
    response = client.post("/api/measurements/semantics", json={
        "yamlText": yaml_ok,
        "confirmations": {**confirmations, "waist_line": "level_from_top_of_head"},
    })
    data = response.json()
    assert data["source"]["canGenerate"] is False
    assert any(b["severity"] == "blocking" for b in data["source"]["blocking"])


def test_measurement_report_download(yaml_ok, confirmations):
    response = client.post("/api/measurements/report",
                           json={"yamlText": yaml_ok, "confirmations": confirmations})
    assert response.headers["content-type"].startswith("application/json")
    assert "measurement-report.json" in response.headers["content-disposition"]
    assert json.loads(response.content)["can_generate"] is True


# ---- step 5: the gate ------------------------------------------------------

def test_generate_is_blocked_until_fields_are_confirmed(yaml_ok):
    response = client.post("/api/patterns/generate", json={"yamlText": yaml_ok})
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "semantics_unresolved"
    assert any(d["code"] == "AMBIGUOUS" for d in error["details"])
    assert response.json()["semantics"]["canGenerate"] is False


def test_generate_is_blocked_for_an_invalid_value(yaml_ok, confirmations):
    response = client.post("/api/patterns/generate", json={
        "yamlText": yaml_ok, "confirmations": confirmations, "corrections": {"bust": 900.0}})
    assert response.status_code == 409
    assert any(d["code"] == "INVALID" for d in response.json()["error"]["details"])


def test_second_profile_is_rejected_by_the_schema(yaml_pascual, confirmations):
    response = client.post("/api/patterns/generate",
                           json={"yamlText": yaml_pascual, "confirmations": confirmations})
    assert response.status_code == 409
    assert any(d["code"] == "INVALID" for d in response.json()["error"]["details"])


# ---- step 6: generation ----------------------------------------------------

def test_generate_returns_pattern_trace_and_validation(generated):
    assert set(generated) >= {"pattern", "trace", "validation", "preview"}
    assert generated["validation"]["status"] == "pass"
    assert generated["trace"]["method"] == "formula-drafting"
    assert generated["trace"]["rules"]


def test_manifest_records_the_semantics_decision(generated):
    trace = generated["trace"]
    assert trace["measurement_schema_version"]
    assert trace["definition_version"]
    assert trace["generated_at"]
    assert trace["software_version"]
    assert trace["units"] == {"source": "cm", "pattern": "mm"}
    assert trace["coordinate_convention"]["vertical_origin"]
    interp = trace["field_interpretations"]["vert_bust_line"]
    assert interp["meaning"] == "neck_to_bust_apex_vertical"
    assert interp["status"] == "assumed"
    assert interp["confirmed_by"] == "operator"
    assert "bust_to_waist_vertical" in trace["derived_measurements"]
    assert trace["drafting_measurements_mm"]["bust_to_waist_vertical"] == 117.0


def test_generate_produces_front_and_back_panels(generated):
    panels = generated["pattern"]["panels"]
    assert len(panels) == 2
    assert [p["label"] for p in panels] == ["Front bodice", "Back bodice"]


def test_generate_preview_has_svg_and_extent(generated):
    assert "<svg" in generated["preview"]["svg"]
    assert len(generated["preview"]["extent"]) == 4


def test_drafting_output_changes_with_a_different_interpretation(yaml_ok, confirmations):
    half = client.post("/api/patterns/generate",
                       json={"yamlText": yaml_ok, "confirmations": confirmations}).json()
    full = client.post("/api/patterns/generate", json={
        "yamlText": yaml_ok,
        "confirmations": {**confirmations, "shoulder_w": "full_across_shoulder_tips"},
    }).json()
    assert half["pattern"] != full["pattern"]
    assert full["trace"]["field_interpretations"]["shoulder_w"]["candidate"] == \
        "full_across_shoulder_tips"


# ---- step 7/8: validation and exports --------------------------------------

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


def test_render_returns_the_same_geometry_as_generate(yaml_ok, confirmations, generated):
    data = client.post("/api/patterns/render",
                       json={"yamlText": yaml_ok, "confirmations": confirmations}).json()
    assert data["svg"] == generated["preview"]["svg"]


@pytest.mark.parametrize("path,name,magic", [
    ("/api/patterns/export/pdf", "pdf", b"%PDF"),
    ("/api/patterns/export/dxf", "dxf", b"SECTION"),
    ("/api/patterns/export/svg", "svg", b"<svg"),
])
def test_exports(yaml_ok, confirmations, path, name, magic):
    response = client.post(path, json={"yamlText": yaml_ok, "confirmations": confirmations})
    assert response.status_code == 200
    assert magic in response.content
    assert name in response.headers["content-disposition"]


def test_exports_are_blocked_when_semantics_are_unresolved(yaml_ok):
    for path in ("/api/patterns/export/pdf", "/api/patterns/export/dxf", "/api/patterns/export/bundle"):
        response = client.post(path, json={"yamlText": yaml_ok})
        assert response.status_code == 409, path


def test_export_bundle(yaml_ok, confirmations):
    response = client.post("/api/patterns/export/bundle",
                           json={"yamlText": yaml_ok, "confirmations": confirmations})
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        assert {"pattern.json", "validation-report.json", "pattern-a0.pdf",
                "generation-manifest.json"} <= set(zf.namelist())
        report = json.loads(zf.read("validation-report.json"))
        assert report["status"] == "pass"
        manifest_data = json.loads(zf.read("generation-manifest.json"))
        assert manifest_data["measurement_schema_version"]


def test_config_endpoint():
    data = client.get("/api/config").json()
    assert "fitted" in data["easePresets"]
    assert data["paper"]["default_size"] == "A0"
    assert "cutting" in data["viewLayers"]
    assert data["measurementSchema"]["schemaVersion"]


def test_healthz():
    assert client.get("/healthz").json() == {"status": "ok"}
