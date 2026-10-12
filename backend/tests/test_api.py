"""HTTP contract: the semantics gate, the envelope gate, transformation, exports, errors."""
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
def yaml_target():
    return yaml_text("body_measurements_target_a.yaml")


@pytest.fixture(scope="module")
def yaml_pascual():
    return yaml_text("body_measurements_pascual.yaml")


@pytest.fixture(scope="module")
def confirmations(base_confirmations):
    """The interpretations the base block was measured with."""
    return base_confirmations


@pytest.fixture(scope="module")
def generated(yaml_target, confirmations):
    response = client.post("/api/patterns/transform",
                           json={"yamlText": yaml_target, "confirmations": confirmations})
    assert response.status_code == 200, response.text
    return response.json()


# ---- step 1: syntax only ----------------------------------------------------

def test_validate_measurements_returns_every_value(yaml_ok):
    data = client.post("/api/measurements/validate",
                       json={"filename": "body.yaml", "content": yaml_ok}).json()
    assert data["ok"] and len(data["rows"]) == 26
    assert data["rows"][0]["interpretation"]


def test_both_supplied_fixtures_load_and_display(yaml_ok, yaml_pascual):
    """Both supplied profiles import and show every value, whatever happens later."""
    for content in (yaml_ok, yaml_pascual):
        data = client.post("/api/measurements/validate",
                           json={"filename": "body.yaml", "content": content}).json()
        assert data["ok"] and len(data["rows"]) == 26
        assert all(row["valueMm"] == pytest.approx(row["value"] * 10.0) for row in data["rows"])


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


def test_the_second_fixture_displays_its_blocking_reason(yaml_pascual, confirmations):
    """It loads and displays; it cannot be transformed, and the screen says why."""
    data = client.post("/api/measurements/semantics",
                       json={"yamlText": yaml_pascual, "confirmations": confirmations}).json()
    assert data["source"]["canGenerate"] is False
    fields = {b["field"] for b in data["source"]["blocking"]}
    assert {"bust_line", "waist_line"} <= fields
    assert all(b["message"] and b["suggestedAction"] for b in data["source"]["blocking"])


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


# ---- the base block --------------------------------------------------------

def test_the_catalog_lists_the_blocks_on_offer():
    data = client.get("/api/base-blocks").json()
    ids = [b["id"] for b in data["blocks"]]
    assert ids == ["bodice-ref-a", "bodice-ref-b", "bodice-ref-c"]
    assert data["default"] in ids
    assert data["reference"]["profile"]
    assert all(b["validationStatus"] == "pass" for b in data["blocks"])
    assert data["templateSelection"]["supported"] is True


def test_base_block_endpoint_describes_one_block():
    data = client.get("/api/base-blocks/bodice-ref-a").json()
    assert data["id"] == "bodice-ref-a"
    assert data["version"] and data["sourceFile"].endswith(".json")
    assert data["sourceSha256"] and data["reviewed"] is True
    assert data["validationStatus"] == "pass"
    assert [p["key"] for p in data["panels"]] == ["front", "back"]
    assert "<svg" in data["previewSvg"]
    assert data["referenceMeasurementsMm"]["bust_circumference"] == 858.0
    assert data["garmentDimensionsMm"]["bust_girth"] > 0
    assert "Modifications" in data["provenance"]


def test_base_block_publishes_its_derived_maps():
    maps = client.get("/api/base-blocks/bodice-ref-a").json()["maps"]
    assert set(maps) == {"landmarks", "edges", "seams", "darts", "grainlines"}
    assert maps["landmarks"]["panels"]["front"]
    assert maps["edges"]["panels"]["front"]["roles"][0]
    assert maps["seams"]["pairs"]
    assert maps["grainlines"]["panels"]["front"]["fromMm"]


def test_any_template_can_be_selected_at_runtime():
    source = client.get("/api/base-blocks/bodice-ref-b").json()["sourceFile"]
    data = client.get("/api/base-blocks/template:" + source).json()
    assert data["id"] == "template:" + source
    assert data["reviewed"] is False
    assert any(n["code"] == "block_not_reviewed" for n in data["notes"])


def test_an_unknown_block_is_a_structured_404():
    response = client.get("/api/base-blocks/nope")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "base_block_unavailable"


def test_base_block_declares_its_semantic_vocabulary():
    vocabulary = client.get("/api/base-blocks/bodice-ref-a").json()["semanticMap"]
    assert "shoulder_tip" in vocabulary["pairedLandmarks"]
    assert "center_neck" in vocabulary["centreLandmarks"]
    assert {"hip_center", "hip_side", "dart_apex"} <= set(vocabulary["conditionalLandmarks"])
    assert "side_seam" in vocabulary["pairedEdgeRoles"]
    assert vocabulary["sides"] == ["left", "right"]


@pytest.mark.parametrize("block_id", ["bodice-ref-a", "bodice-ref-b", "bodice-ref-c"])
def test_base_block_validate_endpoint_runs_the_shared_validator(block_id):
    data = client.post("/api/base-blocks/" + block_id + "/validate").json()
    assert data["ok"] is True
    assert data["status"] == "pass"
    assert any(c["id"].startswith("base_landmark_roles") for c in data["checks"])
    assert any(c["id"].startswith("self_intersection") for c in data["checks"])


def test_deltas_endpoint_reports_differences_and_the_envelope(yaml_target, confirmations):
    data = client.post("/api/base-block/deltas",
                       json={"yamlText": yaml_target, "confirmations": confirmations}).json()
    bust = next(d for d in data["deltas"] if d["name"] == "bust_circumference")
    assert bust["baseValue"] == 858.0
    assert bust["targetValue"] == 940.0
    assert bust["deltaValue"] == pytest.approx(82.0)
    assert bust["mapped"] is True
    assert data["envelope"]["status"] == "ok"
    assert data["influenceMap"]["version"]
    assert data["referenceProfile"]["sourceFile"]
    assert data["garmentDimensionsMm"]["bust_girth"] > 0
    assert data["measurementDefinitions"]["bust_girth"]


def test_deltas_endpoint_is_blocked_before_confirmation(yaml_target):
    response = client.post("/api/base-block/deltas", json={"yamlText": yaml_target})
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "semantics_unresolved"


# ---- the gates -------------------------------------------------------------

def test_transform_is_blocked_until_fields_are_confirmed(yaml_ok):
    response = client.post("/api/patterns/transform", json={"yamlText": yaml_ok})
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "semantics_unresolved"
    assert any(d["code"] == "AMBIGUOUS" for d in error["details"])
    assert response.json()["semantics"]["canGenerate"] is False


def test_transform_is_blocked_for_an_invalid_value(yaml_ok, confirmations):
    response = client.post("/api/patterns/transform", json={
        "yamlText": yaml_ok, "confirmations": confirmations, "corrections": {"bust": 900.0}})
    assert response.status_code == 409
    assert any(d["code"] == "INVALID" for d in response.json()["error"]["details"])


def test_transform_is_blocked_outside_the_envelope(yaml_ok, confirmations):
    """A bust 25 cm from the base profile is past the tested range."""
    response = client.post("/api/patterns/transform", json={
        "yamlText": yaml_ok, "confirmations": confirmations, "corrections": {"bust": 111.0}})
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] in ("envelope_confirmation_required", "envelope_exceeded")
    assert any(d["field"] == "bust_circumference" for d in error["details"])


def test_an_envelope_warning_proceeds_once_confirmed(yaml_ok, confirmations):
    payload = {"yamlText": yaml_ok, "confirmations": confirmations,
               "corrections": {"bust": 101.0}}
    assert client.post("/api/patterns/transform", json=payload).status_code == 409
    confirmed = {**payload, "settings": {"confirm_envelope": True}}
    response = client.post("/api/patterns/transform", json=confirmed)
    assert response.status_code == 200, response.text
    assert response.json()["diagnostics"]["envelope"]["status"] == "warn"
    assert response.json()["diagnostics"]["envelope"]["confirmed"] is True


def test_the_second_profile_is_refused_with_its_reason(yaml_pascual, confirmations):
    response = client.post("/api/patterns/transform",
                           json={"yamlText": yaml_pascual, "confirmations": confirmations})
    assert response.status_code == 409
    assert any(d["code"] == "INVALID" for d in response.json()["error"]["details"])


# ---- transformation --------------------------------------------------------

def test_transform_returns_pattern_trace_diagnostics_and_validation(generated):
    assert set(generated) >= {"pattern", "trace", "diagnostics", "validation", "preview"}
    assert generated["validation"]["status"] == "pass"
    assert generated["trace"]["method"] == "base-block-transformation"


def test_the_manifest_identifies_the_base_block_and_configuration(generated):
    trace = generated["trace"]
    assert trace["base_block"]["id"] == "bodice-ref-a"
    assert trace["base_block"]["version"]
    assert trace["base_block"]["source_sha256"]
    assert trace["base_block"]["reviewed"] is True
    assert trace["base_block"]["reference_measurements_mm"]["bust_circumference"] == 858.0
    assert trace["base_block"]["measurement_definitions"]["bust_girth"]
    assert trace["transformation_config"]["version"]
    assert trace["transformation_config"]["influence_map_version"]
    assert trace["transformation_config"]["objective"]["w_landmark"] == 1.0
    assert trace["units"] == {"source": "cm", "pattern": "mm"}
    assert trace["generated_at"] and trace["software_version"]


def test_the_response_carries_deltas_diagnostics_and_achieved_dimensions(generated):
    diagnostics = generated["diagnostics"]
    assert diagnostics["solver"]["front"]["converged"] is True
    assert diagnostics["solver"]["front"]["method"] == "thin_plate_spline"
    assert any(d["name"] == "bust_circumference" for d in diagnostics["deltas"])
    assert diagnostics["achievedDimensions"]["bust_girth"] > 0
    assert diagnostics["requestedDimensions"]["bust_girth"]["requestedMm"] > 0
    assert diagnostics["seamMismatchBeforeBalancing"]
    assert diagnostics["seamMismatchAfterMm"]
    assert diagnostics["seamBalanceRounds"]
    assert diagnostics["landmarkTargets"][0]["displacementMm"] > 0


def test_transform_preserves_panels_and_identifiers(generated):
    panels = generated["pattern"]["panels"]
    assert len(panels) == 2
    assert [p["label"] for p in panels] == ["Front bodice", "Back bodice"]
    block_id = generated["trace"]["base_block"]["id"]
    base = client.get("/api/base-blocks/" + block_id).json()
    assert [p["id"] for p in panels] == [p["id"] for p in base["panels"]]


def test_each_block_can_be_transformed_through_the_api(yaml_target, confirmations):
    payloads = set()
    for block_id in ("bodice-ref-a", "bodice-ref-b", "bodice-ref-c"):
        response = client.post("/api/patterns/transform", json={
            "yamlText": yaml_target, "confirmations": confirmations,
            "settings": {"base_block_id": block_id}})
        assert response.status_code == 200, (block_id, response.text)
        data = response.json()
        assert data["validation"]["status"] == "pass"
        assert data["trace"]["base_block"]["id"] == block_id
        payloads.add(json.dumps(data["pattern"]))
    assert len(payloads) == 3


def test_the_preview_carries_both_blocks_for_comparison(generated):
    preview = generated["preview"]
    assert "<svg" in preview["svg"]
    assert 'data-layer="base"' in preview["svg"]
    assert 'data-layer="displacement"' in preview["svg"]
    assert 'data-layer="heatmap"' in preview["svg"]
    assert len(preview["extent"]) == 4
    assert [p["key"] for p in preview["basePanels"]] == ["front", "back"]
    assert preview["displacements"]


def test_a_zero_delta_target_reproduces_the_base_block_over_http(yaml_ok, confirmations):
    response = client.post("/api/patterns/transform",
                           json={"yamlText": yaml_ok, "confirmations": confirmations})
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["diagnostics"]["landmarkTargets"][0]["displacementMm"] == 0.0
    assert all(abs(d["deltaValue"]) < 1e-6 for d in data["diagnostics"]["deltas"])


def test_a_different_target_moves_the_geometry(yaml_ok, yaml_target, confirmations):
    base = client.post("/api/patterns/transform",
                       json={"yamlText": yaml_ok, "confirmations": confirmations}).json()
    other = client.post("/api/patterns/transform",
                        json={"yamlText": yaml_target, "confirmations": confirmations}).json()
    assert base["pattern"] != other["pattern"]
    assert other["diagnostics"]["landmarkTargets"][0]["displacementMm"] > 1.0


# ---- validation and exports ------------------------------------------------

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


def test_render_returns_the_same_geometry_as_transform(yaml_target, confirmations, generated):
    data = client.post("/api/patterns/render",
                       json={"yamlText": yaml_target, "confirmations": confirmations}).json()
    assert data["svg"] == generated["preview"]["svg"]


@pytest.mark.parametrize("path,name,magic", [
    ("/api/patterns/export/pdf", "pdf", b"%PDF"),
    ("/api/patterns/export/dxf", "dxf", b"SECTION"),
    ("/api/patterns/export/svg", "svg", b"<svg"),
])
def test_exports(yaml_target, confirmations, path, name, magic):
    response = client.post(path, json={"yamlText": yaml_target, "confirmations": confirmations})
    assert response.status_code == 200
    assert magic in response.content
    assert name in response.headers["content-disposition"]


def test_exports_are_blocked_when_semantics_are_unresolved(yaml_ok):
    for path in ("/api/patterns/export/pdf", "/api/patterns/export/dxf",
                 "/api/patterns/export/bundle"):
        response = client.post(path, json={"yamlText": yaml_ok})
        assert response.status_code == 409, path


def test_export_bundle_contains_every_required_file(yaml_target, confirmations):
    response = client.post("/api/patterns/export/bundle",
                           json={"yamlText": yaml_target, "confirmations": confirmations})
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        names = set(zf.namelist())
        assert {"pattern.json", "generation-manifest.json", "transformation-diagnostics.json",
                "validation-report.json", "base-vs-target-preview.png", "pattern-a0.pdf",
                "pattern.dxf", "base-block-landmark-map.json",
                "base-block-edge-map.json", "base-block-seam-map.json",
                "base-block-dart-map.json", "base-block-grainline-map.json"} <= names
        assert json.loads(zf.read("validation-report.json"))["status"] == "pass"
        manifest_data = json.loads(zf.read("generation-manifest.json"))
        assert manifest_data["base_block"]["id"] == "bodice-ref-a"
        diagnostics = json.loads(zf.read("transformation-diagnostics.json"))
        assert diagnostics["solver"]["front"]["converged"] is True


def test_config_endpoint():
    data = client.get("/api/config").json()
    assert "fitted" in data["easePresets"]
    assert data["paper"]["default_size"] == "A0"
    assert "cutting" in data["viewLayers"]
    assert set(data["comparisonLayers"]) == {"base", "displacement", "heatmap"}
    assert data["transformation"]["method"] == "thin_plate_spline"
    assert data["transformation"]["envelope"]["stop_factor"] == 2.0
    assert data["measurementSchema"]["schemaVersion"]


def test_healthz():
    assert client.get("/healthz").json() == {"status": "ok"}


def test_index_is_served_at_the_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "<div id=\"root\">" in response.text


def test_unknown_paths_are_404_not_a_page_of_html():
    """A probe must not be answered with 200 and an index page."""
    for path in ("/health", "/v1/models", "/api/does-not-exist"):
        response = client.get(path)
        assert response.status_code == 404, path
        if path.startswith("/api"):
            assert response.json()["error"]["code"] == "not_found"


def test_built_assets_are_served():
    index = client.get("/").text
    asset = next(part for part in index.split('"') if part.startswith("/assets/") and part.endswith(".js"))
    assert client.get(asset).status_code == 200


# ---- the exact sequence a browser runs --------------------------------------

def test_browser_sequence_from_upload_to_export(yaml_target, confirmations):
    """Reproduce the click path: syntax, definitions, base deltas, transform, export."""
    assert client.post("/api/measurements/validate",
                       json={"filename": "body.yaml", "content": yaml_target}).status_code == 200
    assert client.post("/api/measurements/semantics",
                       json={"yamlText": yaml_target}).status_code == 200
    assert client.get("/api/base-blocks").status_code == 200
    assert client.post("/api/base-block/deltas",
                       json={"yamlText": yaml_target,
                             "confirmations": confirmations}).status_code == 200
    assert client.post("/api/patterns/transform",
                       json={"yamlText": yaml_target,
                             "confirmations": confirmations}).status_code == 200
    assert client.post("/api/patterns/export/bundle",
                       json={"yamlText": yaml_target,
                             "confirmations": confirmations}).status_code == 200


def test_invalid_yaml_is_a_structured_error_everywhere():
    payload = {"yamlText": "body: [invalid"}
    for path in ("/api/measurements/semantics", "/api/measurements/report",
                 "/api/base-block/deltas", "/api/patterns/transform"):
        response = client.post(path, json=payload)
        assert response.status_code == 422, path
        assert response.json()["error"]["code"] == "invalid_yaml"
        assert "Traceback" not in response.text


def test_no_endpoint_returns_a_raw_traceback(yaml_ok, confirmations):
    """Broken payloads come back as structured errors, never a 500 stack trace."""
    reported = {"yamlText": yaml_ok, "confirmations": {"shoulder_w": "not_a_candidate"}}
    response = client.post("/api/measurements/semantics", json=reported)
    assert response.status_code == 200
    assert "Traceback" not in response.text
    gated = client.post("/api/patterns/transform", json=reported)
    assert gated.status_code == 409
    assert {"code", "message", "details"} <= set(gated.json()["error"])

    unparseable = {"yamlText": yaml_ok, "confirmations": confirmations,
                   "corrections": {"bust": "wide"}}
    for path in ("/api/measurements/semantics", "/api/patterns/transform"):
        response = client.post(path, json=unparseable)
        assert response.status_code == 422, path
        assert "Traceback" not in response.text


def test_unknown_request_keys_are_rejected_by_the_schema(yaml_ok):
    response = client.post("/api/measurements/semantics",
                           json={"yamlText": yaml_ok, "confirmations_unknown": 1})
    assert response.status_code == 422
    assert "Traceback" not in response.text


def test_an_unknown_candidate_is_refused_and_stays_blocked(yaml_ok):
    """Reported on /semantics so the operator can correct it, refused by /transform."""
    payload = {"yamlText": yaml_ok, "confirmations": {"shoulder_w": "nope"}}
    data = client.post("/api/measurements/semantics", json=payload).json()
    assert data["source"]["canGenerate"] is False
    detail = next(d for d in data["source"]["blocking"] if d["field"] == "shoulder_w")
    assert "not a candidate" in detail["message"]
    assert "half_neck_to_shoulder_tip" in detail["message"]

    gated = client.post("/api/patterns/transform", json=payload)
    assert gated.status_code == 409
    assert gated.json()["error"]["code"] == "semantics_unresolved"
