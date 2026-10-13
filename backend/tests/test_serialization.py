"""GarmageSet serialisation: reference fixture, round trip, determinism."""
from __future__ import annotations

import json

import pytest

from app.schemas.garmage import GarmageSet
from app.serialization.garmageset import serialize, to_fixture_dict


def test_reference_fixture_matches_the_typed_schema(reference_fixture):
    model = GarmageSet.model_validate(reference_fixture)
    assert model.panels and model.stitches
    assert len(model.stitches[0].records) == 2


def test_reference_fixture_uses_the_documented_defaults(reference_fixture, drafting_config):
    defaults = drafting_config["fixture_defaults"]
    panel = reference_fixture["panels"][0]
    assert panel["particleDistance"] == defaults["panel_particle_distance"]
    assert panel["collisionLayer"] == defaults["panel_collision_layer"]
    seq = panel["seqEdges"][0]
    assert seq["type"] == defaults["seq_edge_type"]
    assert seq["circleType"] == defaults["seq_edge_circle_type"]
    assert all(edge["bezierPoints"] == defaults["edge_bezier_points"] for edge in seq["edges"])


def test_reference_fixture_is_renderable(reference_fixture):
    """Our typed model round-trips the reference document's own shape."""
    model = GarmageSet.model_validate(reference_fixture)
    again = GarmageSet.model_validate(model.model_dump())
    assert len(again.panels) == len(model.panels)
    assert all(len(p.center) == 3 and p.center[2] == 0 for p in model.panels)


def test_generated_output_matches_the_schema(basic_result):
    model = serialize(basic_result.document)
    assert len(model.panels) == 2
    for panel in model.panels:
        assert len(panel.seqEdges) >= 1
        for seq in panel.seqEdges:
            assert len(seq.vertices) == len(seq.edges) >= 3
            assert seq.type == 3 and seq.circleType == 0


def test_straight_edges_serialise_two_endpoints(basic_result):
    payload = to_fixture_dict(basic_result.document)
    front = payload["panels"][0]["seqEdges"][0]
    straight = [e for e in front["edges"] if len(e["controlPoints"]) == 2]
    assert straight, "expected straight edges to serialise as two endpoints"


def test_curved_edges_keep_their_control_polyline(basic_result):
    """A template edge is a control polyline, and it survives serialisation as one."""
    payload = to_fixture_dict(basic_result.document)
    front = payload["panels"][0]["seqEdges"][0]
    curved = [e for e in front["edges"] if len(e["controlPoints"]) > 2]
    assert curved
    for edge in curved:
        assert all(len(p) == 3 and p[2] == 0 for p in edge["controlPoints"])


def test_coordinates_are_triples_in_mm(basic_result):
    payload = to_fixture_dict(basic_result.document)
    for panel in payload["panels"]:
        assert len(panel["center"]) == 3
        for seq in panel["seqEdges"]:
            for v in seq["vertices"]:
                assert len(v) == 3


def test_panel_geometry_is_local_and_center_carries_placement(basic_result):
    payload = to_fixture_dict(basic_result.document)
    for panel in payload["panels"]:
        xs = [v[0] for seq in panel["seqEdges"] for v in seq["vertices"]]
        assert min(xs) < 0 < max(xs), "panel-local geometry should straddle its own centre"


def test_identifiers_are_uuid5_and_unique(basic_result):
    payload = to_fixture_dict(basic_result.document)
    ids = [p["id"] for p in payload["panels"]]
    ids += [e["id"] for p in payload["panels"] for s in p["seqEdges"] for e in s["edges"]]
    assert len(ids) == len(set(ids))
    assert all(len(i) == 36 for i in ids)


def test_stitches_reference_generated_edges(basic_result):
    payload = to_fixture_dict(basic_result.document)
    known = {e["id"]: p["id"] for p in payload["panels"] for s in p["seqEdges"] for e in s["edges"]}
    for pair in payload["stitches"]:
        assert len(pair) == 2
        for record in pair:
            for ref in (record["start"], record["end"]):
                assert ref["edgeId"] in known
                assert known[ref["edgeId"]] == ref["clothPieceId"]
                assert 0.0 <= ref["param"] <= 1.0


def test_round_trip_through_json(basic_result):
    payload = to_fixture_dict(basic_result.document)
    text = json.dumps(payload, ensure_ascii=False)
    again = GarmageSet.model_validate(json.loads(text))
    assert again == serialize(basic_result.document)


def test_round_trip_preserves_coordinates_exactly(basic_result):
    payload = to_fixture_dict(basic_result.document)
    again = json.loads(json.dumps(payload))
    assert again == payload


def test_serialisation_is_deterministic():
    from conftest import generate_profile

    first = to_fixture_dict(generate_profile("body_measurements_target_a.yaml").document)
    second = to_fixture_dict(generate_profile("body_measurements_target_a.yaml").document)
    assert json.dumps(first) == json.dumps(second)


def test_unknown_top_level_keys_are_rejected(reference_fixture):
    reference_fixture["surprise"] = 1
    with pytest.raises(Exception):
        GarmageSet.model_validate(reference_fixture)


@pytest.fixture(scope="module")
def drafting_config():
    from app.config import drafting_config as load
    return load()
