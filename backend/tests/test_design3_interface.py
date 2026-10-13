"""The common program interface.

The benchmark protocol calls `generate_pattern` and nothing else, so what is asserted
here is the contract that protocol depends on: the pattern object is JSON-serialisable
and complete in millimetres, refusals carry a code from `FAILURE_CODES`, and the same
input gives the same output.
"""
from __future__ import annotations

import json
import math

import pytest
import yaml

from app.design3_interface import (
    FAILURE_CODES,
    Design3Failure,
    generate_pattern,
    run_design_3,
)
from conftest import FIXTURES, TARGET_PROFILES, yaml_text


@pytest.fixture(scope="module")
def basic_pattern() -> dict:
    return generate_pattern(yaml.safe_load(yaml_text("body_measurements.yaml")),
                            {"ease_preset": "fitted"})


# ---- the object ------------------------------------------------------------


def test_pattern_is_json_serialisable(basic_pattern):
    """The flattened layout is a NumPy array; an uncast scalar would break the harness."""
    assert json.loads(json.dumps(basic_pattern))["schema"]["units"] == "mm"


def test_pattern_carries_every_required_member(basic_pattern):
    pattern = basic_pattern
    assert {"schema", "metadata", "panels", "cuttingLines", "stitches", "validation",
            "provenance"} <= set(pattern)
    assert pattern["metadata"]["units"]["pattern"] == "mm"
    for panel in pattern["panels"]:
        assert panel["id"] and panel["key"]
        assert panel["edges"], "a panel with no boundary is not a pattern"
        assert panel["grainline"] is not None
        assert panel["areaMm2"] > 0.0
        assert panel["foldEdgeIds"]


def test_boundaries_are_ordered_and_closed_in_millimetres(basic_pattern):
    for panel in basic_pattern["panels"]:
        edges = panel["edges"]
        assert edges, "a panel with no edges is not a pattern"
        for a, b in zip(edges, edges[1:] + edges[:1]):
            assert a["end"] == b["start"], f"{panel['key']} boundary is not closed"
        assert all(math.isfinite(c) for edge in edges for point in edge["points"]
                   for c in point)
        assert all(edge["lengthMm"] > 0.0 for edge in edges)


def test_fold_edges_carry_no_allowance_and_stitches_resolve(basic_pattern):
    by_id = {edge["id"]: edge for panel in basic_pattern["panels"]
             for edge in panel["edges"]}
    for panel in basic_pattern["panels"]:
        for edge_id in panel["foldEdgeIds"]:
            assert by_id[edge_id]["isFold"] is True
            assert by_id[edge_id]["seamAllowanceMm"] == 0.0
    for stitch in basic_pattern["stitches"]:
        assert len(stitch["records"]) == 2
        for record in stitch["records"]:
            assert record["startEdgeId"] in by_id
            assert record["endEdgeId"] in by_id


def test_paired_seams_name_each_other(basic_pattern):
    paired = [(panel["key"], edge["id"], edge["pairedEdgeId"])
              for panel in basic_pattern["panels"] for edge in panel["edges"]
              if edge["pairedEdgeId"]]
    assert paired, "the seam network declares paired seams and they must be linked"
    for _, edge_id, other in paired:
        assert other != edge_id


def test_cutting_lines_exist_for_every_panel(basic_pattern):
    assert set(basic_pattern["cuttingLines"]) == {p["key"] for p in basic_pattern["panels"]}
    for cutting in basic_pattern["cuttingLines"].values():
        assert len(cutting["contour"]) >= 3
        assert cutting["closed"] is True


# ---- refusals --------------------------------------------------------------


@pytest.mark.parametrize("case_id,filename", sorted(TARGET_PROFILES.items()))
def test_every_supported_profile_generates(case_id, filename):
    pattern = generate_pattern(yaml.safe_load(yaml_text(filename)), {})
    assert pattern["validation"]["status"] == "pass"
    assert pattern["validation"]["summary"]["criticalFailures"] == 0


@pytest.mark.parametrize("bad,expected", [
    ({}, "MISSING_MEASUREMENT"),
    ({"body": {}}, "MISSING_MEASUREMENT"),
    ({"body": {"bust": "wide"}}, "OUT_OF_DOMAIN"),
    ({"body": {"bust": -1.0}}, "OUT_OF_DOMAIN"),
])
def test_unusable_input_is_a_structured_failure(bad, expected):
    with pytest.raises(Design3Failure) as caught:
        generate_pattern(bad, {})
    assert caught.value.code == expected
    assert caught.value.code in FAILURE_CODES
    assert caught.value.details


def test_an_out_of_domain_profile_reports_out_of_domain():
    """`pascual` loads and displays, then is refused: that is the designed behaviour."""
    refused = yaml.safe_load((FIXTURES / "body_measurements_pascual.yaml")
                             .read_text(encoding="utf-8"))
    with pytest.raises(Design3Failure) as caught:
        generate_pattern(refused, {})
    assert caught.value.code == "OUT_OF_DOMAIN"


def test_unknown_garment_key_is_refused_not_ignored():
    body = yaml.safe_load(yaml_text("body_measurements.yaml"))
    with pytest.raises(Design3Failure) as caught:
        generate_pattern(body, {"ease_preset": "fitted", "silhouette": "hourglass"})
    assert caught.value.code == "OUT_OF_DOMAIN"
    assert "silhouette" in caught.value.message


# ---- the measured boundary -------------------------------------------------


def test_run_design_3_exports_a_dxf_and_reports_mesh_diagnostics(tmp_path):
    body = yaml.safe_load(yaml_text("body_measurements.yaml"))
    out = tmp_path / "pattern.dxf"
    run = run_design_3(body, {}, dxf_output=out)

    assert out.stat().st_size > 0
    assert run.dxf_path == str(out)
    assert {p["panel"] for p in run.diagnostics["panels"]} == {"front", "back"}
    for panel in run.diagnostics["panels"]:
        assert panel["converged"] is True
        assert panel["flippedTriangles"] == 0
        assert panel["diskTopology"] is True
        assert panel["iterations"] > 0


def test_generation_is_deterministic():
    body = yaml.safe_load(yaml_text("body_measurements.yaml"))
    first = generate_pattern(body, {})
    second = generate_pattern(body, {})
    for key in ("panels", "cuttingLines", "stitches"):
        assert json.dumps(first[key], sort_keys=True) == json.dumps(second[key], sort_keys=True)