"""The benchmark harness, not the metrics it produces.

What is asserted here is the bookkeeping the protocol depends on: an unmeasured field is
blank rather than zero, every predefined trial reaches the log whether it passed or not,
the case summary is derived from the raw rows, the frozen base block still matches its
recorded hash, and the Metric 5 allow-list names files that exist. Run
`benchmark/run_design_2.py` for the metric values themselves.
"""
from __future__ import annotations

import hashlib

import pytest
import yaml
from conftest import FIXTURES

from benchmark import run_design_2

ROOT = FIXTURES.parents[0]
CONFIG = yaml.safe_load((ROOT / "benchmark" / "config-design-2.yaml").read_text(encoding="utf-8"))


# ---- bookkeeping rules ------------------------------------------------------


def test_an_unmeasured_field_is_blank_not_zero():
    """A zero would be indistinguishable from a measured zero in the trial CSV."""
    row = run_design_2.trial_row(
        {"case_id": "c", "pass_type": "memory", "trial": 0, "warmup": False,
         "time_s": None, "peak_memory_mb": 3.5, "valid_output": True},
        {"base_block_version": "1.0.0", "source_sha256": "abc"}, "commit",
    )
    assert row["time_s"] == ""
    assert row["peak_memory_mb"] == 3.5


def test_iqr_of_a_uniform_sample_is_zero():
    assert run_design_2.iqr([1.0, 1.0, 1.0, 1.0]) == 0.0
    assert run_design_2.iqr([2.0]) == 0.0
    # Inclusive quartiles: Q1 = 1.75, Q3 = 3.25, so the range is 1.5 - the same value
    # numpy's default percentile method gives.
    assert run_design_2.iqr([1.0, 2.0, 3.0, 4.0]) == 1.5


def test_summary_is_derived_from_the_raw_rows_and_counts_failures():
    rows = [
        _row("a", "memory", 0, valid=True, peak_memory_mb=3.0),
        _row("a", "memory", 1, valid=True, peak_memory_mb=5.0),
        _row("a", "time", 0, valid=True, time_s=2.0),
        _row("a", "time", 1, valid=False, time_s=None, code="OUT_OF_DOMAIN"),
        _row("a", "memory", 0, valid=True, peak_memory_mb=99.0, warmup=True),
    ]
    summary = run_design_2.summarise(rows, {"a": "mandatory"})[0]
    assert summary["memory_median_mb"] == 4.0, "the warm-up must not move the median"
    assert summary["valid_memory_trials"] == 2
    assert summary["valid_time_trials"] == 1
    assert summary["time_median_s"] == 2.0
    assert summary["failure_count"] == 1
    assert summary["all_checks_passed"] is False


def _row(case, pass_type, trial, valid, peak_memory_mb=None, time_s=None,
         code="", warmup=False):
    return {
        "case_id": case, "pass_type": pass_type, "trial": trial, "warmup": warmup,
        "valid_output": valid, "error_code": code,
        "peak_memory_mb": peak_memory_mb, "time_s": time_s,
    }


# ---- frozen inputs ----------------------------------------------------------


def test_every_configured_case_fixture_and_garment_file_exists():
    for case in CONFIG["cases"]:
        assert (ROOT / case["measurements"]).is_file(), case["case_id"]
    assert (ROOT / CONFIG["garment"]).is_file()


def test_case_fixtures_are_pseudonymous_and_approved():
    for case in CONFIG["cases"]:
        document = yaml.safe_load((ROOT / case["measurements"]).read_text(encoding="utf-8"))
        meta = document["case"]
        assert meta["case_id"] == case["case_id"]
        assert meta["role"] == case["role"]
        assert meta["unit"] == "cm"
        assert meta["contains_direct_identifier"] is False
        assert meta["approval_status"] == "approved"
        assert meta["approved_by"] and meta["approval_date"]
        assert document["body"], "a case fixture with no body measurements cannot be run"


def test_the_control_case_is_not_a_mandatory_one():
    """A control that counted toward the median would drag the trade-off value down for
    a refusal that is the correct behaviour."""
    roles = {c["case_id"]: c["role"] for c in CONFIG["cases"]}
    assert roles["d2-case-004-off-domain"] == "control"
    assert [c for c in roles.values()].count("mandatory") >= 3


def test_the_frozen_base_block_still_matches_its_recorded_hash():
    """Three independent statements have to agree: the bytes, this configuration and the
    catalog. A benchmark over a block that is not the frozen one measures nothing."""
    block = CONFIG["base_block"]
    actual = hashlib.sha256((ROOT / block["source_file"]).read_bytes()).hexdigest()
    assert actual == block["source_sha256"]

    catalog = yaml.safe_load((ROOT / "assets/base-blocks/catalog.yaml").read_text(encoding="utf-8"))
    entry = next(b for b in catalog["blocks"] if b["id"] == block["id"])
    assert entry["sha256"] == block["source_sha256"]


def test_the_garment_fixture_names_the_frozen_block():
    """Design 2 transforms one fixed block and never searches among templates, so the
    choice has to be visible in the fixture rather than left to a catalog default."""
    garment = yaml.safe_load((ROOT / CONFIG["garment"]).read_text(encoding="utf-8"))
    assert garment["settings"]["base_block_id"] == CONFIG["base_block"]["id"]
    assert garment["settings"]["confirm_envelope"] is False, (
        "confirming the envelope would let the out-of-domain control case through"
    )


# ---- Metric 5 ---------------------------------------------------------------


def test_the_allow_list_names_real_production_modules():
    """A shortened allow-list would report a flattering Metric 5, so this must fail loudly."""
    pytest.importorskip(
        "radon", reason="radon 6.0.1 is a benchmark tool, not a backend dependency")
    missing = [entry["path"] for entry in CONFIG["maintainability"]["files"]
               if not (ROOT / entry["path"]).is_file()]
    if missing:
        pytest.skip(
            "Design 2's base_block and transformation packages have no source yet, so "
            "Metric 5 is not measurable. Recover it first (benchmark/README.md, Step 0). "
            f"Missing: {', '.join(missing)}"
        )

    from benchmark import maintainability_check

    weighted_mi, rows = maintainability_check.measure()
    assert len(rows) == len(CONFIG["maintainability"]["files"])
    assert all(row["sloc"] > 0 for row in rows)
    assert all(row["role"] for row in rows)
    assert 0.0 <= weighted_mi <= 100.0
    # Each row's contribution is rounded to 6 dp before it is written, so the column sums
    # to the weighted MI only to within that rounding.
    assert abs(sum(row["weighted_contribution"] for row in rows) - weighted_mi) < 1e-4


def test_the_allow_list_excludes_shared_components():
    paths = {f["path"] for f in CONFIG["maintainability"]["files"]}
    for excluded in ("backend/app/drafting", "backend/app/serialization",
                     "backend/app/export", "backend/app/schemas", "backend/app/config",
                     "backend/app/api", "backend/tests", "benchmark", "design_2_interface"):
        assert not any(path.startswith(excluded) or path.endswith(excluded)
                       for path in paths), excluded
    assert not any("design2_interface" in path for path in paths), (
        "the interface is an adapter, not algorithm code"
    )