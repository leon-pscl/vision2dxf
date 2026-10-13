"""The benchmark harness, not the metrics it produces.

What is asserted here is the bookkeeping the protocol depends on: an unmeasured field is
blank rather than zero, every predefined trial reaches the log whether it passed or not,
the case summary is derived from the raw rows, and the Metric 5 allow-list names files
that exist. Run `benchmark/run_design_3.py` for the metric values themselves.
"""
from __future__ import annotations

import pytest
import yaml
from conftest import FIXTURES

from benchmark import run_design_3

CONFIG = yaml.safe_load(
    (FIXTURES.parents[0] / "benchmark" / "config-design-3.yaml").read_text(encoding="utf-8"))


# ---- bookkeeping rules ------------------------------------------------------


def test_an_unmeasured_field_is_blank_not_zero():
    """A zero would be indistinguishable from a measured zero in the trial CSV."""
    row = run_design_3.trial_row(
        {"case_id": "c", "pass_type": "memory", "trial": 0, "warmup": False,
         "time_s": None, "peak_memory_mb": 3.5, "valid_output": True},
        "hash", "commit", "1.0.0",
    )
    assert row["time_s"] == ""
    assert row["peak_memory_mb"] == 3.5


def test_euler_characteristic_of_a_disk():
    """A quad split into two triangles: V=4, F=2, four boundary edges gives chi = 1."""
    assert run_design_3.euler_characteristic(4, 2, 4) == 1


def test_iqr_of_a_uniform_sample_is_zero():
    assert run_design_3.iqr([1.0, 1.0, 1.0, 1.0]) == 0.0
    assert run_design_3.iqr([2.0]) == 0.0
    # Inclusive quartiles: Q1 = 1.75, Q3 = 3.25, so the range is 1.5 - the same value
    # numpy's default percentile method gives.
    assert run_design_3.iqr([1.0, 2.0, 3.0, 4.0]) == 1.5


def test_summary_is_derived_from_the_raw_rows_and_counts_failures():
    rows = [
        _row("a", "memory", 0, valid=True, peak_memory_mb=3.0),
        _row("a", "memory", 1, valid=True, peak_memory_mb=5.0),
        _row("a", "time", 0, valid=True, time_s=2.0),
        _row("a", "time", 1, valid=False, time_s=None, code="OUT_OF_DOMAIN"),
        _row("a", "memory", 0, valid=True, peak_memory_mb=99.0, warmup=True),
    ]
    summary = run_design_3.summarise(rows, {"a": "mandatory"})[0]
    assert summary["memory_median_mb"] == 4.0, "the warm-up must not move the median"
    assert summary["valid_memory_trials"] == 2
    assert summary["valid_time_trials"] == 1
    assert summary["time_median_s"] == 2.0
    assert summary["failure_count"] == 1
    assert summary["all_mesh_checks_passed"] is False


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
        assert (FIXTURES.parents[0] / case["measurements"]).is_file(), case["case_id"]
    assert (FIXTURES.parents[0] / CONFIG["garment"]).is_file()


def test_case_fixtures_are_pseudonymous_and_approved():
    for case in CONFIG["cases"]:
        document = yaml.safe_load(
            (FIXTURES.parents[0] / case["measurements"]).read_text(encoding="utf-8"))
        meta = document["case"]
        assert meta["case_id"] == case["case_id"]
        assert meta["unit"] == "cm"
        assert meta["contains_direct_identifier"] is False
        assert meta["approval_status"] == "approved"
        assert meta["approved_by"] and meta["approval_date"]
        assert document["body"], "a case fixture with no body measurements cannot be run"


def test_the_allow_list_names_real_production_modules():
    """A shortened allow-list would report a flattering Metric 5, so this must fail loudly."""
    maintainability_check = pytest.importorskip(
        "benchmark.maintainability_check",
        reason="radon 6.0.1 is a benchmark tool, not a backend dependency",
    )
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
    for excluded in ("backend/app/validation", "backend/app/export", "backend/app/schemas",
                     "backend/app/config", "backend/tests", "benchmark"):
        assert not any(path.startswith(excluded) for path in paths), excluded