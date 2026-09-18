#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st
import pandas as pd
import json
import yaml
import io

from vision2dxf.core.models import MeasurementProfile
from vision2dxf.core.interfaces import load_profiles, load_json
from vision2dxf.core.validation import validate_pattern
from vision2dxf.core.preview import render_preview
from vision2dxf.core.dxf_export import export_json, export_dxf
from vision2dxf.designs.formula_based.generator import FormulaBasedGenerator
from vision2dxf.designs.parametric_block.generator import ParametricBlockGenerator
from vision2dxf.designs.template_grading.generator import TemplateGradingGenerator
from vision2dxf.evaluation.benchmark import benchmark_generator
from vision2dxf.evaluation.reliability import measure_reliability
from vision2dxf.evaluation.code_metrics import measure_code_metrics
from vision2dxf.sensitivity.normalize import normalize_matrix
from vision2dxf.sensitivity.score import compute_scores
from vision2dxf.sensitivity.summarize import summarize_robustness, generate_recommendation

st.set_page_config(page_title="Vision2DXF", layout="wide")

GENERATORS = {
    "Design A — Formula-Based": FormulaBasedGenerator(),
    "Design B — Parametric Block": ParametricBlockGenerator(),
    "Design C — Template Grading": TemplateGradingGenerator(),
}

PROFILES_PATH = Path("config/test_profiles.json")
CRITERIA_PATH = Path("config/criteria.yaml")


def _load_profiles_cached():
    if PROFILES_PATH.exists():
        return load_profiles(PROFILES_PATH)
    return []


def _profile_to_dict(p: MeasurementProfile) -> dict:
    return {
        "profile_id": p.profile_id, "chest": p.chest, "waist": p.waist,
        "hip": p.hip, "shoulder": p.shoulder, "torso_length": p.torso_length,
        "armhole_depth": p.armhole_depth, "sleeve_length": p.sleeve_length, "ease": p.ease,
    }


def tab_generate():
    st.header("Generate Pattern")

    profiles = _load_profiles_cached()
    if not profiles:
        st.warning("No test profiles found in config/test_profiles.json")
        return

    profile_ids = [p.profile_id for p in profiles]
    selected_id = st.selectbox("Select measurement profile", profile_ids)
    profile = next(p for p in profiles if p.profile_id == selected_id)

    with st.expander("Profile measurements", expanded=False):
        st.json(_profile_to_dict(profile))

    design = st.selectbox("Select design", list(GENERATORS.keys()))
    gen = GENERATORS[design]

    if st.button("Generate"):
        with st.spinner("Generating..."):
            result = gen.generate(profile)
            result = validate_pattern(result)

        col1, col2 = st.columns([1, 1])
        with col1:
            fig = render_preview(result)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("Validation")
            if result.valid:
                st.success("PASS")
            else:
                for err in result.validation_errors:
                    st.error(err)

            st.subheader("Metadata")
            st.json(result.metadata)

        # Export
        st.subheader("Export")
        col_a, col_b = st.columns(2)
        with col_a:
            json_str = json.dumps({
                "design_id": result.design_id,
                "points": {k: list(v) for k, v in result.points.items()},
                "segments": result.segments,
                "valid": result.valid,
                "validation_errors": result.validation_errors,
                "metadata": result.metadata,
            }, indent=2)
            st.download_button("Download JSON", json_str, file_name=f"{result.design_id}.json")
        with col_b:
            dxf_path = Path("outputs/patterns") / f"{result.design_id}.dxf"
            if export_dxf(result, dxf_path):
                st.download_button("Download DXF", dxf_path.read_bytes(), file_name=f"{result.design_id}.dxf")
            else:
                st.info("DXF export requires ezdxf")


def tab_compare():
    st.header("Compare Designs")

    profiles = _load_profiles_cached()
    if not profiles:
        st.warning("No test profiles found")
        return

    profile_ids = [p.profile_id for p in profiles]
    selected_id = st.selectbox("Select measurement profile", profile_ids, key="cmp_profile")
    profile = next(p for p in profiles if p.profile_id == selected_id)

    results = {}
    for name, gen in GENERATORS.items():
        r = gen.generate(profile)
        r = validate_pattern(r)
        results[name] = r

    cols = st.columns(3)
    for i, (name, r) in enumerate(results.items()):
        with cols[i]:
            st.subheader(name)
            fig = render_preview(r)
            st.plotly_chart(fig, use_container_width=True)
            if r.valid:
                st.success("Valid")
            else:
                st.warning(f"{len(r.validation_errors)} errors")

    # Summary table
    rows = []
    for name, r in results.items():
        rows.append({
            "Design": name,
            "Valid": r.valid,
            "Errors": len(r.validation_errors),
            "Points": len(r.points),
            "Segments": len(r.segments),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True)


def tab_evaluate():
    st.header("Evaluate")

    profiles = _load_profiles_cached()
    if not profiles:
        st.warning("No profiles")
        return

    warmup = st.number_input("Warmup runs", min_value=1, value=1)
    reps = st.number_input("Measured runs", min_value=5, value=30)

    if st.button("Run Evaluation"):
        with st.spinner("Running benchmark..."):
            perf_results = []
            rel_results = []
            for name, gen in GENERATORS.items():
                perf_results.extend(benchmark_generator(gen, profiles, warmup, reps))
                rel_results.extend(measure_reliability(gen, profiles))

            design_paths = [
                "src/vision2dxf/designs/formula_based",
                "src/vision2dxf/designs/parametric_block",
                "src/vision2dxf/designs/template_grading",
            ]
            code_met = measure_code_metrics(design_paths)

        st.subheader("Performance")
        perf_rows = []
        for r in perf_results:
            perf_rows.append({
                "Design": r["design_id"],
                "Profile": r["profile_id"],
                "Median (ms)": round(r["median_ms"], 3),
            })
        st.dataframe(pd.DataFrame(perf_rows), hide_index=True)

        st.subheader("Reliability")
        rel_rows = []
        for r in rel_results:
            rel_rows.append({
                "Design": r["design_id"],
                "Profile": r["profile_id"],
                "Failures": r["failures"],
                "Failure %": round(r["failure_rate_percent"], 2),
            })
        st.dataframe(pd.DataFrame(rel_rows), hide_index=True)

        st.subheader("Code Metrics")
        st.dataframe(pd.DataFrame(code_met), hide_index=True)

        st.subheader("SUS Status")
        st.info("Pending — no participant data collected.")


def tab_sensitivity():
    st.header("Sensitivity Analysis")

    raw_path = Path("outputs/raw/raw_constraint_values.csv")
    if not raw_path.exists():
        st.info("Run Evaluate first to generate raw constraint values.")
        return

    raw_df = pd.read_csv(raw_path, index_col=0)
    criteria_cfg = yaml.safe_load(open(CRITERIA_PATH))["criteria"]
    directions = {k: v["direction"] for k, v in criteria_cfg.items()}

    norm_df, missing = compute_scores(raw_df, directions)

    design_ids = [c.replace("score_", "") for c in norm_df.columns if c.startswith("score_")]
    robust = summarize_robustness(norm_df, design_ids)

    st.subheader("Normalized Ratings")
    norm_display = norm_df[[c for c in norm_df.columns if c.startswith("score_")]].copy()
    norm_display.columns = [c.replace("score_", "") for c in norm_display.columns]
    st.dataframe(norm_display.describe().round(2), use_container_width=True)

    st.subheader("Design Robustness")
    st.dataframe(robust, hide_index=True, use_container_width=True)

    st.subheader("Win Distribution")
    st.bar_chart(robust.set_index("design_id")["outright_wins"])

    rec = generate_recommendation(robust, raw_df.to_dict(orient="index"), list(raw_df.columns))
    st.subheader("Recommendation")
    if rec:
        st.json(rec)
    else:
        st.warning("Missing required criterion values (e.g. SUS). Cannot generate recommendation.")


tabs = st.tabs(["Generate", "Compare", "Evaluate", "Sensitivity"])

with tabs[0]:
    tab_generate()
with tabs[1]:
    tab_compare()
with tabs[2]:
    tab_evaluate()
with tabs[3]:
    tab_sensitivity()
