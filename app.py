#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st
import pandas as pd
import json

from vision2dxf.core.models import MeasurementProfile
from vision2dxf.core.interfaces import load_profiles
from vision2dxf.core.validation import validate_pattern
from vision2dxf.core.preview import render_preview
from vision2dxf.core.dxf_export import export_json, export_dxf
from vision2dxf.designs.formula_based.generator import FormulaBasedGenerator
from vision2dxf.designs.parametric_block.generator import ParametricBlockGenerator
from vision2dxf.designs.template_grading.generator import TemplateGradingGenerator
from vision2dxf.evaluation.code_metrics import measure_code_metrics

st.set_page_config(page_title="Vision2DXF", layout="wide")

GENERATORS = {
    "Design A — Formula-Based": FormulaBasedGenerator(),
    "Design B — Parametric Block": ParametricBlockGenerator(),
    "Design C — Template Grading": TemplateGradingGenerator(),
}

PROFILES_PATH = Path("config/test_profiles.json")

DESIGN_PATHS = [
    "src/vision2dxf/designs/formula_based",
    "src/vision2dxf/designs/parametric_block",
    "src/vision2dxf/designs/template_grading",
]


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

    if st.button("Run Evaluation"):
        with st.spinner("Running code analysis..."):
            code_met = measure_code_metrics(DESIGN_PATHS)

        st.subheader("Raw Constraint Values")

        out_path = Path("outputs/raw/raw_constraint_values.csv")
        if out_path.exists():
            raw_df = pd.read_csv(out_path, index_col=0)
            st.dataframe(raw_df, use_container_width=True)
        else:
            st.info("Evaluation did not produce output. Check logs.")

        st.subheader("Code Metrics Detail")
        st.dataframe(pd.DataFrame(code_met), hide_index=True)


tabs = st.tabs(["Generate", "Compare", "Evaluate"])

with tabs[0]:
    tab_generate()
with tabs[1]:
    tab_compare()
with tabs[2]:
    tab_evaluate()
