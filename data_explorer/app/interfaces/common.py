"""Interface for the data explorer app."""

import os

import streamlit as st
from interfaces.utils import get_index_from_state


def render_common_interface():
    cols = st.columns(2)
    with cols[0]:
        _select_pipeline()
    with cols[1]:
        _select_run()


def _select_pipeline():
    """Selects a pipeline from available pipelines."""
    available_pipelines = [
        os.path.basename(item)
        for item in st.session_state.file_system.ls(st.session_state.base_path)
    ]

    selected_pipeline = st.selectbox(
        "Pipeline",
        options=available_pipelines,
        index=get_index_from_state("pipeline", available_pipelines),
    )
    st.session_state.pipeline = selected_pipeline
    st.session_state.pipeline_path = st.session_state.base_path / selected_pipeline


def _select_run():
    """Selects a run from available runs within the chosen pipeline."""
    pipeline_path = st.session_state.base_path / st.session_state.pipeline

    def has_manifest_file(path):
        return any("manifest.json" in files for _, _, files in os.walk(path))

    available_runs = []
    for run in os.listdir(pipeline_path):
        run_path = pipeline_path / run
        if run_path.is_dir() and run != "cache" and has_manifest_file(run_path):
            available_runs.append(os.path.basename(run))

    available_runs.sort(reverse=True)

    selected_run = st.selectbox(
        "Run",
        options=available_runs,
        index=get_index_from_state("run", available_runs),
    )
    st.session_state.run = selected_run
    st.session_state.run_path = st.session_state.pipeline_path / selected_run
    st.session_state.available_runs = available_runs
