"""Main module for the data explorer interface."""

import argparse
import logging
import os
import typing as t
from datetime import datetime

import dask
import graphviz
import pandas as pd
import streamlit as st
from config import SESSION_STATE_VARIABLES
from fondant.core.manifest import Manifest
from interfaces.common_interface import MainInterface
from st_pages import show_pages_from_config

LOGGER = logging.getLogger(__name__)

dask.config.set({"dataframe.convert-string": False})


def initialize_state():
    for session_state_variable in SESSION_STATE_VARIABLES:
        if session_state_variable not in st.session_state:
            st.session_state[session_state_variable] = None

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_path",
        type=str,
        help="Mounted, remote, or local base path",
    )
    args, _ = parser.parse_known_args()

    st.session_state["base_path"] = args.base_path

    print(st.session_state)


class PipelineOverviewApp(MainInterface):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_ordered_manifests_paths(run_path: str) -> t.List[str]:
        """
        Function to get the ordered manifests where the first manifest is the first
        step in the pipeline.

        Returns:
            List of manifests path in order of their component execution.
        """
        run_components = [os.path.basename(item) for item in os.listdir(run_path)]

        # Get manifest paths
        manifest_paths = [
            os.path.join(*[run_path, component, "manifest.json"])
            for component in run_components
        ]

        # Filter for non existing manifests (e.g. if the component failed)
        manifest_paths = [path for path in manifest_paths if os.path.exists(path)]

        # Sort paths based on the modified time
        manifest_paths = sorted(manifest_paths, key=os.path.getmtime)

        return manifest_paths

    @staticmethod
    def get_ordered_manifests(ordered_manifest_paths: t.List[str]) -> t.List[Manifest]:
        """
        Function to get the ordered manifests where the first manifest is the first
        step in the pipeline.

        Returns:
            List of manifests in order of their component execution

        """
        # Get the manifests
        return [Manifest.from_file(path) for path in ordered_manifest_paths]

    @staticmethod
    def create_component_table(manifest: Manifest) -> str:
        """
        Function to create a table for the component.

        Args:
            manifest: Manifest of the component.

        Returns:
            graphviz table of the component a graphviz string.
        """
        fields = manifest.fields
        component_name = manifest.component_id

        fields_with_schema = [
            (field_name, field_schema.type.to_json()["type"])
            for field_name, field_schema in fields.items()
        ]

        fields_list = [
            f"{field_name} \\n({field_schema})"
            for field_name, field_schema in fields_with_schema
        ]
        fields_str = "|\n".join(fields_list)

        return f"{{ {component_name} |{{\n{fields_str}\n}}}}"

    def get_pipeline_graph(self, selected_run_path: str) -> graphviz.Digraph:
        """Get the pipeline graph."""
        ordered_manifest_paths = self.get_ordered_manifests_paths(selected_run_path)
        ordered_manifest_list = self.get_ordered_manifests(ordered_manifest_paths)

        graph = graphviz.Digraph("structs", node_attr={"shape": "record"})
        graph.attr("node", fontsize="12")
        previous_component_id = None

        for manifest in ordered_manifest_list:
            component_id = manifest.component_id

            table_text = self.create_component_table(manifest)

            graph.node(
                name=component_id,
                label=rf"{table_text}",
            )

            if previous_component_id is not None:
                graph.edge(tail_name=previous_component_id, head_name=component_id)
            previous_component_id = component_id

        return graph

    def get_pipeline_info_df(self):
        """Get the pipeline info dataframe."""
        selected_pipeline_path = st.session_state["pipeline_path"]

        available_runs = st.session_state["available_runs"]
        available_runs_path = [
            os.path.join(selected_pipeline_path, run) for run in available_runs
        ]

        data = []
        for run_path in available_runs_path:
            ordered_manifest_paths = self.get_ordered_manifests_paths(run_path)
            last_manifest_path = ordered_manifest_paths[-1]
            last_update_date = datetime.fromtimestamp(
                os.path.getmtime(last_manifest_path),
            ).strftime("%-d/%-m/%Y  %H:%M:%S")
            run_name = os.path.basename(run_path)
            data.append([run_name, last_update_date, run_path])

        return pd.DataFrame(data, columns=["Run Name", "Last Updated", "Run Path"])

    def setup_app_page(self, max_runs_to_display=5):
        """Setup the main page of the app."""
        pipeline_df = self.get_pipeline_info_df()
        selected_run = st.session_state["run"]
        selected_run_info = pipeline_df[pipeline_df["Run Name"] == selected_run]
        selected_run_update_date = selected_run_info["Last Updated"].iloc[0]
        selected_run_path = selected_run_info["Run Path"].iloc[0]

        st.markdown(f" **Last Updated**: {selected_run_update_date}")

        cols = st.columns([0.4, 0.6])
        with cols[0]:
            graph = self.get_pipeline_graph(selected_run_path)
            st.graphviz_chart(graph)

        with cols[1]:
            pipeline_df = pipeline_df.drop(columns=["Run Path"])
            st.markdown(f"Last {max_runs_to_display} runs:")
            st.table(pipeline_df[:max_runs_to_display])


initialize_state()

# Show streamlit page from pages.toml config file
show_pages_from_config()

app = PipelineOverviewApp()

app.create_common_interface()

app.setup_app_page()
