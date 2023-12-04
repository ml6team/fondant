"""Interface for the data explorer app."""

import argparse
import logging
import os

import fsspec
import streamlit as st
from config import SESSION_STATE_VARIABLES
from interfaces.utils import get_default_index
from streamlit_extras.app_logo import add_logo

LOGGER = logging.getLogger(__name__)


class AppStateInterface:
    """
    Interface for the app state. The app state is responsible for initializing the app state
    variables and handling base path retrieval. It's implemented as a singleton class to ensure
    that the app state is initialized only once.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @staticmethod
    def _initialize_state_variables():
        """Initializes session state variables if not already initialized."""
        for session_state_variable in SESSION_STATE_VARIABLES:
            if session_state_variable not in st.session_state:
                st.session_state[session_state_variable] = None

    @staticmethod
    def _get_base_path() -> str:
        """Retrieves the base path from command line arguments or session state."""
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--base_path",
            type=str,
            help="Mounted, remote, or local base path",
        )
        args, _ = parser.parse_known_args()

        st.session_state["base_path"] = args.base_path

        return args.base_path

    def initialize(self):
        """Initializes the app state."""
        self._initialize_state_variables()
        self._get_base_path()


class MainInterface:
    """Abstract app class for the data explorer. The app class is responsible for
    initializing the state variables, setting up the sidebar and main page of the app,
    and providing a method to setup the main page content.
    """

    def __init__(self):
        app_interface = AppStateInterface()
        app_interface.initialize()
        st.set_page_config(layout="wide")
        self.fs, _ = fsspec.core.url_to_fs(st.session_state["base_path"])

    def _display_base_info(self):
        """Displays basic information about the base path and its type."""
        base_path = st.session_state["base_path"]
        with st.expander("## General Configuration"):
            st.markdown(f"### Base path: \n {base_path}")
            st.markdown(f"### Base path type: \n {self.fs.__class__.__name__}")

    def _select_pipeline(self):
        """Selects a pipeline from available pipelines."""
        base_path = st.session_state["base_path"]
        available_pipelines = [
            os.path.basename(item) for item in self.fs.ls(st.session_state["base_path"])
        ]

        default_index = get_default_index("pipeline", available_pipelines)
        selected_pipeline = st.selectbox(
            "Pipeline",
            available_pipelines,
            default_index,
        )
        selected_pipeline_path = os.path.join(base_path, selected_pipeline)

        st.session_state["pipeline"] = selected_pipeline
        st.session_state["pipeline_path"] = selected_pipeline_path

    @staticmethod
    def _select_run():
        """Selects a run from available runs within the chosen pipeline."""
        selected_pipeline_path = st.session_state["pipeline_path"]

        def has_manifest_file(path):
            return any("manifest.json" in files for _, _, files in os.walk(path))

        available_runs = []
        for run in os.listdir(selected_pipeline_path):
            run_path = os.path.join(selected_pipeline_path, run)
            if (
                os.path.isdir(run_path)
                and run != "cache"
                and has_manifest_file(run_path)
            ):
                available_runs.append(os.path.basename(run))

        available_runs.sort(reverse=True)

        default_index = get_default_index("run", available_runs)
        selected_run = st.selectbox("Run", available_runs, default_index)
        selected_run_path = os.path.join(selected_pipeline_path, selected_run)

        st.session_state["run"] = selected_run
        st.session_state["run_path"] = selected_run_path
        st.session_state["available_runs"] = available_runs

    def create_common_interface(self):
        """Sets up the Streamlit app's main interface with common elements."""
        add_logo("content/fondant_logo.png")

        with st.sidebar:
            # Increase the width of the sidebar to accommodate logo
            st.markdown(
                """
                <style>
                    section[data-testid="stSidebar"] {
                        width: 350px !important;
                    }
                </style>
                """,
                unsafe_allow_html=True,
            )
            self._display_base_info()

        cols = st.columns(2)
        with cols[0]:
            self._select_pipeline()
        with cols[1]:
            self._select_run()
