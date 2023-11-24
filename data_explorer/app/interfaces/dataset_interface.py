"""Dataset interface for the data explorer app."""

import os
import typing as t

import dask.dataframe as dd
import streamlit as st
from fondant.core.manifest import Manifest
from interfaces.common_interface import MainInterface
from interfaces.utils import get_default_index


class DatasetLoaderApp(MainInterface):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _select_component():
        """Select component from available components."""
        selected_run_path = st.session_state["run_path"]
        available_components = [
            os.path.basename(item) for item in os.listdir(selected_run_path)
        ]

        default_index = get_default_index("component", available_components)
        selected_component = st.selectbox(
            "Select component",
            available_components,
            default_index,
        )
        selected_component_path = os.path.join(selected_run_path, selected_component)

        st.session_state["component"] = selected_component
        st.session_state["selected_component_path"] = selected_component_path

        return selected_component_path

    def _get_manifest_fields_and_subset(self):
        """Get fields and subset from manifest and store them in session state."""
        cols = st.columns(3)

        with cols[0]:
            selected_component_path = self._select_component()

        manifest_path = os.path.join(selected_component_path, "manifest.json")
        manifest = Manifest.from_file(manifest_path)
        subsets = manifest.subsets.keys()

        with cols[1]:
            subset = st.selectbox("Select subset", subsets)

        fields = manifest.subsets[subset].fields

        with cols[2]:
            fields = st.multiselect("Fields", fields, default=fields)

        field_types = {
            f"{field}": manifest.subsets[subset].fields[field].type.name
            for field in fields
        }
        return manifest, subset, field_types

    def _get_subset_path(self, manifest, subset):
        """
        Get path to subset from manifest. If the base path is not mounted, the subset path is
        assumed to be relative to the base path. If the base path is mounted, the subset path is
        assumed to be absolute.
        """
        base_path = st.session_state["base_path"]
        subset = manifest.subsets[subset]

        if (
            os.path.ismount(base_path) is False
            and self.fs.__class__.__name__ == "LocalFileSystem"
        ):
            # Used for local development when running the app locally
            subset_path = os.path.join(
                os.path.dirname(base_path),
                subset.location.lstrip("/"),
            )
        else:
            # Used for mounted data (production)
            subset_path = subset.location

        return subset_path

    @staticmethod
    @st.cache_data
    def _load_dask_dataframe(subset_path, fields):
        return dd.read_parquet(subset_path, columns=list(fields.keys()))

    # TODO: change later to accept range of partitions
    @staticmethod
    def _get_partition_to_load(dask_df: dd.DataFrame) -> t.Union[int, None]:
        """Get the partition of the dataframe to load from a slider."""
        partition = None

        if dask_df.npartitions > 1:
            if st.session_state["partition"] is None:
                starting_value = 0
            else:
                starting_value = st.session_state["partition"]

            partition = st.slider("partition", 1, dask_df.npartitions, starting_value)
            st.session_state["partition"] = partition

        return partition

    @staticmethod
    def _get_dataframe_partition(
        dask_df: dd.DataFrame,
        partition: t.Union[int, None],
    ) -> dd.DataFrame:
        """Get the partition of the dataframe to load."""
        if partition is not None:
            return dask_df.get_partition(partition)

        return dask_df

    def create_loader_widget(self):
        """
        Provides common widgets for loading a dataframe and selecting a partition to load. Uses
        Cached dataframes to avoid reloading the dataframe when changing the partition.

        Returns:
            Dataframe and fields
        """
        manifest, subset, fields = self._get_manifest_fields_and_subset()
        subset_path = self._get_subset_path(manifest, subset)
        df = self._load_dask_dataframe(subset_path, fields)
        partition = self._get_partition_to_load(df)
        dataframe = self._get_dataframe_partition(df, partition)

        return dataframe, fields
