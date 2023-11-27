"""Dataset interface for the data explorer app."""

import os
import typing as t
from collections import defaultdict

import dask.dataframe as dd
import streamlit as st
from config import DEFAULT_INDEX_NAME
from fondant.core.manifest import Manifest
from fondant.core.schema import Field
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

    def _get_manifest_fields(self) -> t.Tuple[Manifest, t.Dict[str, Field]]:
        """Get fields from manifest and store them in session state."""
        cols = st.columns(2)

        with cols[0]:
            selected_component_path = self._select_component()

        manifest_path = os.path.join(selected_component_path, "manifest.json")
        manifest = Manifest.from_file(manifest_path)
        fields = manifest.fields
        field_list = list(fields.keys())

        with cols[1]:
            selected_field_names = st.multiselect(
                "Fields",
                field_list,
                default=field_list,
            )

        selected_fields = {
            field_name: fields[field_name] for field_name in selected_field_names
        }

        return manifest, selected_fields

    def _get_field_location(self, manifest: Manifest, field_name: str) -> str:
        """
        Get path to the fields from manifest. If the base path is not mounted, the fields path are
        assumed to be relative to the base path. If the base path is mounted, the fields path are
        assumed to be absolute.
        """
        base_path = st.session_state["base_path"]
        field_location = manifest.get_field_location(field_name)

        if (
            os.path.ismount(base_path) is False
            and self.fs.__class__.__name__ == "LocalFileSystem"
        ):
            # Used for local development when running the app locally
            field_location = os.path.join(
                os.path.dirname(base_path),
                field_location.lstrip("/"),
            )

        return field_location

    def get_fields_mapping(
        self,
        manifest: Manifest,
        selected_fields: t.Dict[str, Field],
    ) -> defaultdict[t.Any, list]:
        field_mapping = defaultdict(list)

        # Add index field to field mapping to guarantee start reading with the index dataframe
        index_location = self._get_field_location(manifest, DEFAULT_INDEX_NAME)
        field_mapping[index_location].append(DEFAULT_INDEX_NAME)

        for field_name, field in selected_fields.items():
            field_location = self._get_field_location(manifest, field_name)
            field_mapping[field_location].append(field_name)

        return field_mapping

    @staticmethod
    @st.cache_data
    def _load_dask_dataframe(field_mapping):
        dataframe = None
        for location, fields in field_mapping.items():
            if DEFAULT_INDEX_NAME in fields:
                fields.remove(DEFAULT_INDEX_NAME)

            partial_df = dd.read_parquet(
                location,
                columns=fields,
                index=DEFAULT_INDEX_NAME,
                calculate_divisions=True,
            )

            if dataframe is None:
                # ensure that the index is set correctly and divisions are known.
                dataframe = partial_df
            else:
                dataframe = dataframe.merge(
                    partial_df,
                    how="left",
                    left_index=True,
                    right_index=True,
                )
        return dataframe

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
        manifest, selected_fields = self._get_manifest_fields()
        # Get field mapping from manifest and selected fields
        field_mapping = self.get_fields_mapping(manifest, selected_fields)

        dataframe = self._load_dask_dataframe(field_mapping)
        partition = self._get_partition_to_load(dataframe)
        dataframe = self._get_dataframe_partition(dataframe, partition)

        return dataframe, selected_fields
