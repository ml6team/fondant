"""Dataset interface for the data explorer app."""

import os
import typing as t
from collections import defaultdict

import dask.dataframe as dd
import pandas as pd
import streamlit as st
from config import DEFAULT_INDEX_NAME, ROWS_TO_RETURN
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

        available_components = ["chunk_text"]
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

    def get_fields_mapping(self):
        field_mapping = defaultdict(list)

        manifest, selected_fields = self._get_manifest_fields()
        # Add index field to field mapping to guarantee start reading with the index dataframe
        index_location = self._get_field_location(manifest, DEFAULT_INDEX_NAME)
        field_mapping[index_location].append(DEFAULT_INDEX_NAME)

        for field_name, field in selected_fields.items():
            field_location = self._get_field_location(manifest, field_name)
            field_mapping[field_location].append(field_name)

        return field_mapping, selected_fields

    @staticmethod
    @st.cache_data
    def load_dask_dataframe(field_mapping):
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

    @staticmethod
    @st.cache_data
    def get_pandas_from_dask(
        field_mapping,
        _dask_df: dd.DataFrame,
        rows_to_return: int,
        partition_index: int,
        partition_row_index: int,
    ):
        """
        Converts a Dask DataFrame into a Pandas DataFrame with specified number of rows.

        Args:
         field_mapping: Mapping of fields to their location. Used as a unique cache key.
         _dask_df: Input Dask DataFrame.
         rows_to_return: Number of rows needed in the resulting Pandas DataFrame.
         partition_index: Index of the partition to start from.
         partition_row_index: Index from the last partition.

        Returns:
         result_df: Pandas DataFrame with the specified number of rows.
         partition_row_index: Index of the last used partition.
        """
        rows_returned = 0
        data_to_return = []

        dask_df = _dask_df.partitions[partition_index:]

        for partition_index, partition in enumerate(
            dask_df.partitions,
            start=partition_index,
        ):
            # Materialize partition as a pandas DataFrame
            partition_df = partition.compute().reset_index(drop=False)
            partition_df = partition_df[partition_row_index:]
            partition_length = len(partition_df)

            # Check if adding this partition exceeds the required rows
            if rows_returned + partition_length <= rows_to_return:
                data_to_return.append(partition_df)
                rows_returned += partition_length
                partition_row_index = 0
            else:
                # Calculate how many rows to take from this partition
                rows_to_take = rows_to_return - rows_returned
                sliced_partition_df = partition_df.head(rows_to_take)

                # Check if the partition is empty
                if len(sliced_partition_df) == 0:
                    partition_row_index = 0
                    continue

                data_to_return.append(sliced_partition_df)
                rows_returned += len(sliced_partition_df)

                # Check if we have reached the required number of rows
                if rows_returned >= rows_to_return or len(sliced_partition_df) == 0:
                    partition_row_index = partition_row_index + len(
                        sliced_partition_df,
                    )
                    break

                # Check if the last row of the sliced partition is the same as the last row of
                # the original partition. If so, we have reached the end of the dataframe.
                if partition_df.tail(1).equals(sliced_partition_df.tail(1)):
                    partition_row_index = 0

        # Concatenate the selected partitions into a single pandas DataFrame
        df = pd.concat(data_to_return)

        return df, partition_index, partition_row_index

    @staticmethod
    def _initialize_page_view_dict(component):
        page_view_dict = st.session_state.get("page_view_dict", {})

        if component not in page_view_dict:
            page_view_dict[component] = {
                0: {
                    "start_index": 0,
                    "start_partition": 0,
                },
            }
            st.session_state["page_index"] = 0

        return page_view_dict

    @staticmethod
    def _update_page_view_dict(
        *,
        page_view_dict,
        page_index,
        start_index,
        start_partition,
        component,
    ):
        page_view_dict[component][page_index] = {
            "start_index": start_index,
            "start_partition": start_partition,
        }
        st.session_state["page_view_dict"] = page_view_dict

        return page_view_dict

    def load_pandas_dataframe(self):
        """
        Provides common widgets for loading a dataframe and selecting a partition to load. Uses
        Cached dataframes to avoid reloading the dataframe when changing the partition.

        Returns:
            Dataframe and fields
        """
        previous_button_disabled = True
        next_button_disabled = False

        # Get field mapping from manifest and selected fields
        field_mapping, selected_fields = self.get_fields_mapping()

        # Get the manifest, subset, and fields
        dask_df = self.load_dask_dataframe(field_mapping)

        # Initialize page view dict if it doesn't exist
        component = st.session_state["component"]
        page_view_dict = self._initialize_page_view_dict(component)
        page_index = st.session_state.get("page_index", 0)

        # Get the starting index and partition for the current page
        start_partition = page_view_dict[component][page_index]["start_partition"]
        start_index = page_view_dict[component][page_index]["start_index"]

        pandas_df, partition_index, partition_row_index = self.get_pandas_from_dask(
            field_mapping=field_mapping,
            _dask_df=dask_df,
            rows_to_return=ROWS_TO_RETURN,
            partition_index=start_partition,
            partition_row_index=start_index,
        )

        self._update_page_view_dict(
            page_view_dict=page_view_dict,
            page_index=page_index + 1,
            start_partition=partition_index,
            start_index=partition_row_index,
            component=component,
        )

        st.info(
            f"Showing {len(pandas_df)} rows. Click on the 'next' and 'previous' "
            f"buttons to navigate through the dataset.",
        )
        previous_col, _, next_col = st.columns([0.2, 0.6, 0.2])

        if page_index != 0:
            previous_button_disabled = False

        if partition_index == dask_df.npartitions - 1:
            # Check if the last row of the last partition is the same as the last row of the
            # dataframe. If so, we have reached the end of the dataframe.
            partition_index = dask_df.tail(1).index[0]
            dask_df_index = pandas_df.tail(1)[DEFAULT_INDEX_NAME].iloc[0]
            if partition_index == dask_df_index:
                next_button_disabled = True

        if previous_col.button(
            "⏮️ Previous",
            use_container_width=True,
            disabled=previous_button_disabled,
        ):
            st.session_state["page_index"] = page_index - 1
            st.rerun()

        if next_col.button(
            "Next ⏭️",
            use_container_width=True,
            disabled=next_button_disabled,
        ):
            st.session_state["page_index"] = page_index + 1
            st.rerun()

        st.markdown(f"Page {page_index}")

        return pandas_df, selected_fields
