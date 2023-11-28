"""Dataset interface for the data explorer app."""

import os

import dask.dataframe as dd
import pandas as pd
import streamlit as st
from config import ROWS_TO_RETURN
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

    def load_dask_dataframe(self):
        """Loads a Dask DataFrame from a subset of the dataset."""
        manifest, subset, fields = self._get_manifest_fields_and_subset()
        subset_path = self._get_subset_path(manifest, subset)
        dask_df = dd.read_parquet(subset_path, columns=list(fields.keys())).reset_index(
            drop=False,
        )

        return dask_df, fields

    @staticmethod
    def get_pandas_from_dask(
        dask_df: dd.DataFrame,
        rows_to_return: int,
        partition_index: int,
        rows_from_last_partition: int,
    ):
        """
        Converts a Dask DataFrame into a Pandas DataFrame with specified number of rows.

        Args:
         dask_df: Input Dask DataFrame.
         rows_to_return: Number of rows needed in the resulting Pandas DataFrame.
         partition_index: Index of the partition to start from.
         rows_from_last_partition: Number of rows to take from the last partition.

        Returns:
         result_df: Pandas DataFrame with the specified number of rows.
         last_partition_index: Index of the last used partition.
         rows_from_last_partition: Number of rows taken from the last partition.
        """
        rows_returned = 0
        data_to_return = []

        dask_df = dask_df.partitions[partition_index:]

        for partition_index, partition in enumerate(
            dask_df.partitions,
            start=partition_index,
        ):
            # Materialize partition as a pandas DataFrame
            partition_df = partition.compute()
            partition_df = partition_df[rows_from_last_partition:]

            # Check if adding this partition exceeds the required rows
            if rows_returned + len(partition_df) <= rows_to_return:
                data_to_return.append(partition_df)
                rows_returned += len(partition_df)
            else:
                # Calculate how many rows to take from this partition
                rows_from_last_partition = rows_to_return - rows_returned
                partition_df = partition_df.head(rows_from_last_partition)
                data_to_return.append(partition_df)
                break

        # Concatenate the selected partitions into a single pandas DataFrame
        df = pd.concat(data_to_return)

        return df, partition_index, rows_from_last_partition

    @staticmethod
    def _initialize_page_view_dict():
        return st.session_state.get(
            "page_view_dict",
            {
                0: {
                    "start_index": 0,
                    "start_partition": 0,
                },
            },
        )

    @staticmethod
    def _update_page_view_dict(
        page_view_dict,
        page_index,
        start_index,
        start_partition,
    ):
        page_view_dict[page_index] = {
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

        # Get the manifest, subset, and fields
        dask_df, fields = self.load_dask_dataframe()

        # Initialize page view dict if it doesn't exist
        page_index = st.session_state.get("page_index", 0)
        page_view_dict = self._initialize_page_view_dict()

        # Get the starting index and partition for the current page
        start_index = page_view_dict[page_index]["start_index"]
        start_partition = page_view_dict[page_index]["start_partition"]

        pandas_df, next_partition, next_index = self.get_pandas_from_dask(
            dask_df,
            ROWS_TO_RETURN,
            start_index,
            start_partition,
        )
        self._update_page_view_dict(
            page_view_dict,
            page_index + 1,
            next_index,
            next_partition,
        )

        st.info(
            f"Showing {len(pandas_df)} rows. Click on the 'next' and 'previous' "
            f"buttons to navigate through the dataset.",
        )
        previous_col, _, next_col = st.columns([0.2, 0.6, 0.2])

        if page_index != 0:
            previous_button_disabled = False

        if len(pandas_df) < ROWS_TO_RETURN:
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

        return pandas_df, fields
