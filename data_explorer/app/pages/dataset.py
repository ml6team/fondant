"""Data exploration page of the app."""
import base64
import hashlib
import json
import typing as t

import dask.dataframe as dd
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from bs4 import BeautifulSoup
from df_helpers.fields import get_image_fields, get_numeric_fields, get_string_fields
from df_helpers.image_render import configure_image_builder, convert_image_column
from fondant.core.schema import Field
from fpdf import FPDF
from interfaces.common import render_common_interface
from interfaces.dataset_interface import DatasetLoaderApp
from interfaces.sidebar import render_sidebar
from st_aggrid import AgGrid, AgGridReturn, ColumnsAutoSizeMode, GridOptionsBuilder


def is_html(text: str):
    return bool(BeautifulSoup(text, "html.parser").find())


def is_json(text: str):
    try:
        json_object = json.loads(text)
        return bool(isinstance(json_object, (dict, list)))
    except ValueError:
        return False


def is_pdf_base64(text: str):
    try:
        _bytes = base64.b64decode(text)
        return _bytes[0:4] == b"%PDF"
    except ValueError:
        return False


def create_pdf_from_text(raw_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(200, 10, txt=raw_text)
    return pdf.output(dest="S").encode("latin-1")


class DatasetExplorerApp(DatasetLoaderApp):
    @staticmethod
    def setup_app_page(dataframe: pd.DataFrame, fields) -> AgGridReturn:
        """Build the dataframe explorer table."""
        image_fields = get_image_fields(fields)

        for field in image_fields:
            dataframe = convert_image_column(dataframe, field)

        # TODO: add formatting for other datatypes?

        # configure builder
        options_builder = GridOptionsBuilder.from_dataframe(dataframe)

        # Add tooltip hover for all fields
        for field in fields:
            if field not in image_fields:
                options_builder.configure_column(
                    field=field,
                    tooltipField=field,
                    max_width=400,
                )

        grid_options: t.Dict[str, t.Any] = {"rowWidth": "auto", "tooltipShowDelay": 500}

        if len(image_fields) > 0:
            grid_options["rowHeight"] = 100
            options_builder.configure_grid_options(**grid_options)
        else:
            grid_options["rowHeight"] = "auto"
            options_builder.configure_grid_options(**grid_options)

        # format the image columns
        for field in image_fields:
            configure_image_builder(options_builder, field)

        options_builder.configure_default_column(
            editable=False,
            groupable=True,
            wrapText=True,
            resizable=True,
            filterable=True,
            sortable=True,
        )
        options_builder.configure_selection(
            selection_mode="single",
            use_checkbox=False,
            pre_selected_rows=[0],
        )

        # display the Ag Grid table
        return AgGrid(
            dataframe,
            gridOptions=options_builder.build(),
            allow_unsafe_jscode=True,
            columns_auto_size_mode=ColumnsAutoSizeMode.FIT_ALL_COLUMNS_TO_VIEW,
        )

    @staticmethod
    def render_text(text: str):
        if is_html(text):
            st.text("HTML detected")
            render = st.checkbox("Render HTML")
            if render:
                components.html(text, height=600)
            else:
                st.code(text, language="html")

        elif is_json(text):
            st.text("JSON detected")
            st.json(json.loads(text))

        elif is_pdf_base64(text):
            st.text("PDF detected")
            pdf_data = create_pdf_from_text(text)
            encoded_pdf = base64.b64encode(pdf_data).decode("utf-8")
            data = (
                f'<embed src="data:application/pdf;base64,{encoded_pdf}" width="100%" '
                f'height="1000" type="application/pdf">'
            )
            st.markdown(data, unsafe_allow_html=True)
        else:
            st.markdown(text)

    def setup_viewer_widget(
        self,
        grid_dict: AgGridReturn,
        fields: t.Dict[str, t.Any],
        extra_text_fields: t.Optional[t.List[str]],
    ):
        """Setup the viewer widget. This widget allows the user to view the selected row in the
        dataframe.
        """
        text_fields = get_string_fields(fields)

        if extra_text_fields:
            text_fields.extend(extra_text_fields)

        if text_fields:
            st.markdown("### Document Viewer")
            selected_column = st.selectbox("View column", text_fields)
            if grid_dict["selected_rows"]:
                data = str(grid_dict["selected_rows"][0][selected_column])
                self.render_text(data)
            else:
                st.info("No text fields found in dataframe")

    @staticmethod
    def search_df(
        *,
        df: dd.DataFrame,
        search_field: str,
        search_value: str,
        exact_search: bool,
        selected_fields: t.Dict[str, Field],
    ) -> dd.DataFrame:
        """Search the dataframe for the given search field and search value."""
        if st.session_state.search:
            if exact_search:
                field_type = selected_fields[search_field].type.name
                if "int" in field_type:
                    search_value = int(search_value)
                elif "float" in field_type:
                    search_value = float(search_value)
                return df[df[search_field] == search_value]

            return df[df[search_field].str.contains(search_value)]
        return None

    def setup_search_widget(
        self,
        df: dd.DataFrame,
        selected_fields: t.Dict[str, Field],
        field_mapping: t.Dict[str, str],
    ) -> t.Tuple[dd.DataFrame, str]:
        """Setup the viewer widget. This widget allows the user to view the selected row in the
        dataframe.
        """

        # Functions for handling button clicks
        def search_button():
            st.session_state.search = True

        def result_found():
            st.session_state.result_found = True

        # Initializing session states if not present
        if "search" not in st.session_state or st.session_state.get("clear"):
            st.session_state.search = False

        if "search_value" not in st.session_state:
            st.session_state.search_value = ""

        if "result_found" not in st.session_state:
            st.session_state.result_found = True

        # Get numerical columns
        get_numeric_fields(selected_fields)
        filter_cache_key = ""

        # Creating columns for layout
        col_1, col_2, col_3, col_4, col_5 = st.columns([0.2, 0.2, 0.1, 0.1, 0.1])

        # Widgets for user input
        with col_1:
            search_value = st.text_input(
                ":mag: Search Value",
                placeholder="Enter search value",
                disabled=st.session_state.search,
                value=st.session_state.search_value,
            )

        with col_2:
            search_field = st.selectbox(
                "Search Field",
                list(selected_fields.keys()),
            )
        with col_3:
            st.text("")
            st.text("")
            exact_search = st.checkbox(
                "Exact match",
                list(selected_fields.keys()),
                help="Toggle to search for exact matches, otherwise search for partial matches."
                "Note that searching for partial matches may take longer.",
            )

        # Buttons for initiating and clearing search
        with col_4:
            st.text("")
            st.button(
                "Search",
                on_click=search_button,
                use_container_width=True,
                disabled=st.session_state.search,
            )

        with col_5:
            st.text("")
            st.button(
                "Clear",
                on_click=result_found,
                key="clear",
                disabled=not st.session_state.search,
                use_container_width=True,
            )

        # Perform search if activated
        if st.session_state.search and st.session_state.result_found is not False:
            if not search_value:
                st.warning("Please enter a search value")
            else:
                st.session_state.search_value = search_value
                # Generate cache key for filtering
                filter_cache_key = hashlib.md5(  # nosec
                    f"{search_field}_{search_value}_{exact_search}".encode(),
                ).hexdigest()
                df = self.search_df(
                    df=df,
                    search_field=search_field,
                    search_value=search_value,
                    exact_search=exact_search,
                    selected_fields=selected_fields,
                )

                # Display warning if no results found
                if len(df) == 0:
                    st.session_state.result_found = False
                else:
                    st.session_state.result_found = True

        if st.session_state.result_found is False:
            st.warning("No results found, click clear to search again")

        return df, filter_cache_key


render_sidebar()
render_common_interface()

app = DatasetExplorerApp()
field_mapping, selected_fields = app.get_fields_mapping()
dask_df = app.load_dask_dataframe(field_mapping)
dask_df, cache_key = app.setup_search_widget(dask_df, selected_fields, field_mapping)
if st.session_state.result_found is True:
    loaded_df, additional_text_fields = app.load_pandas_dataframe(
        dask_df=dask_df,
        field_mapping=field_mapping,
        cache_key=cache_key,
        selected_fields=selected_fields,
    )
    grid_data_dict = app.setup_app_page(loaded_df, selected_fields)
    app.setup_viewer_widget(grid_data_dict, selected_fields, additional_text_fields)
