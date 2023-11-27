"""Data exploration page of the app."""
import base64
import json
import typing as t

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from bs4 import BeautifulSoup
from config import ROWS_PER_PAGE
from df_helpers.fields import get_image_fields, get_string_fields
from df_helpers.image_render import configure_image_builder, convert_image_column
from fpdf import FPDF
from interfaces.dataset_interface import DatasetLoaderApp
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

        # configure pagination and sidebar
        options_builder.configure_pagination(
            enabled=True,
            paginationPageSize=ROWS_PER_PAGE,
            paginationAutoPageSize=False,
        )
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

    def setup_viewer_widget(self, grid_dict: AgGridReturn, fields: t.Dict[str, t.Any]):
        """Setup the viewer widget. This widget allows the user to view the selected row in the
        dataframe.
        """
        text_fields = get_string_fields(fields)
        with st.expander("Document Viewer"):
            if text_fields:
                selected_column = st.selectbox("View column", text_fields)
                if grid_dict["selected_rows"]:
                    data = str(grid_dict["selected_rows"][0][selected_column])
                    self.render_text(data)
            else:
                st.info("No text fields found in dataframe")


app = DatasetExplorerApp()
app.create_common_interface()
df, df_fields = app.load_pandas_dataframe()
grid_data_dict = app.setup_app_page(df, df_fields)
app.setup_viewer_widget(grid_data_dict, df_fields)
