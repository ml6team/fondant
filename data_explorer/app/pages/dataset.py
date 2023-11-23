"""Data exploration page of the app."""
import typing as t

import streamlit as st
from df_helpers.fields import get_image_fields
from df_helpers.image_render import configure_image_builder, convert_image_column
from interfaces.dataset_interface import DatasetLoaderApp
from st_aggrid import AgGrid, ColumnsAutoSizeMode, GridOptionsBuilder, JsCode


class DatasetExplorerApp(DatasetLoaderApp):
    @staticmethod
    def _get_hover_vis_script(field: str) -> JsCode:
        """Get the javascript code for the hover visualization."""
        code = f"""
        class CustomTooltip {{
            eGui;
            init(params) {{
                const eGui = (this.eGui = document.createElement('div'));
                const color = params.color || 'black';
                const data = params.api.getDisplayedRowAtIndex(params.rowIndex).data;
                eGui.classList.add('custom-tooltip');
                //@ts-ignore
                eGui.style['background-color'] = color;
                eGui.style['color'] = 'white';
                eGui.style['padding'] = "5px 5px 5px 5px";
                eGui.style['font-size'] = "15px";
                eGui.style['border-style'] = 'double';
                this.eGui.innerText = data.{field};
            }}
            getGui() {{
                return this.eGui;
            }}
        }}
        """
        return JsCode(code)

    def setup_app_page(self, dataframe, fields):
        """Build the dataframe explorer table."""
        image_fields = get_image_fields(fields)

        # get the first rows of the dataframe
        cols = st.columns(2)
        with cols[0]:
            rows = st.slider(
                "Dataframe rows to load",
                1,
                len(dataframe),
                min(len(dataframe), 20),
            )
        with cols[1]:
            rows_per_page = st.slider("Amount of rows per page", 5, 50, 10)

        dataframe_explorer = dataframe.head(rows).reset_index(drop=False)

        for field in image_fields:
            dataframe_explorer = convert_image_column(dataframe_explorer, field)

        # TODO: add formatting for other datatypes?

        # configure builder
        options_builder = GridOptionsBuilder.from_dataframe(dataframe_explorer)

        # Add tooltip hover for all fields
        for field in fields:
            vis_js_script = self._get_hover_vis_script(field)
            if field not in image_fields:
                options_builder.configure_column(
                    field=field,
                    tooltipField=field,
                    max_width=400,
                    tooltipComponent=vis_js_script,
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
            paginationPageSize=rows_per_page,
            paginationAutoPageSize=False,
        )
        options_builder.configure_side_bar()

        # display the Ag Grid table
        AgGrid(
            dataframe_explorer,
            gridOptions=options_builder.build(),
            allow_unsafe_jscode=True,
            columns_auto_size_mode=ColumnsAutoSizeMode.FIT_ALL_COLUMNS_TO_VIEW,
        )


app = DatasetExplorerApp()
app.create_common_interface()
df, df_fields = app.create_loader_widget()
app.setup_app_page(df, df_fields)
