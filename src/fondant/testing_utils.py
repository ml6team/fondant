import dask.dataframe as dd
import pandas as pd

from fondant.component import DaskTransformComponent, PandasTransformComponent


def execute_pandas_transform_component(
    component: PandasTransformComponent,
    input_dataframe: pd.DataFrame,
    expected_output: pd.DataFrame,
):
    """Helper method for executing pandas transform component."""
    _compare_pandas_dataframe(component.transform(input_dataframe), expected_output)


def _compare_pandas_dataframe(
    expected_output: pd.DataFrame,
    output_dataframe: pd.DataFrame,
):
    """Comparing to pandas dataframes."""
    pd.testing.assert_frame_equal(
        left=expected_output,
        right=output_dataframe,
        check_dtype=False,
    )


def execute_dask_transform_component(
    component: DaskTransformComponent,
    input_dataframe: dd,
    expected_output: dd,
):
    """Helper method for executing pandas transform component."""
    _compare_dask_dataframe(component.transform(input_dataframe), expected_output)


def _compare_dask_dataframe(expected_output: dd, output_dataframe: dd):
    msg = "Not implemented."
    raise NotImplementedError(msg)
