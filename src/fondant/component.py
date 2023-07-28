"""This module defines interfaces which components should implement to be executed by fondant."""

import typing as t

import dask.dataframe as dd
import pandas as pd

from fondant.component_spec import ComponentSpec


class BaseComponent:
    """Base interface for each component, specifying only the constructor.

    Args:
        spec: The specification of the component
        **kwargs: The provided user arguments are passed in as keyword arguments
    """

    def __init__(self, spec: ComponentSpec, **kwargs):
        pass


class DaskLoadComponent(BaseComponent):
    """Component that loads data and returns a Dask DataFrame."""

    def load(self) -> dd.DataFrame:
        raise NotImplementedError


class DaskTransformComponent(BaseComponent):
    """Component that transforms an incoming Dask DataFrame."""

    def transform(self, dataframe: dd.DataFrame) -> dd.DataFrame:
        """
        Abstract method for applying data transformations to the input dataframe.

        Args:
            dataframe: A Dask dataframe containing the data specified in the `consumes` section
                of the component specification
        """
        raise NotImplementedError


class DaskWriteComponent(BaseComponent):
    """Component that accepts a Dask DataFrame and writes its contents."""

    def write(self, dataframe: dd.DataFrame) -> None:
        raise NotImplementedError


class PandasTransformComponent(BaseComponent):
    """Component that transforms the incoming dataset partition per partition as a pandas
    DataFrame.
    """

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method for applying data transformations to the input dataframe.
        Called once for each partition of the data.

        Args:
            dataframe: A Pandas dataframe containing a partition of the data
        """
        raise NotImplementedError


Component = t.TypeVar("Component", bound=BaseComponent)
"""Component type which can represents any of the subclasses of BaseComponent"""
