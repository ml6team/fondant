"""This module defines interfaces which components should implement to be executed by fondant."""

import typing as t
from abc import abstractmethod

import dask.dataframe as dd
import pandas as pd
import pandera

from fondant.core.schema import Field


class BaseComponent:
    """Base interface for each component, specifying only the constructor.

    Args:
        consumes: The schema the component should consume
        produces: The schema the component should produce
        **kwargs: The provided user arguments are passed in as keyword arguments
    """

    def __init__(
        self,
        *,
        consumes: t.Mapping[str, Field],
        produces: t.Mapping[str, Field],
        **kwargs,
    ):
        pass

    def teardown(self) -> None:
        """Method called after the component has been executed."""


class DaskLoadComponent(BaseComponent):
    """Component that loads data and returns a Dask DataFrame."""

    @abstractmethod
    def load(self) -> dd.DataFrame:
        pass


class DaskTransformComponent(BaseComponent):
    """Component that transforms an incoming Dask DataFrame."""

    @abstractmethod
    def transform(self, dataframe: dd.DataFrame) -> dd.DataFrame:
        """
        Abstract method for applying data transformations to the input dataframe.

        Args:
            dataframe: A Dask dataframe containing the data specified in the `consumes` section
                of the component specification
        """


class DaskWriteComponent(BaseComponent):
    """Component that accepts a Dask DataFrame and writes its contents."""

    @abstractmethod
    def write(self, dataframe: dd.DataFrame) -> None:
        pass


class PandasTransformComponent(BaseComponent):
    """Component that transforms the incoming dataset partition per partition as a pandas
    DataFrame.
    """

    @abstractmethod
    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method for applying data transformations to the input dataframe.
        Called once for each partition of the data.

        Args:
            dataframe: A Pandas dataframe containing a partition of the data
        """

    @classmethod
    def resolve_produces(cls, consumes):
        """Simulate a transform function and collect schema information."""
        # TODO: Convert consumes pyarrow dict into pandera types dict
        input_schema = pandera.DataFrameSchema(
            {
                "context_precision": pandera.Column(float),
                "context_relevancy": pandera.Column(float),
            },
        )

        input_df = input_schema.example(size=2)
        output_df = cls(consumes={}, produces={}).transform(dataframe=input_df)
        output_schema = pandera.infer_schema(output_df)

        # TODO: convert schema into pyarrow types and produces dict before returning
        return output_schema  # noqa RET504


Component = t.TypeVar("Component", bound=BaseComponent)
"""Component type which can represents any of the subclasses of BaseComponent"""
