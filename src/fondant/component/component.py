"""This module defines interfaces which components should implement to be executed by fondant."""

import logging
import typing as t
from abc import abstractmethod

import dask.dataframe as dd
import pandas as pd
import pyarrow as pa

from fondant.core.schema import Field, Type

logger = logging.getLogger(__name__)


class BaseComponent:
    """Base interface for each component, specifying only the constructor.

    Args:
        consume: The schema the component should consume
        produces: The schema the component should produce
        **kwargs: The provided user arguments are passed in as keyword arguments
    """

    def __init__(
        self,
        **kwargs,
    ):
        self.consumes = None
        self.produces = None

    def teardown(self) -> None:
        """Method called after the component has been executed."""

    @staticmethod
    def modify_spec_consumes(
        spec_consumes: t.Dict[str, t.Any],
        apply_consumes: t.Optional[t.Dict[str, pa.DataType]],
    ):
        """Modify fields based on the consumes argument in the 'apply' method."""
        if apply_consumes:
            for k, v in apply_consumes.items():
                if isinstance(v, str):
                    spec_consumes[k] = spec_consumes.pop(v)
                else:
                    msg = (
                        f"Invalid data type for field `{k}` in the `apply_consumes` "
                        f"argument. Only string types are allowed."
                    )
                    raise ValueError(
                        msg,
                    )
        return spec_consumes

    @staticmethod
    def get_spec_consumes(
        component_consumes: t.Dict[str, t.Any],
        dataset_fields: t.Mapping[str, Field],
        apply_consumes: t.Optional[t.Dict[str, t.Any]] = None,
    ):
        """
        Function that get the consumes spec for the component based on the dataset fields and
        the apply_consumes argument.

        Args:
            component_consumes: Component spec consumes.
            dataset_fields: The fields of the dataset.
            apply_consumes: The consumes argument in the apply method.

        Returns:
            The consumes spec for the component.
        """
        if dataset_fields and component_consumes is None:
            # Get consumes spec from the dataset
            spec_consumes = {k: v.type.to_dict() for k, v in dataset_fields.items()}

            spec_consumes = BaseComponent.modify_spec_consumes(
                spec_consumes,
                apply_consumes,
            )

            logger.warning(
                "No consumes defined. Consumes will be inferred from the dataset."
                " All field will be consumed which may lead to additional computation,"
                " Consider defining consumes in the component.\n Consumes: %s",
                spec_consumes,
            )
            return None

        elif dataset_fields:
            spec_consumes = {
                k: (Type(v).to_dict() if k != "additionalProperties" else v)
                for k, v in component_consumes.items()
            }
            return spec_consumes
        else:
            return {}

    def get_spec_produces(self) -> None:
        pass


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


Component = t.TypeVar("Component", bound=BaseComponent)
"""Component type which can represents any of the subclasses of BaseComponent"""
