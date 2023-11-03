"""This component filters the dataset based on filtering conditions."""
import logging
import typing as t

import pandas as pd
from fondant.component import DaskTransformComponent

logger = logging.getLogger(__name__)


class FilterDataset(DaskTransformComponent):
    """Component that filters images based on height and width."""

    def __init__(self, filtering_conditions: t.List[t.Tuple]) -> None:
        """
        Args:
            filtering_conditions: the filtering conditions to apply to the dataframe.
        """
        self.filtering_conditions = filtering_conditions

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        for condition in self.filtering_conditions:
            column_name, op, value = condition
            if op == "in":
                dataframe = dataframe[dataframe[column_name].isin(value)]
            elif op == "not in":
                dataframe = dataframe[~dataframe[column_name].isin(value)]
            elif op == "contains":
                dataframe = dataframe[
                    dataframe[column_name].str.contains(value, na=False)
                ]
            elif op == "==":
                dataframe = dataframe[dataframe[column_name] == value]
            elif op == "!=":
                dataframe = dataframe[dataframe[column_name] != value]
            elif op == ">":
                dataframe = dataframe[dataframe[column_name] > value]
            elif op == "<":
                dataframe = dataframe[dataframe[column_name] < value]
            elif op == ">=":
                dataframe = dataframe[dataframe[column_name] >= value]
            elif op == "<=":
                dataframe = dataframe[dataframe[column_name] <= value]

        return dataframe
