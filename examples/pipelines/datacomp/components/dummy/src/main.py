"""
Dummy component for debugging.
"""
import logging

import dask.dataframe as dd

from fondant.component import DaskTransformComponent
from fondant.executor import DaskTransformExecutor

logger = logging.getLogger(__name__)


class DummyComponent(DaskTransformComponent):
    """Component that downloads images based on URLs."""

    def transform(self, dataframe: dd.DataFrame) -> dd.DataFrame:

        logger.info(f"Length of the dataframe: {len(dataframe)}")
        print("Columns of the dataframe:", dataframe.columns)
        print("Dyptes of the dataframe:", dataframe.dtypes)

        return dataframe


if __name__ == "__main__":
    executor = DaskTransformExecutor.from_args()
    executor.execute(DummyComponent)