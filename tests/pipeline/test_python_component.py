import textwrap

import dask.dataframe as dd
import pandas as pd
import pyarrow as pa
from fondant.component import DaskLoadComponent, PandasTransformComponent
from fondant.pipeline import Pipeline, lightweight_component


def test_build_python_script():
    @lightweight_component()
    class CreateData(DaskLoadComponent):
        def load(self) -> dd.DataFrame:
            df = pd.DataFrame(
                {
                    "x": [1, 2, 3],
                    "y": [4, 5, 6],
                },
                index=pd.Index(["a", "b", "c"], name="id"),
            )
            return dd.from_pandas(df, npartitions=1)

    assert CreateData.image().script == textwrap.dedent(
        """\
        from typing import *
        import typing as t

        import dask.dataframe as dd
        import fondant
        import pandas as pd
        from fondant.component import *
        from fondant.core import *


        class CreateData(DaskLoadComponent):
            def load(self) -> dd.DataFrame:
                df = pd.DataFrame(
                    {
                        "x": [1, 2, 3],
                        "y": [4, 5, 6],
                    },
                    index=pd.Index(["a", "b", "c"], name="id"),
                )
                return dd.from_pandas(df, npartitions=1)
    """,
    )


def test_lightweight_component(tmp_path_factory):
    pipeline = Pipeline(
        name="dummy-pipeline",
        base_path="./data",
    )

    @lightweight_component(
        base_image="python:3.8-slim-buster",
        extra_requires=["pandas", "dask"],
    )
    class CreateData(DaskLoadComponent):
        def load(self) -> dd.DataFrame:
            df = pd.DataFrame(
                {
                    "x": [1, 2, 3],
                    "y": [4, 5, 6],
                },
                index=pd.Index(["a", "b", "c"], name="id"),
            )
            return dd.from_pandas(df, npartitions=1)

    dataset = pipeline.read(
        ref=CreateData,
        produces={"x": pa.int32(), "y": pa.int32()},
    )

    @lightweight_component()
    class AddN(PandasTransformComponent):
        def __init__(self, n: int, **kwargs):
            self.n = n

        def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
            dataframe["x"] = dataframe["x"].map(lambda x: x + self.n)
            return dataframe

    _ = dataset.apply(
        ref=AddN,
        produces={"x": pa.int32(), "y": pa.int32()},
        consumes={"x": pa.int32(), "y": pa.int32()},
        arguments={"n": 1},
    )
