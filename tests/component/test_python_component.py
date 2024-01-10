import dask.dataframe as dd
import pandas as pd
import pyarrow as pa
from fondant.component import DaskLoadComponent, python_component
from fondant.pipeline import Pipeline


def test_python_component():
    pipeline = Pipeline(name="dummy-pipeline", base_path="./data")

    @python_component(
        base_image="python:3.8-slim-buster",
        extra_requires=["pandas", "dask"],
    )
    class CreateData(DaskLoadComponent):
        def __init__(self, *_, **__):
            pass

        def load(self) -> dd.DataFrame:
            df = pd.DataFrame(
                {
                    "x": [1, 2, 3],
                    "y": [4, 5, 6],
                },
                index=pd.Index(["a", "b", "c"], name="id"),
            )
            return dd.from_pandas(df, npartitions=1)

    pipeline.read(
        ref=CreateData,
        produces={"x": pa.int32(), "y": pa.int32()},
    )
