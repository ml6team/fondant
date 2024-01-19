import textwrap

import dask.dataframe as dd
import pandas as pd
import pyarrow as pa
import pytest
from fondant.component import DaskLoadComponent, PandasTransformComponent
from fondant.core.exceptions import InvalidPythonComponent
from fondant.pipeline import Pipeline, lightweight_component
from fondant.pipeline.compiler import DockerCompiler


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


def test_lightweight_component_sdk():
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

    assert len(pipeline._graph.keys()) == 1
    operation_spec_dict = pipeline._graph["CreateData"][
        "operation"
    ].operation_spec.to_dict()
    assert operation_spec_dict == {
        "specification": {
            "name": "CreateData",
            "image": "python:3.8-slim-buster",
            "description": "python component",
            "consumes": {"additionalProperties": True},
            "produces": {"additionalProperties": True},
        },
        "consumes": {},
        "produces": {"x": {"type": "int32"}, "y": {"type": "int32"}},
    }

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
    assert len(pipeline._graph.keys()) == 1 + 1
    assert pipeline._graph["AddN"]["dependencies"] == ["CreateData"]
    operation_spec_dict = pipeline._graph["AddN"]["operation"].operation_spec.to_dict()
    assert operation_spec_dict == {
        "specification": {
            "name": "AddN",
            "image": "fondant:latest",
            "description": "python component",
            "consumes": {"additionalProperties": True},
            "produces": {"additionalProperties": True},
            "args": {"n": {"type": "int"}},
        },
        "consumes": {"x": {"type": "int32"}, "y": {"type": "int32"}},
        "produces": {"x": {"type": "int32"}, "y": {"type": "int32"}},
    }
    pipeline._validate_pipeline_definition(run_id="dummy-run-id")

    DockerCompiler().compile(pipeline)


def test_lightweight_component_missing_decorator():
    pipeline = Pipeline(
        name="dummy-pipeline",
        base_path="./data",
    )

    class Foo(DaskLoadComponent):
        def load(self) -> str:
            return "bar"

    with pytest.raises(InvalidPythonComponent):
        _ = pipeline.read(
            ref=Foo,
            produces={"x": pa.int32(), "y": pa.int32()},
        )
