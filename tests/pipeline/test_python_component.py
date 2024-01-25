import json
import re
import sys
import textwrap
from importlib.metadata import version

import dask.dataframe as dd
import pandas as pd
import pyarrow as pa
import pytest
from fondant.component import DaskLoadComponent, PandasTransformComponent
from fondant.core.exceptions import InvalidPythonComponent
from fondant.pipeline import Pipeline, lightweight_component
from fondant.pipeline.compiler import DockerCompiler


@pytest.fixture()
def default_fondant_image():
    basename = "fndnt/fondant"
    fondant_version = version("fondant")
    python_version = sys.version_info
    python_version = f"{python_version.major}.{python_version.minor}"
    return f"{basename}:{fondant_version}-py{python_version}"


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


def test_lightweight_component_sdk(default_fondant_image, caplog):
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
    operation_spec_dict = pipeline._graph["createdata"][
        "operation"
    ].operation_spec.to_dict()
    assert operation_spec_dict == {
        "specification": {
            "name": "createdata",
            "image": "python:3.8-slim-buster",
            "description": "python component",
            "consumes": {"additionalProperties": True},
            "produces": {"additionalProperties": True},
        },
        "consumes": {},
        "produces": {"x": {"type": "int32"}, "y": {"type": "int32"}},
    }

    # check warning: fondant is not part of the requirements
    msg = (
        "You are not using a Fondant default base image, and Fondant is not part of"
        "your extra requirements. Please make sure that you have installed fondant "
        "inside your container. Alternatively, you can should add Fondant to "
        "the extra requirements. \n"
        "E.g. \n"
        '@lightweight_component(..., extra_requires=["fondant"])'
    )

    assert any(msg in record.message for record in caplog.records)

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
    assert pipeline._graph["addn"]["dependencies"] == ["createdata"]
    pipeline._graph["addn"]["operation"].operation_spec.to_json()

    operation_spec_dict = pipeline._graph["addn"]["operation"].operation_spec.to_dict()
    assert operation_spec_dict == {
        "specification": {
            "name": "addn",
            "image": default_fondant_image,
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


def test_valid_load_component():
    @lightweight_component(
        base_image="python:3.8-slim-buster",
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

    pipeline = Pipeline(
        name="dummy-pipeline",
        base_path="./data",
    )

    pipeline.read(
        ref=CreateData,
    )

    assert len(pipeline._graph.keys()) == 1
    operation_spec = pipeline._graph["createdata"]["operation"].operation_spec.to_json()
    operation_spec_without_image = json.loads(operation_spec)

    assert operation_spec_without_image == {
        "specification": {
            "name": "createdata",
            "image": "python:3.8-slim-buster",
            "description": "python component",
            "consumes": {"additionalProperties": True},
            "produces": {"additionalProperties": True},
        },
        "consumes": {},
        "produces": {},
    }


def test_invalid_load_component():
    with pytest.raises(  # noqa: PT012
        ValueError,
        match="Every required function must be overridden in the PythonComponent. "
        "Missing implementations for the following functions: \\['load'\\]",
    ):

        @lightweight_component(
            base_image="python:3.8-slim-buster",
        )
        class CreateData(DaskLoadComponent):
            def custom_load(self) -> int:
                return 1

        CreateData(produces={}, consumes={})


def test_invalid_load_transform_component():
    with pytest.raises(  # noqa: PT012
        ValueError,
        match="Multiple base classes detected. Only one component should be inherited "
        "or implemented.Found classes: DaskLoadComponent, PandasTransformComponent",
    ):

        @lightweight_component(
            base_image="python:3.8-slim-buster",
        )
        class CreateData(DaskLoadComponent, PandasTransformComponent):
            def load(self) -> dd.DataFrame:
                pass

            def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
                pass

        CreateData(produces={}, consumes={})


def test_invalid_load_component_wrong_return_type():
    with pytest.raises(  # noqa: PT012
        ValueError,
        match=re.escape(
            "Invalid function definition of function load. "
            "The expected function signature "
            "is (self) -> dask.dataframe.core.DataFrame",
        ),
    ):

        @lightweight_component(
            base_image="python:3.8-slim-buster",
        )
        class CreateData(DaskLoadComponent):
            def load(self) -> int:
                return 1

        CreateData(produces={}, consumes={})


def test_lightweight_component_decorator_without_parentheses(default_fondant_image):
    @lightweight_component
    class CreateData(DaskLoadComponent):
        def load(self) -> dd.DataFrame:
            return None

    pipeline = Pipeline(
        name="dummy-pipeline",
        base_path="./data",
    )

    pipeline.read(
        ref=CreateData,
    )

    assert len(pipeline._graph.keys()) == 1
    operation_spec = pipeline._graph["createdata"]["operation"].operation_spec.to_json()
    operation_spec_without_image = json.loads(operation_spec)

    assert operation_spec_without_image == {
        "specification": {
            "name": "createdata",
            "image": default_fondant_image,
            "description": "python component",
            "consumes": {"additionalProperties": True},
            "produces": {"additionalProperties": True},
        },
        "consumes": {},
        "produces": {},
    }
