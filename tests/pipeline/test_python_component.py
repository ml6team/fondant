import json
import re
import textwrap

import dask.dataframe as dd
import pandas as pd
import pyarrow as pa
import pytest
from fondant.component import DaskLoadComponent, PandasTransformComponent
from fondant.core.component_spec import OperationSpec
from fondant.core.exceptions import InvalidPythonComponent
from fondant.pipeline import Pipeline, lightweight_component
from fondant.pipeline.compiler import DockerCompiler
from fondant.testing import DockerComposeConfigs


@pytest.fixture()
def load_pipeline():
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
                    "z": [7, 8, 9],
                },
                index=pd.Index(["a", "b", "c"], name="id"),
            )
            return dd.from_pandas(df, npartitions=1)

    load_script = CreateData.image().script

    dataset = pipeline.read(
        ref=CreateData,
        produces={"x": pa.int32(), "y": pa.int32(), "z": pa.int32()},
    )

    return pipeline, dataset, load_script


def test_build_python_script(load_pipeline):
    _, _, load_script = load_pipeline
    assert load_script == textwrap.dedent(
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
                        "z": [7, 8, 9],
                    },
                    index=pd.Index(["a", "b", "c"], name="id"),
                )
                return dd.from_pandas(df, npartitions=1)
    """,
    )


def test_lightweight_component_sdk(load_pipeline):
    pipeline, dataset, load_script = load_pipeline

    assert len(pipeline._graph.keys()) == 1
    operation_spec = pipeline._graph["CreateData"]["operation"].operation_spec.to_json()
    assert json.loads(operation_spec) == {
        "specification": {
            "name": "CreateData",
            "image": "python:3.8-slim-buster",
            "description": "python component",
            "consumes": {"additionalProperties": True},
            "produces": {"additionalProperties": True},
        },
        "consumes": {},
        "produces": {
            "x": {"type": "int32"},
            "y": {"type": "int32"},
            "z": {"type": "int32"},
        },
    }

    @lightweight_component
    class AddN(PandasTransformComponent):
        def __init__(self, n: int, **kwargs):
            self.n = n

        def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
            dataframe["x"] = dataframe["x"].map(lambda x: x + self.n)
            return dataframe

    _ = dataset.apply(
        ref=AddN,
        produces={"x": pa.int32(), "y": pa.int32(), "z": pa.int32()},
        consumes={"x": pa.int32(), "y": pa.int32(), "z": pa.int32()},
        arguments={"n": 1},
    )

    assert len(pipeline._graph.keys()) == 1 + 1
    assert pipeline._graph["AddN"]["dependencies"] == ["CreateData"]
    operation_spec = pipeline._graph["AddN"]["operation"].operation_spec.to_json()
    assert json.loads(operation_spec) == {
        "specification": {
            "name": "AddN",
            "image": "fondant:latest",
            "description": "python component",
            "consumes": {"additionalProperties": True},
            "produces": {"additionalProperties": True},
            "args": {"n": {"type": "int"}},
        },
        "consumes": {
            "x": {"type": "int32"},
            "y": {"type": "int32"},
            "z": {"type": "int32"},
        },
        "produces": {
            "x": {"type": "int32"},
            "y": {"type": "int32"},
            "z": {"type": "int32"},
        },
    }
    pipeline._validate_pipeline_definition(run_id="dummy-run-id")

    DockerCompiler().compile(pipeline)


def test_valid_consumes_mapping(tmp_path_factory, load_pipeline):
    @lightweight_component(
        base_image="python:3.8",
        extra_requires=[
            "fondant[component]@git+https://github.com/ml6team/fondant@main",
        ],
        consumes=["a", "y"],
    )
    class AddN(PandasTransformComponent):
        def __init__(self, n: int, **kwargs):
            self.n = n

        def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
            dataframe["a"] = dataframe["a"].map(lambda x: x + self.n)
            return dataframe

    pipeline, dataset, _ = load_pipeline

    _ = dataset.apply(
        ref=AddN,
        consumes={"a": "x"},
        produces={"a": pa.int32()},
        arguments={"n": 1},
    )

    with tmp_path_factory.mktemp("temp") as fn:
        output_path = str(fn / "kubeflow_pipeline.yml")
        DockerCompiler().compile(pipeline=pipeline, output_path=output_path)
        pipeline_configs = DockerComposeConfigs.from_spec(output_path)
        operation_spec = OperationSpec.from_json(
            pipeline_configs.component_configs["AddN"].arguments["operation_spec"],
        )
        assert all(k in ["a", "y"] for k in operation_spec.inner_consumes)
        assert "z" not in operation_spec.inner_consumes


def test_invalid_consumes_mapping(tmp_path_factory, load_pipeline):
    @lightweight_component(
        base_image="python:3.8",
        extra_requires=[
            "fondant[component]@git+https://github.com/ml6team/fondant@main",
        ],
        consumes=["nonExistingField"],
    )
    class AddN(PandasTransformComponent):
        def __init__(self, n: int, **kwargs):
            self.n = n

        def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
            dataframe["a"] = dataframe["a"].map(lambda x: x + self.n)
            return dataframe

    _, dataset, _ = load_pipeline

    with pytest.raises(
        ValueError,
        match="Field `nonExistingField` is not available in the dataset.",
    ):
        _ = dataset.apply(
            ref=AddN,
            consumes={"a": "x"},
            produces={"a": pa.int32()},
            arguments={"n": 1},
        )


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

    CreateData(produces={}, consumes={})


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


def test_lightweight_component_decorator_without_parentheses():
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
    operation_spec = pipeline._graph["CreateData"]["operation"].operation_spec.to_json()
    assert json.loads(operation_spec) == {
        "specification": {
            "name": "CreateData",
            "image": "fondant:latest",
            "description": "python component",
            "consumes": {"additionalProperties": True},
            "produces": {"additionalProperties": True},
        },
        "consumes": {},
        "produces": {},
    }
