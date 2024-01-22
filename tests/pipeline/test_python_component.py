import textwrap

import dask.dataframe as dd
import pandas as pd
import pyarrow as pa
import pytest
from fondant.component import DaskLoadComponent, PandasTransformComponent
from fondant.core.component_spec import OperationSpec
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
        base_image="python:3.8",
        extra_requires=[
            "fondant[component]@git+https://github.com/ml6team/fondant@main",
        ],
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

    dataset = pipeline.read(
        ref=CreateData,
        produces={"x": pa.int32(), "y": pa.int32(), "z": pa.int32()},
    )

    return pipeline, dataset


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


def test_compile_lightweight_component(load_pipeline):
    pipeline, dataset = load_pipeline

    @lightweight_component(
        base_image="python:3.8",
        extra_requires=[
            "fondant[component]@git+https://github.com/ml6team/fondant@main",
        ],
        consumes="generic",
    )
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

    pipeline, dataset = load_pipeline

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

    _, dataset = load_pipeline

    with pytest.raises(
        ValueError,
        match="Field `nonExistingField` is not available in" " the dataset.",
    ):
        _ = dataset.apply(
            ref=AddN,
            consumes={"a": "x"},
            produces={"a": pa.int32()},
            arguments={"n": 1},
        )
