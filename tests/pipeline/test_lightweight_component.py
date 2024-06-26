import json
import re
import sys
import textwrap
from dataclasses import dataclass
from importlib.metadata import version
from unittest import mock

import dask.dataframe as dd
import pandas as pd
import pyarrow as pa
import pytest
from fondant.component import DaskLoadComponent, PandasTransformComponent
from fondant.core.component_spec import OperationSpec
from fondant.core.exceptions import InvalidLightweightComponent
from fondant.dataset import Dataset, Image, lightweight_component
from fondant.dataset.compiler import DockerCompiler
from fondant.testing import DockerComposeConfigs


@pytest.fixture()
def default_fondant_image():
    basename = "fndnt/fondant"
    fondant_version = version("fondant")
    python_version = sys.version_info
    python_version = f"{python_version.major}.{python_version.minor}"
    return f"{basename}:{fondant_version}-py{python_version}"


@pytest.fixture()
def load_pipeline(caplog):
    @lightweight_component(
        base_image="python:3.10-slim-buster",
        extra_requires=["pandas", "dask"],
        produces={"x": pa.int32(), "y": pa.int32(), "z": pa.int32()},
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

    dataset = Dataset.create(
        ref=CreateData,
        dataset_name="dummy-dataset",
    )

    caplog_records = caplog.records
    return dataset, load_script, caplog_records


def test_build_python_script(load_pipeline):
    _, load_script, _ = load_pipeline
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


def test_lightweight_component_sdk(
    tmp_path_factory,
    default_fondant_image,
    load_pipeline,
):
    dataset, load_script, caplog_records = load_pipeline

    assert len(dataset._graph.keys()) == 1
    operation_spec_dict = dataset._graph["createdata"][
        "operation"
    ].operation_spec.to_dict()
    assert operation_spec_dict == {
        "specification": {
            "name": "CreateData",
            "image": "python:3.10-slim-buster",
            "description": "lightweight component",
            "consumes": {"additionalProperties": True},
            "produces": {
                "x": {"type": "int32"},
                "y": {"type": "int32"},
                "z": {"type": "int32"},
            },
        },
        "consumes": {},
        "produces": {},
    }

    # check warning: fondant is not part of the requirements
    msg = "You are not using a Fondant default base image"

    assert any(msg in record.message for record in caplog_records)

    @lightweight_component(produces={"x": pa.int32()})
    class AddN(PandasTransformComponent):
        import typing as t

        def __init__(
            self,
            n: int,
            y: t.Optional[int] = None,
            b: t.Optional[int] = 10,
            z: t.Optional[dict] = [],
        ):
            self.n = n
            self.b = b
            self.y = y
            self.z = z

        def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
            dataframe["x"] = dataframe["x"].map(lambda x: x + self.n)
            return dataframe

    _ = dataset.apply(
        ref=AddN,
        arguments={"n": 1},
    )
    assert len(dataset._graph.keys()) == 1 + 1
    assert dataset._graph["addn"]["dependencies"] == ["createdata"]
    dataset._graph["addn"]["operation"].operation_spec.to_json()

    operation_spec_dict = dataset._graph["addn"]["operation"].operation_spec.to_dict()
    assert operation_spec_dict == {
        "specification": {
            "name": "AddN",
            "image": Image.resolve_fndnt_base_image(),
            "description": "lightweight component",
            "consumes": {
                "additionalProperties": True,
            },
            "produces": {"x": {"type": "int32"}},
            "args": {
                "b": {"default": 10, "description": "None", "type": "int"},
                "n": {"description": "None", "type": "int"},
                "y": {"default": "None", "description": "None", "type": "int"},
                "z": {"default": [], "description": "None", "type": "dict"},
            },
        },
        "consumes": {
            "x": {"type": "int32"},
            "y": {"type": "int32"},
            "z": {"type": "int32"},
        },
        "produces": {},
    }
    dataset._validate_dataset_definition()

    with tmp_path_factory.mktemp("temp") as fn:
        DockerCompiler().compile(dataset=dataset, working_directory=str(fn))


def test_consumes_mapping_all_fields(tmp_path_factory, load_pipeline):
    @lightweight_component(
        base_image="python:3.10",
        extra_requires=[
            "fondant[component]@git+https://github.com/ml6team/fondant@main",
        ],
        consumes={"a": pa.int32()},
        produces={"a": pa.int32()},
    )
    class AddN(PandasTransformComponent):
        def __init__(self, n: int):
            self.n = n

        def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
            dataframe["a"] = dataframe["a"].map(lambda x: x + self.n)
            return dataframe

    dataset, _, _ = load_pipeline

    _ = dataset.apply(
        ref=AddN,
        consumes={"a": "x"},
        arguments={"n": 1},
    )

    with tmp_path_factory.mktemp("temp") as fn:
        output_path = str(fn / "kubeflow_pipeline.yml")
        DockerCompiler().compile(
            dataset=dataset,
            output_path=output_path,
            working_directory=str(fn),
        )
        pipeline_configs = DockerComposeConfigs.from_spec(output_path)
        operation_spec = OperationSpec.from_json(
            pipeline_configs.component_configs["addn"].arguments["operation_spec"],
        )
        assert all(k in ["a", "y", "z"] for k in operation_spec.operation_consumes)
        assert "x" in operation_spec.consumes_from_dataset


def test_consumes_mapping_specific_fields(tmp_path_factory, load_pipeline):
    @lightweight_component(
        base_image="python:3.10",
        extra_requires=[
            "fondant[component]@git+https://github.com/ml6team/fondant@main",
        ],
        consumes={"a": pa.int32()},
        produces={"a": pa.int32()},
    )
    class AddN(PandasTransformComponent):
        def __init__(self, n: int):
            self.n = n

        def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
            dataframe["a"] = dataframe["a"].map(lambda x: x + self.n)
            return dataframe

    dataset, _, _ = load_pipeline

    dataset = dataset.apply(
        ref=AddN,
        consumes={"a": "x"},
        arguments={"n": 1},
    )

    with tmp_path_factory.mktemp("temp") as fn:
        output_path = str(fn / "kubeflow_pipeline.yml")
        DockerCompiler().compile(
            dataset=dataset,
            output_path=output_path,
            working_directory=str(fn),
        )
        pipeline_configs = DockerComposeConfigs.from_spec(output_path)
        operation_spec = OperationSpec.from_json(
            pipeline_configs.component_configs["addn"].arguments["operation_spec"],
        )
        assert "a" in operation_spec.operation_consumes
        assert "x" in operation_spec.consumes_from_dataset
        assert "z" not in operation_spec.operation_consumes


def test_consumes_mapping_additional_fields(tmp_path_factory, load_pipeline):
    @lightweight_component(
        base_image="python:3.10",
        extra_requires=[
            "fondant[component]@git+https://github.com/ml6team/fondant@main",
        ],
        consumes={"additionalProperties": True},
        produces={"a": pa.int32()},
    )
    class AddN(PandasTransformComponent):
        def __init__(self, n: int):
            self.n = n

        def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
            dataframe["a"] = dataframe["x"].map(lambda x: x + self.n)
            return dataframe

    dataset, _, _ = load_pipeline

    dataset = dataset.apply(
        ref=AddN,
        consumes={"x": pa.int32()},
        arguments={"n": 1},
    )

    with tmp_path_factory.mktemp("temp") as fn:
        output_path = str(fn / "kubeflow_pipeline.yml")
        DockerCompiler().compile(
            dataset=dataset,
            output_path=output_path,
            working_directory=str(fn),
        )
        pipeline_configs = DockerComposeConfigs.from_spec(output_path)
        operation_spec = OperationSpec.from_json(
            pipeline_configs.component_configs["addn"].arguments["operation_spec"],
        )
        assert "x" in operation_spec.operation_consumes
        assert "a" in operation_spec.operation_produces
        assert "z" not in operation_spec.operation_consumes


def test_produces_mapping_additional_fields(tmp_path_factory, load_pipeline):
    @lightweight_component(
        base_image="python:3.10",
        extra_requires=[
            "fondant[component]@git+https://github.com/ml6team/fondant@main",
        ],
        consumes={"additionalProperties": True},
    )
    class AddN(PandasTransformComponent):
        def __init__(self, n: int):
            self.n = n

        def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
            dataframe["a"] = dataframe["x"].map(lambda x: x + self.n)
            dataframe["b"] = dataframe["x"].map(lambda x: x + self.n)
            dataframe["c"] = dataframe["x"].map(lambda x: x + self.n)
            return dataframe

    dataset, _, _ = load_pipeline

    dataset = dataset.apply(
        ref=AddN,
        consumes={"x": pa.int32()},
        produces={"a": pa.int32(), "b": pa.int32(), "c": pa.int32()},
        arguments={"n": 1},
    )

    with tmp_path_factory.mktemp("temp") as fn:
        output_path = str(fn / "kubeflow_pipeline.yml")
        DockerCompiler().compile(
            dataset=dataset,
            output_path=output_path,
            working_directory=str(fn),
        )
        pipeline_configs = DockerComposeConfigs.from_spec(output_path)
        operation_spec = OperationSpec.from_json(
            pipeline_configs.component_configs["addn"].arguments["operation_spec"],
        )
        assert all(k in ["a", "b", "c"] for k in operation_spec.operation_produces)


def test_lightweight_component_missing_decorator():
    class Foo(DaskLoadComponent):
        def load(self) -> str:
            return "bar"

    with pytest.raises(InvalidLightweightComponent):
        Dataset.create(
            ref=Foo,
            produces={"x": pa.int32(), "y": pa.int32()},
        )


def test_valid_load_component():
    @lightweight_component(
        base_image="python:3.10-slim-buster",
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

    dataset = Dataset.create(
        ref=CreateData,
    )

    assert len(dataset._graph.keys()) == 1
    operation_spec = dataset._graph["createdata"]["operation"].operation_spec.to_json()
    operation_spec_without_image = json.loads(operation_spec)

    assert operation_spec_without_image == {
        "specification": {
            "name": "CreateData",
            "image": "python:3.10-slim-buster",
            "description": "lightweight component",
            "consumes": {"additionalProperties": True},
            "produces": {"additionalProperties": True},
        },
        "consumes": {},
        "produces": {},
    }


def test_invalid_load_component():
    with pytest.raises(  # noqa: PT012
        ValueError,
        match="Every required function must be overridden in the LightweightComponent. "
        "Missing implementations for the following functions: \\['load'\\]",
    ):

        @lightweight_component(
            base_image="python:3.10-slim-buster",
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
            base_image="python:3.10-slim-buster",
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
            base_image="python:3.10-slim-buster",
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

    dataset = Dataset.create(
        ref=CreateData,
    )

    assert len(dataset._graph.keys()) == 1
    operation_spec = dataset._graph["createdata"]["operation"].operation_spec.to_json()
    operation_spec_without_image = json.loads(operation_spec)

    assert operation_spec_without_image == {
        "specification": {
            "name": "CreateData",
            "image": Image.resolve_fndnt_base_image(),
            "description": "lightweight component",
            "consumes": {"additionalProperties": True},
            "produces": {"additionalProperties": True},
        },
        "consumes": {},
        "produces": {},
    }


@dataclass
class MockSysVersionInfo:
    major: int
    minor: int
    micro: int

    def __lt__(self, other):
        return (self.major, self.minor, self.micro) <= other

    def __ge__(self, other):
        return other < (self.major, self.minor, self.micro)


def test_fndnt_base_image_resolution():
    # Base image version is set to python version
    with mock.patch.object(sys, "version_info", MockSysVersionInfo(3, 10, 0)):
        base_image_name = Image.resolve_fndnt_base_image()
        assert base_image_name == "fndnt/fondant:dev-py3.10"

    # Local python version is not supported
    with mock.patch.object(sys, "version_info", MockSysVersionInfo(3, 12, 0)):
        base_image_name = Image.resolve_fndnt_base_image()
        assert base_image_name == "fndnt/fondant:dev-py3.11"

    with mock.patch.object(sys, "version_info", MockSysVersionInfo(3, 7, 0)):
        base_image_name = Image.resolve_fndnt_base_image()
        assert base_image_name == "fndnt/fondant:dev-py3.11"

    with mock.patch.object(
        sys,
        "version_info",
        MockSysVersionInfo(3, 9, 0),
    ), mock.patch("importlib.metadata.version") as mock_call:
        mock_call.return_value = "0.9"
        base_image_name = Image.resolve_fndnt_base_image()
        assert base_image_name == "fndnt/fondant:0.9-py3.9"


def test_infer_consumes_if_not_defined(load_pipeline):
    """
    Test that the consumes mapping is inferred when not defined in dataset interface.
    All columns of the dataset are consumed.
    """
    dataset, _, _ = load_pipeline

    @lightweight_component(
        base_image="python:3.10-slim-buster",
        extra_requires=["pandas", "dask"],
        consumes={"x": pa.int32(), "y": pa.int32(), "z": pa.int32()},
        produces={"x": pa.int32(), "y": pa.int32(), "z": pa.int32()},
    )
    class Bar(PandasTransformComponent):
        def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
            return dataframe

    dataset = dataset.apply(
        ref=Bar,
    )

    operation_spec_dict = dataset._graph["bar"]["operation"].operation_spec.to_dict()
    assert operation_spec_dict == {
        "consumes": {
            "x": {"type": "int32"},
            "y": {"type": "int32"},
            "z": {"type": "int32"},
        },
        "produces": {},
        "specification": {
            "consumes": {
                "x": {"type": "int32"},
                "y": {"type": "int32"},
                "z": {"type": "int32"},
            },
            "description": "lightweight component",
            "image": "python:3.10-slim-buster",
            "name": "Bar",
            "produces": {
                "x": {"type": "int32"},
                "y": {"type": "int32"},
                "z": {"type": "int32"},
            },
        },
    }


def test_infer_consumes_if_additional_properties_true(load_pipeline):
    """
    Test when additional properties is true (no consumes defined in the lightweight component),
    the consumes is inferred from the dataset interface.
    """
    dataset, _, _ = load_pipeline

    @lightweight_component(
        base_image="python:3.10-slim-buster",
        extra_requires=["pandas", "dask"],
        produces={"x": pa.int32(), "y": pa.int32(), "z": pa.int32()},
    )
    class Bar(PandasTransformComponent):
        def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
            return dataframe

    dataset = dataset.apply(
        ref=Bar,
    )

    operation_spec_dict = dataset._graph["bar"]["operation"].operation_spec.to_dict()
    assert operation_spec_dict == {
        "consumes": {
            "x": {"type": "int32"},
            "y": {"type": "int32"},
            "z": {"type": "int32"},
        },
        "produces": {},
        "specification": {
            "consumes": {"additionalProperties": True},
            "description": "lightweight component",
            "image": "python:3.10-slim-buster",
            "name": "Bar",
            "produces": {
                "x": {"type": "int32"},
                "y": {"type": "int32"},
                "z": {"type": "int32"},
            },
        },
    }


def test_warning_is_logged_when_produces_is_not_defined(caplog):
    @lightweight_component
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

    _ = Dataset.create(
        ref=CreateData,
        dataset_name="dummy-dataset",
    )

    assert (
        "Can not infer produces. The component will not produce any new columns"
        in caplog.text
    )
