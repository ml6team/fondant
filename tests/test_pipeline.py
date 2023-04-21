"""Express pipelines test"""
import os
import pytest
from unittest import mock

from express.exceptions import InvalidPipelineDefinition
from express.pipeline import ExpressComponentOperation, ExpressPipeline

valid_pipelines_path = os.path.join("pipeline_examples", "valid_pipeline")
invalid_pipelines_path = os.path.join("pipeline_examples", "invalid_pipeline")


@pytest.fixture
def default_pipeline_args():
    return {
        'pipeline_name': 'pipeline',
        'pipeline_description': 'pipeline_description'
    }


@pytest.fixture
def mock_host():
    return "http://mock-host-url"


@pytest.mark.parametrize("valid_pipeline_example", [
    ("example_1", ["first_component.yaml", "second_component.yaml", "third_component.yaml"]),
])
def test_valid_pipeline(mock_host, default_pipeline_args, valid_pipeline_example, tmp_path):
    """Test that a valid pipeline definition can be compiled without errors."""
    example_dir, component_names = valid_pipeline_example
    component_args = {"storage_args": "a dummy string arg"}
    components_path = os.path.join(valid_pipelines_path, example_dir)
    operations = [ExpressComponentOperation(os.path.join(components_path, name), component_args) for
                  name in component_names]

    with mock.patch('express.pipeline.kfp.Client'):
        pipeline = ExpressPipeline(mock_host)
        pipeline_package_path = os.path.join(tmp_path, "test.tgz")
        pipeline.compile_pipeline(
            express_components_operation=operations,
            pipeline_package_path=pipeline_package_path,
            **default_pipeline_args)


@pytest.mark.parametrize("invalid_pipeline_example", [
    ("example_1", ["first_component.yaml", "second_component.yaml"]),
    ("example_2", ["first_component.yaml", "second_component.yaml"]),
])
def test_invalid_pipeline(mock_host, default_pipeline_args, invalid_pipeline_example, tmp_path):
    """
    Test that an InvalidPipelineDefinition exception is raised when attempting to compile
    an invalid pipeline definition.
    """
    example_dir, component_names = invalid_pipeline_example
    components_path = os.path.join(invalid_pipelines_path, example_dir)
    component_args = {"storage_args": "a dummy string arg"}
    operations = [ExpressComponentOperation(os.path.join(components_path, name), component_args) for
                  name in component_names]

    with mock.patch('express.pipeline.kfp.Client'):
        pipeline = ExpressPipeline(mock_host)
        pipeline_package_path = os.path.join(tmp_path, "test.tgz")
        with pytest.raises(InvalidPipelineDefinition):
            pipeline.compile_pipeline(
                express_components_operation=operations,
                pipeline_package_path=pipeline_package_path,
                **default_pipeline_args)


@pytest.mark.parametrize("invalid_component_args", [
    {"invalid_arg": "a dummy string arg", "storage_args": "a dummy string arg"},
    {"args": 1, "storage_args": "a dummy string arg"},
])
def test_invalid_argument(mock_host, default_pipeline_args, invalid_component_args, tmp_path):
    """
    Test that an exception is raised when the passed invalid argument name or type to the express
    component does not match the ones specified in the express specifications
    """
    components_spec_path = os.path.join(
        *[valid_pipelines_path, "example_1", "first_component.yaml"])
    component_operation = ExpressComponentOperation(components_spec_path, invalid_component_args)
    operations = [component_operation]

    with mock.patch('express.pipeline.kfp.Client'):
        pipeline = ExpressPipeline(mock_host)
        pipeline_package_path = os.path.join(tmp_path, "test.tgz")
        with pytest.raises((ValueError, TypeError)):
            pipeline.compile_pipeline(
                express_components_operation=operations,
                pipeline_package_path=pipeline_package_path,
                **default_pipeline_args)
