from express.pipeline import ExpressComponentOperation, ExpressPipeline
from express.logger import configure_logging
from pathlib import Path
import os

configure_logging()
#
host = "https://472c61c751ab9be9-dot-europe-west1.pipelines.googleusercontent.com"
comp_1_yaml_path = \
    "/home/philippe/Scripts/express/tests/component_example/valid_component/express_component.yaml"

comp_2_yaml_path = \
    "/home/philippe/Scripts/express/tests/component_example/valid_component/express_component.yaml"
#
component_args = {"args": "11", "storage_args": "a dummy string arg"}
pipeline_exanple_path = "/tests/pipeline_examples"
valid_pipeline_path = os.path.join(pipeline_exanple_path, "valid_pipeline")
valid_pipeline_example_1 = os.path.join(valid_pipeline_path, "example_1")
# invalid_pipeline_path = os.path.join(pipeline_exanple_path, "invalid_pipeline")
#
# component_names = ["first_component.yaml", "second_component.yaml"]
# valid_component_ops = \
#     [ExpressComponentOperation(os.path.join(valid_pipeline_example_1, component_name), component_args)
#      for component_name in component_names]
#
# pipeline = ExpressPipeline(host)
# pipeline._validate_pipeline_definition(valid_component_ops)
#
# component_1_op = ExpressComponentOperation(Path(valid_pipeline_path / "ca"), component_args)
# component_1_op = ExpressComponentOperation(comp_1_yaml_path, component_args)

component_1_op = ExpressComponentOperation(
    comp_1_yaml_path,
    component_args,
    number_of_gpus=1,
    node_pool_name="nvme",
    ephemeral_storage_size="2T")
component_2_op = ExpressComponentOperation(comp_2_yaml_path, component_args)
component_ops = [component_1_op, component_2_op]

name = "test-a"
description = "test_pipeline"
express_components_operation = component_ops
pipeline_package_path = "test.tgz"

pipeline = ExpressPipeline(host)

pipeline.compile_pipeline(name, description, express_components_operation, pipeline_package_path)
# pipeline.upload_pipeline(name, pipeline_package_path)
