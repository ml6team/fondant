"""Pipeline used to create a stable diffusion dataset from a set of given images."""
from config.general_config import KubeflowConfig
from config.components_config import (
    LoadFromHubConfig,
    ImageFilterConfig,
)
import kfp

from fondant.pipeline import FondantComponentOp, FondantPipeline

# General configs
pipeline_name = "Test fondant pipeline"
pipeline_description = "A test pipeline"
pipeline_package_path = "test.tgz"
pipeline_host = KubeflowConfig.HOST
pipeline_base_path = KubeflowConfig.BASE_PATH
client = kfp.Client(host=pipeline_host)

# Define component ops
load_from_hub_op = FondantComponentOp(
    component_spec_path="components/load_from_hub/fondant_component.yaml",
    arguments={"dataset_name": LoadFromHubConfig.DATASET_NAME}

)

image_filtering_op = FondantComponentOp(
    component_spec_path="components/image_filtering/fondant_component.yaml",
    arguments={"min_width": ImageFilterConfig.MIN_WIDTH,
               "min_height": ImageFilterConfig.MIN_HEIGHT}

)


pipeline = FondantPipeline(host=pipeline_host)
pipeline_operations = pipeline.chain_operations(load_from_hub_op, image_filtering_op)

pipeline.compile_pipeline(
    pipeline_name=pipeline_name,
    pipeline_description=pipeline_description,
    base_path=pipeline_base_path,
    fondant_components_operation=pipeline_operations,
    pipeline_package_path=pipeline_package_path
)

pipeline.upload_pipeline(pipeline_name=pipeline_name,
                         pipeline_package_path=pipeline_package_path)
