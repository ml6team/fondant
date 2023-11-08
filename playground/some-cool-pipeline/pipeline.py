# Pipeline code goes here
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)
sys.path.append("../")
from fondant.pipeline.pipeline import ComponentOp, Pipeline


def create_directory_if_not_exists(path):
    p_base_path = Path(path).resolve()
    p_base_path.mkdir(parents=True, exist_ok=True)
    return str(p_base_path)


PIPELINE_NAME = "cool-pipeline"
PIPELINE_DESCRIPTION = "Load cc image dataset"
BASE_PATH = "./data"
BASE_PATH = create_directory_if_not_exists(BASE_PATH)

# Define pipeline
pipeline = Pipeline(pipeline_name=PIPELINE_NAME, base_path=BASE_PATH)

# Load from hub component
load_component_column_mapping = {
    "alt_text": "images_alt+text",
    "image_url": "images_url",
    "license_location": "images_license+location",
    "license_type": "images_license+type",
    "webpage_url": "images_webpage+url",
    "surt_url": "images_surt+url",
    "top_level_domain": "images_top+level+domain",
}

load_from_hf_hub = ComponentOp(
    component_dir="components/load_from_hf_hub",
    arguments={
        "dataset_name": "fondant-ai/fondant-cc-25m",
        "column_name_mapping": load_component_column_mapping,
        "n_rows_to_load": 1000,  # Here you can modify the number of images you want to download.
        "input_partition_rows": 100,
    },
)

# Download images component
download_images = ComponentOp(
    component_dir="components/download_images",
    arguments={
        "resize_mode": "no",
        "resize_only_if_bigger": False,
        "input_partition_rows": 100,
    },
)

# Filter on resolution
filter_on_resolution = ComponentOp(
    component_dir="components/filter_image_resolution",
    arguments={
        "min_image_dim": 512,
        "max_aspect_ratio": 2.5,
        "input_partition_rows": 100,
    },
)

# Add components to the pipeline
pipeline.add_op(load_from_hf_hub)
pipeline.add_op(download_images, dependencies=[load_from_hf_hub])
pipeline.add_op(filter_on_resolution, dependencies=[download_images])


if __name__ == "__main__":
    from fondant.pipeline.compiler import SagemakerCompiler

    pipeline.base_path = "S3://s3-fondant-artifacts/"
    compiler = SagemakerCompiler()
    compiler.compile(pipeline, output_path="spec.json")

    from fondant.pipeline.runner import SagemakerRunner

    runner = SagemakerRunner()
    runner.run(input_spec="spec.json")
