# Pipeline code goes here
from pathlib import Path
import logging
import sys

sys.path.append("../")
from fondant.pipeline import ComponentOp, Pipeline


def create_directory_if_not_exists(path):
    p_base_path = Path(path).resolve()
    p_base_path.mkdir(parents=True, exist_ok=True)
    return str(p_base_path)


logger = logging.getLogger(__name__)

PIPELINE_NAME = "cc-image-filter-pipeline"
PIPELINE_DESCRIPTION = "Load cc image dataset and reduce to PNG files"
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
}

load_from_hf_hub = ComponentOp(
    component_dir="components/load_from_hf_hub",
    arguments={
        "dataset_name": "fondant-ai/fondant-cc-25m",
        "column_name_mapping": load_component_column_mapping,
        "n_rows_to_load": 100,
    },
    cache=False,
)

# Filter mime type component
filter_mime_type = ComponentOp(
    component_dir="components/filter_file_type",
    arguments={"mime_type": "image/png"},
    cache=False,
)

# Download images component
download_images = ComponentOp(
    component_dir="components/download_images", arguments={}, cache=False
)


# Add components to the pipeline
pipeline.add_op(load_from_hf_hub)
pipeline.add_op(filter_mime_type, dependencies=[load_from_hf_hub])
pipeline.add_op(download_images, dependencies=[filter_mime_type])
