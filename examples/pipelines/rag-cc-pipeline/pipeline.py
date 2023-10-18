"""Pipeline used to create a stable diffusion dataset from a set of given images."""
import logging
import sys

sys.path.append("../")

from pipeline_configs import PipelineConfigs

from fondant.pipeline import ComponentOp, Pipeline

logger = logging.getLogger(__name__)

pipeline = Pipeline(
    pipeline_name="rag-cc-pipeline",
    pipeline_description="Pipeline to prepare and process data for building a RAG solution",
    base_path=PipelineConfigs.BASE_PATH,
)

load_component_column_mapping = {"document_text": "text_data"}

# Define component ops
load_from_hub_op = ComponentOp(
    component_dir="components/load_from_hf_hub",
    arguments={
        "dataset_name": "lukesjordan/worldbank-project-documents@~parquet",
        "column_name_mapping": load_component_column_mapping,
        "n_rows_to_load": 20,
        "index_column": "project_id",
    },
)

chunk_text_op = ComponentOp(
    component_dir="components/chunk_text",
    arguments={
        "chunk_size": 512,
        "chunk_overlap": 10,
    },
)

embed_text_op = ComponentOp(
    component_dir="components/embed_text",
    arguments={
        "model_provider": "vertexai",
        "auth_kwargs": {"project": "soy-audio-379412"},
    },
)

pipeline.add_op(load_from_hub_op)
pipeline.add_op(chunk_text_op, dependencies=load_from_hub_op)
pipeline.add_op(embed_text_op, dependencies=chunk_text_op)
