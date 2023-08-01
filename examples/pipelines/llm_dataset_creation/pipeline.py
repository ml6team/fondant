"""Pipeline used to create the dataset to train the StarCoder model."""

import logging
import sys
from dataclasses import dataclass

sys.path.append("../../")

from fondant.pipeline import ComponentOp, Pipeline, Client

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfigs:
    HOST = "http://localhost:8080"
    BASE_PATH = "s3://local-language-dataset-artifacts"


# initialize pipeline
pipeline = Pipeline(
    pipeline_name="local-llm-dataset-creation",
    base_path=PipelineConfigs.BASE_PATH,
    pipeline_description="A pipeline for downloading Common crawl files and filter text files",
)

client = Client(host=PipelineConfigs.HOST)

# define ops
load_from_commoncrawl_op = ComponentOp(
    component_dir="components/load_from_commoncrawl",
    arguments={
        "index_name": "CC-MAIN-2023-14",
        "n_segments_to_load": 1,
    },
)

download_commoncrawl_segments_op = ComponentOp(
    component_dir="components/download_commoncrawl_segments",
    arguments={
        "n_records_to_download": 5000,
        # "partition_size": 250,
        "get_plain_text": True,
        "use_s3": True,
    },
)

text_length_filter = ComponentOp(
    component_dir="components/text_length_filter",
    arguments={"min_characters_length": 150, "min_words_length": 150},
)

text_normalization = ComponentOp(
    component_dir="components/text_normalization",
    arguments={
        "apply_nfc": True,
        "do_lowercase": True,
        "characters_to_remove": [
            "[\r\n\t]"
        ],  # remove whitespace, tab, newline and punctation
    },
)

language_filter = ComponentOp(
    component_dir="components/language_filter",
    arguments={"language": "de"},
)

minhash_generator = ComponentOp(
    component_dir="components/minhash_generator",
    arguments={},
)

cluster_hash_values = ComponentOp(
    component_dir="components/cluster_hash_values",
    arguments={"sample_ratio": 0.5, "num_clusters": 3},
)

dedup_entries = ComponentOp(
    component_dir="components/dedup_minhashes",
    arguments={"num_perm": 128, "epsilon": 0.8},
)


# add ops to pipeline
pipeline.add_op(load_from_commoncrawl_op)
pipeline.add_op(download_commoncrawl_segments_op, dependencies=load_from_commoncrawl_op)
pipeline.add_op(text_normalization, dependencies=load_from_commoncrawl_op)
pipeline.add_op(text_length_filter, dependencies=text_normalization)
pipeline.add_op(language_filter, dependencies=text_normalization)
pipeline.add_op(minhash_generator, dependencies=language_filter)
pipeline.add_op(cluster_hash_values, dependencies=minhash_generator)
pipeline.add_op(dedup_entries, dependencies=cluster_hash_values)
client.compile_and_run(pipeline=pipeline)
