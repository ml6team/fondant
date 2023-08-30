import logging
import sys

sys.path.append("../")

from pipeline_configs import PipelineConfigs

from fondant.pipeline import ComponentOp, Pipeline

logger = logging.getLogger(__name__)
# General configs
pipeline_name = "commoncrawl-pipeline"
pipeline_description = "Pipeline that downloads from commoncrawl with dask.distributed"

read_warc_paths_op = ComponentOp(
    component_dir="components/read_warc_paths",
    arguments={"common_crawl_indices": ["CC-MAIN-2023-23"]},
    cache=False,
)

load_warc_files_op = ComponentOp(
    component_dir="components/download_warc_files",
    node_pool_label="node_pool",
    node_pool_name="n2-standard-128-pool",
)

pipeline = Pipeline(pipeline_name=pipeline_name, base_path=PipelineConfigs.BASE_PATH)

pipeline.add_op(read_warc_paths_op)
pipeline.add_op(load_warc_files_op, dependencies=[read_warc_paths_op])
