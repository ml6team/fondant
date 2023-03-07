"""
Script for generating the dataset manifest that will be passed and updated through different
components of the pipeline
"""
from dataclasses import dataclass, field
from typing import Dict

from dataclasses_json import dataclass_json


@dataclass
class AssociatedData:
    """
    The data information associated with the manifest
    Args:
        dataset (str): the path to the dataset parquet file containing image ids, gcs paths and
         their associated metadata (url, format, ...)
        caption (str): the path to the caption parquet file containing image ids and their
        associated captions
        embedding (str): the path to the embedding gcs url containing the `.npy` image embeddings
    """
    dataset: Dict[str, str] = field(default_factory=dict)
    caption: Dict[str, str] = field(default_factory=dict)
    embedding: Dict[str, str] = field(default_factory=dict)


@dataclass
class Metadata:
    """
    The metadata associated with the manifest
    Args:
        branch (str): the git branch associated with that manifest
        commit_hash (str): the commit hash associated with that manifest
        creation_date (str): the creation date of the manifest
        run_id (str): the kfp run id associated with the manifest (kfp.dsl.EXECUTION_ID_PLACEHOLDER)
    """
    branch: str = field(default_factory=str)
    commit_hash: str = field(default_factory=str)
    creation_date: str = field(default_factory=str)
    run_id: str = field(default_factory=str)


@dataclass_json
@dataclass
class DataManifest:
    """
    The dataset manifest.
    
    Args:
        dataset_id (str): the id of the dataset
        index (str): the path to the index parquet file
        associated_data (AssociatedData): The data information associated with the manifest
        metadata (Metadata): The metadata associated with the manifest

    """
    dataset_id: str = field(default_factory=str)
    index: str = field(default_factory=str)
    associated_data: AssociatedData = field(default_factory=AssociatedData)
    metadata: Metadata = field(default_factory=Metadata)
