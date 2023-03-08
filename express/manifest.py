"""
Script for generating the dataset manifest that will be passed and updated through different
components of the pipeline
"""
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict

from dataclasses_json import dataclass_json


class DataType(str, Enum):
    """
    Supported types for stored data.
    """
    PARQUET = 'parquet'
    BLOB = 'blob'

    @classmethod
    def is_valid(cls, data_type: str) -> bool:
        """Check if data type is valid"""
        # pylint: disable=no-member
        return data_type in cls._member_names_


@dataclass
class DataSource:
    """
    Information about the location and contents of a single data source.
    Args:
        location (str): the URI of the data root
        type (Union[DataType, str]): the type of data, which determines how it's interpreted
        extensions (List[str]): under what file extensions the data is stored.
        n_files (int): how many files make up the data
        n_items (int): how many items (e.g. distinct images, captions, etc.) are covered by the
         data.
    """
    location: str
    type: DataType
    extensions: List[str]
    n_files: int = field(default_factory=int)
    n_items: int = field(default_factory=int)


# pylint: disable=too-few-public-methods, too-many-instance-attributes
@dataclass_json
@dataclass
class Metadata:
    """
    The metadata associated with the manifest
    Args:
        artifact_bucket (str): remote location of all newly created job artifacts.
        run_id (str): the kfp run id associated with the manifest (kfp.dsl.EXECUTION_ID_PLACEHOLDER)
        component_id (str): if this metadata is passed as an input to a component, this is the id of
         the current component. If this metadata is stored as part of a manifest, this is the id of
        the component that generated the manifest.
        component_name (str): name of the current or originating component (see component_id for
         more details).
        branch (str): the git branch associated with that manifest
        commit_hash (str): the commit hash associated with that manifest
        creation_date (str): the creation date of the manifest
        num_items (int): total number of rows in the index
    """
    # TODO: get rid of defaults
    artifact_bucket: str = field(default_factory=str)
    run_id: str = field(default_factory=str)
    component_id: str = field(default_factory=str)
    component_name: str = field(default_factory=str)
    branch: str = field(default_factory=str)
    commit_hash: str = field(default_factory=str)
    creation_date: str = field(default_factory=str)
    num_items: int = field(default_factory=int)


@dataclass_json
@dataclass
class DataManifest:
    """
    The dataset Manifest
    Args:
        index (DataSource): the index parquet file which indexes all the data_sources
        data_sources (List[DataSource]): Location and metadata of various data sources associated
         with the index.
        metadata (Metadata): The metadata associated with the manifest

    """
    index: DataSource
    data_sources: Dict[str, DataSource] = field(default_factory=dict)
    metadata: Metadata = field(default_factory=Metadata)  # TODO: make mandatory during construction

    def __post_init__(self):
        if (self.index.type != DataType.PARQUET) or (not DataType.is_valid(self.index.type)):
            raise TypeError("Index needs to be of type 'parquet'.")
        for name, dataset in self.data_sources.items():
            if not DataType.is_valid(dataset.type):
                raise TypeError(
                    f"Data type '{dataset.type}' for data source '{name}' is not valid.")

    @classmethod
    def from_path(cls, manifest_path):
        """Load data manifest from a given manifest path"""
        with open(manifest_path, encoding="utf-8") as file_:
            manifest_load = json.load(file_)
            # pylint: disable=no-member
            return DataManifest.from_dict(manifest_load)
