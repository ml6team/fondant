"""
Script for generating the dataset manifest that will be passed and updated through different
components of the pipeline
"""
import json
from dataclasses import dataclass, field
from typing import Dict

from dataclasses_json import dataclass_json


@dataclass
class Image:
    """
    A single image
    Args:
        width (int): width of the image
        height (int): height of the image
    """

    width: int
    height: int


@dataclass
class Text:
    """
    A single text
    Args:
        len (int): length of the text
    """

    len: int


@dataclass
class Vector:
    """
    A single vector
    Args:
        size (int): size of the vector
    """

    size: int


@dataclass
class DataSource:
    """
    Information about the location and contents of a single data source.
    Args:
        location (str): the URI of the data root
        len (int): length of the data source
    """

    location: str
    len: int


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
    The manifest
    Args:
        index (DataSource): the index parquet file which indexes all the data_sources
        data_sources (List[DataSource]): Location and metadata of various data sources associated
         with the index.
        metadata (Metadata): The metadata associated with the manifest

    """

    index: DataSource
    data_sources: Dict[str, DataSource] = field(default_factory=dict)
    metadata: Metadata

    @classmethod
    def from_path(cls, manifest_path):
        """Load manifest from a given manifest path"""
        with open(manifest_path, encoding="utf-8") as file_:
            manifest_load = json.load(file_)
            return DataManifest.from_dict(manifest_load)
