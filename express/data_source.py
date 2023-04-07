from typing import List

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
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
    len: int
    column_names: List[str]


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
