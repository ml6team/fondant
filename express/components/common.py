"""
This file is the entrypoint of the component. It will parse all arguments
and give them to the actual core of the component.
"""
import argparse
import json
import os
import importlib
from datetime import datetime
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, TypeVar, Generic, Union

from express.manifest import DataManifest, DataSource, Metadata
from express.storage_interface import StorageHandlerModule

STORAGE_MODULE_PATH = StorageHandlerModule().to_dict()[
    os.environ.get("CLOUD_ENV", "GCP")
]
STORAGE_HANDLER = importlib.import_module(STORAGE_MODULE_PATH).StorageHandler()

IndexT = TypeVar("IndexT")
DataT = TypeVar("DataT")


class ExpressDataset(ABC, Generic[IndexT, DataT]):
    """
    An abstract wrapper class that gives read access to Express Datasets.
    It can be extended to create a draft for a new (output) dataset.

    Args:
        manifest (DataManifest): A manifest that describes the different data sources comprising
        the Dataset, as well as their locations.
    """

    def __init__(self, manifest: DataManifest):
        self.manifest = manifest
        self._index_data = self.load_index()

    def extend(self) -> "ExpressDatasetDraft[IndexT, DataT]":
        """
        Create an `ExpressDatasetDraft` that extends this dataset.
        """
        return ExpressDatasetDraft.extend(self)

    @abstractmethod
    def load_index(self) -> IndexT:
        """
        Loads the index data.
        """

    def load(self, data_source: str, for_index: Optional[IndexT] = None) -> DataT:
        """
        Load data from a named data source.

        Args:
            data_source (str): A named data source from the input manifest.
            for_index (Optional[TIndex]): Pass in an index to filter the data on. By default, the
             original Dataset index is used. This argument can be used to use a different
              index instead.

        Returns:
            TData: Data of type TData
        """
        if data_source not in self.manifest.data_sources:
            raise ValueError(
                f"Named source {data_source} not part of recognised data sources: "
                f"{self.manifest.data_sources.keys()}."
            )
        if for_index is None:
            for_index = self._index_data
        return self._load_data_source(
            self.manifest.data_sources[data_source], for_index
        )

    @staticmethod
    @abstractmethod
    def _load_data_source(
        data_source: DataSource, index_filter: Optional[IndexT]
    ) -> DataT:
        """
        Load data from a (possibly remote) path.
        This method can be subclassed to present data in a specific way. For example, as a local
         dataframe, a lazy access method, or a distributed set of data.

        Args:
            data_source (DataSource): The DataSource to load the data from.
            index_filter (Optional[TIndex]): Pass in an index to filter the data on.

        Returns:
            TData: Data of type TData
        """

    # TODO should we keep track of both input and output in the manifests?


class ExpressDatasetDraft(ABC, Generic[IndexT, DataT]):
    """
    Draft of an `ExpressDataset`, tracking both preexisting data sources and local data that still
    needs to be uploaded.

    Args:
        index (Union[DataSource, TIndex]): Index of the output dataset. Needs to be present if no
         `extending_dataset` is set.
        data_sources (Dict[str, Union[DataSource, TData]]): Named preexisting data sources or local
         data to be uploaded. Each data source should have data available for each item in
         the shared index.
        extending_dataset (ExpressDataset[TIndex, TData]): Existing dataset to extend, which will
        take over both its index an all data sources. Needs to be present if no `index` is set.
    """

    def __init__(
        self,
        index: Optional[Union[DataSource, IndexT]] = None,
        data_sources: Dict[str, Union[DataSource, DataT]] = None,
        extending_dataset: Optional[ExpressDataset[IndexT, DataT]] = None,
    ):
        self.index = index
        self.data_sources = data_sources or {}
        if not (extending_dataset is None) ^ (index is None):
            raise ValueError(
                "A dataset draft needs to have a single valid index. Either pass an index "
                "or a pre-existing dataset to extend. Additional data sources can be added "
                "to an extending dataset draft after it's been constructed."
            )
        if extending_dataset is not None:
            if index is not None:
                raise ValueError(
                    "A dataset draft needs to have a valid index. Either pass an index or a "
                    "pre-existing dataset to extend. Not both. Additional data sources can be "
                    "added to an extending dataset draft after it's been constructed."
                )
            self.index = extending_dataset.manifest.index
            for name, dataset in extending_dataset.manifest.associated_data.items():
                self.with_data_source(name, dataset, replace_ok=False)

    @classmethod
    def extend(
        cls, dataset: ExpressDataset[IndexT, DataT]
    ) -> "ExpressDatasetDraft[IndexT, DataT]":
        """
        Creates a new Express Dataset draft extending the given dataset, which will take over both
         its index and all data sources.
        """
        return cls(extending_dataset=dataset)

    def with_index(self, index: DataT) -> "ExpressDatasetDraft[IndexT, DataT]":
        """
        Replaces the current index with the given index.

        Returns:
            ExpressDatasetDraft[TIndex, TData]: self, for easier chaining
        """
        self.index = index
        return self

    def with_data_source(
        self, name: str, data: Union[DataT, DataSource], replace_ok=False
    ) -> "ExpressDatasetDraft[IndexT, DataT]":
        """
        Adds a new data source or replaces a preexisting data source with the same name.

        Args:
            name (str): Name of the data source.
            data (Union[TData, DataSource]): Local data of type `TData`, or a preexisting
             `DataSource`.
            replace_ok (bool): Default=False. Whether to replace a preexisting Data Source of the
            same name, if such a Data Source exists.

        Returns:
            ExpressDatasetDraft[TIndex, TData]: self, for easier chaining
        """
        if (name in self.data_sources) and (not replace_ok):
            raise ValueError(
                f"A conflicting data source with identifier {name} is already set "
                f"in this draft. Data sources on a dataset draft can be replaced "
                f"after it's been constructed."
            )
        # TODO: verify same namespace?
        self.data_sources[name] = data
        return self


class ExpressDatasetHandler(ABC, Generic[IndexT, DataT]):
    """
    Abstract mixin class to read from and write to Express Datasets.
    Can be subclassed to deal with a specific type of parsed data representations, like reading to a
     Pandas DataFrame or a Spark RDD.
    """

    @staticmethod
    def _path_for_upload(metadata: Metadata, name: str) -> str:
        """
        Constructs a remote path for new data sources.

        Args:
            metadata (MetaData): Component metadata, which is used to construct the base path.
            name (str): The name of the data source that's being created.
        Returns:
            str: the destination blob path (indicating a folder) where to upload the file/folder.
        """
        artifact_bucket_blob_path = (
            f"custom_artifact/{metadata.run_id}/{metadata.component_name}/{name}"
        )
        destination_path = STORAGE_HANDLER.construct_blob_path(
            metadata.artifact_bucket, artifact_bucket_blob_path
        )
        return destination_path

    @classmethod
    @abstractmethod
    def _load_dataset(
        cls, input_manifest: DataManifest
    ) -> ExpressDataset[IndexT, DataT]:
        """
        Parses a manifest to an ExpressDataset of a specific type, for downstream use by transform
        components.
        """

    @classmethod
    @abstractmethod
    def _upload_index(cls, index: IndexT, remote_path: str) -> DataSource:
        """
        Uploads index data of a certain type as parquet and creates a new DataSource.

        Args:
            index (TIndex): index data of type `TIndex`
            remote_path (str): fully qualified remote path where to upload the data to.

        Returns:
            DataSource: DataSource for the newly uploaded index data
        """

    @classmethod
    @abstractmethod
    def _upload_data_source(
        cls, name: str, data: DataT, remote_path: str
    ) -> DataSource:
        """
        Uploads data of a certain type as parquet and creates a new DataSource.

        Args:
            name (str): name of the data source to be created.
            data (TData): data of type `TData`
            remote_path (str): fully qualified remote path where to upload the data to.

        Returns:
            DataSource: DataSource for the newly uploaded data source
        """

    @classmethod
    def _create_metadata(cls, metadata_args: dict) -> Metadata:
        """
        Create the manifest metadata
        Args:
            metadata_args (dict): a dictionary containing metadata information

        Returns:
            Metadata: the initial metadata
        """

        initial_metadata = Metadata()
        return cls._update_metadata(initial_metadata, metadata_args)

    @classmethod
    def _update_metadata(
        cls,
        metadata: Metadata,
        metadata_args: Optional[Dict[str, Union[str, int, float, bool]]],
    ) -> Metadata:
        """
        Update the manifest metadata
        Args:
            metadata (metadata): the previous component metadata
            metadata_args (dict): a dictionary containing metadata information related to the
            current component
        Returns:
            Metadata: the initial metadata
        """
        metadata_dict = metadata.to_dict()
        for metadata_key, metadata_value in metadata_args.items():
            metadata_dict[metadata_key] = metadata_value
        metadata_dict["branch"] = ""  # TODO: Fill from docker build env var
        metadata_dict["commit_hash"] = ""  # TODO: Fill from docker build env var
        metadata_dict["creation_date"] = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

        return Metadata.from_dict(metadata_dict)

    @classmethod
    def _create_output_dataset(
        cls,
        draft: ExpressDatasetDraft[IndexT, DataT],
        metadata: Metadata,
        save_path: str,
    ) -> DataManifest:
        """
        Processes a dataset draft of a specific type, uploading all local data to storage and
        composing the output manifest.
        """
        if isinstance(draft.index, DataSource):
            index = draft.index
        else:
            remote_path = cls._path_for_upload(metadata, "index")
            index = cls._upload_index(draft.index, remote_path)

        data_sources = {}
        for name, dataset in draft.data_sources.items():
            if isinstance(dataset, DataSource):
                data_sources[name] = dataset
            else:
                remote_path = cls._path_for_upload(metadata, name)
                data_sources[name] = cls._upload_data_source(name, dataset, remote_path)
        manifest = DataManifest(
            index=index, data_sources=data_sources, metadata=metadata
        )
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(save_path).write_text(manifest.to_json(), encoding="utf-8")
        return manifest


class ExpressTransformComponent(ExpressDatasetHandler, Generic[IndexT, DataT]):
    """
    An abstract component that facilitates end-to-end transformation of Express Datasets.
    It can be subclassed or used with a mixin to support reading and writing of a specific data
     type, and to implement specific dataset transformations.
    """

    @classmethod
    def run(cls) -> DataManifest:
        """
        Parses input data, executes the transform, and creates output artifacts.

        Returns:
            DataManifest: the output manifest
        """
        args = cls._parse_args()
        input_dataset = cls._load_dataset(
            input_manifest=DataManifest.from_path(args.input_manifest)
        )
        output_dataset_draft = cls.transform(
            data=input_dataset, extra_args=json.loads(args.extra_args)
        )
        output_manifest = cls._create_output_dataset(
            draft=output_dataset_draft,
            metadata=json.loads(args.metadata),
            save_path=args.output_manifest,
        )
        return output_manifest

    @classmethod
    def _parse_args(cls):
        """
        Parse component arguments
        """
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--input-manifest",
            type=str,
            required=True,
            help="The input data manifest artifact",
        )
        parser.add_argument(
            "--metadata",
            type=str,
            required=True,
            help="The metadata associated with this pipeline run",
        )
        parser.add_argument(
            "--extra-args",
            type=str,
            required=False,
            help="Extra arguments for the component, passed as a json dict string",
        )
        parser.add_argument(
            "--output-manifest",
            type=str,
            required=True,
            help="The output data manifest artifact",
        )
        return parser.parse_args()

    @classmethod
    @abstractmethod
    def transform(
        cls,
        data: ExpressDataset[IndexT, DataT],
        extra_args: Optional[Dict[str, Union[str, int, float, bool]]] = None,
    ) -> ExpressDatasetDraft[IndexT, DataT]:
        """
        Applies transformations to the input dataset and creates a draft for a new dataset.
        The recommended pattern for a transform is to extend the input dataset with a filtered index
        , and/or with additional data sources.
        If the transform generated data that is independent of the input data, or if the output size
         is significantly smaller than the input size, it can be beneficial to create a draft for
          a completely new dataset.

        Args:
            data (ExpressDataset[TIndex, TData]): express dataset providing access to data of a
             given type
            extra_args (Optional[Dict[str, Union[str, int, float, bool]]]): an optional dictionary
             of additional arguments passed in by the pipeline run

        Returns:
            ExpressDatasetDraft[TIndex, TData]: draft of output dataset, to be uploaded after this
             transform completes. Can be created by calling `extend` on an existing dataset, or by
              directly calling the constructor.
        """


class ExpressLoaderComponent(ExpressDatasetHandler, Generic[IndexT, DataT]):
    """
    An abstract component that facilitates creation of a new Express Dataset.
    This will commonly be the first component in an Express Pipeline. It can be subclassed or used
     with a mixin to support loading of a specific data type, and to implement specific dataset
      loaders.
    """

    @classmethod
    def run(cls) -> DataManifest:
        """
        Parses input data, executes the data loader, and creates output artifacts.

        Returns:
            DataManifest: the output manifest
        """
        args = cls._parse_args()
        output_dataset_draft = cls.load(extra_args=json.loads(args.extra_args))
        metadata = cls._create_metadata(metadata_args=json.loads(args.metadata_args))
        # Create metadata
        output_manifest = cls._create_output_dataset(
            draft=output_dataset_draft,
            save_path=args.output_manifest,
            metadata=metadata,
        )
        return output_manifest

    @classmethod
    def _parse_args(cls):
        """
        Parse component arguments
        """
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--metadata-args",
            type=str,
            required=False,
            help="Metadata arguments, passed as a json dict string",
        )
        parser.add_argument(
            "--extra-args",
            type=str,
            required=False,
            help="Extra arguments, passed as a json dict string",
        )
        parser.add_argument(
            "--output-manifest",
            type=str,
            required=True,
            help="The output data manifest artifact",
        )
        return parser.parse_args()

    @classmethod
    @abstractmethod
    def load(
        cls, extra_args: Optional[Dict[str, Union[str, int, float, bool]]] = None
    ) -> ExpressDatasetDraft[IndexT, DataT]:
        """
        Loads data from an arbitrary source to create a draft for a new dataset.

        Args:
            extra_args (Optional[Dict[str, Union[str, int, float, bool]]): an optional dictionary
             of additional arguments passed in by the pipeline run

        Returns:
            ExpressDatasetDraft[TIndex, TData]: draft of output dataset, to be uploaded after this
             loader completes.
        """
