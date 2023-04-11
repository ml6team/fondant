"""
This file is the entrypoint of the component. It will parse all arguments
and give them to the actual core of the component.
"""
import argparse
import json
import os
import tempfile
import importlib
from abc import abstractmethod
from pathlib import Path
from typing import Dict, Optional, TypeVar, Generic, Union

import datasets
from datasets import load_dataset

from express.data_source import DataSource, Metadata
from express.storage_interface import StorageHandlerModule

STORAGE_MODULE_PATH = StorageHandlerModule().to_dict()[
    os.environ.get("CLOUD_ENV", "GCP")
]
STORAGE_HANDLER = importlib.import_module(STORAGE_MODULE_PATH).StorageHandler()

IndexT = TypeVar("IndexT")
DataT = TypeVar("DataT")


class Manifest:
    """
    A class representing a data manifest.

    A data manifest consists of a single index and one or more data sources.
    """

    def __init__(self, index, data_sources, metadata):
        metadata = self._create_metadata(metadata)
        self.index = self._create_index(index, metadata)
        self.data_sources = self._create_data_sources(data_sources, metadata)
        self.metadata = metadata

    def _path_for_upload(self, metadata: Metadata, name: str) -> str:
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

    # TODO this is framework specific
    @staticmethod
    def _upload_parquet(
        data: datasets.Dataset, name: str, remote_path: str
    ) -> DataSource:
        with tempfile.TemporaryDirectory() as temp_folder:
            # TODO: uploading without writing to temp file
            # TODO: sharded parquet? not sure if we should shard the index or only the data sources
            dataset_path = f"{temp_folder}/{name}.parquet"

            data.to_parquet(path_or_buf=dataset_path)

            fully_qualified_blob_path = f"{remote_path}/{name}.parquet"
            STORAGE_HANDLER.copy_file(
                source_file=dataset_path, destination=fully_qualified_blob_path
            )

            return DataSource(
                location=fully_qualified_blob_path,
                len=len(data),
                column_names=data.column_names,
            )

    @classmethod
    def _upload_index(cls, index: datasets.Dataset, remote_path: str) -> DataSource:
        data_source = cls._upload_parquet(
            data=index, name="index", remote_path=remote_path
        )
        return data_source

    @classmethod
    def _upload_data_source(
        cls,
        name: str,
        data: datasets.Dataset,
        remote_path: str,
    ) -> DataSource:
        data_source = cls._upload_parquet(data=data, name=name, remote_path=remote_path)
        return data_source

    def _create_index(self, index, metadata):
        if isinstance(index, DataSource):
            pass
        else:
            remote_path = self._path_for_upload(metadata, "index")
            index = self._upload_index(index, remote_path)

        return index

    def _create_data_sources(self, data_sources, metadata):
        for name, dataset in data_sources.items():
            if isinstance(dataset, DataSource):
                data_sources[name] = dataset
            else:
                remote_path = self._path_for_upload(metadata, name)
                data_sources[name] = self._upload_data_source(
                    name, dataset, remote_path
                )

        return data_sources

    def add_data_sources(self, data_sources: Dict[str, DataT]):
        """
        Add one or more data sources to the manifest.
        """
        for name, dataset in data_sources.items():
            self.with_data_source(name, dataset, replace_ok=False)

    def with_data_source(
        self, name: str, data: Union[DataT, DataSource], replace_ok=False
    ):
        """
        Adds a new data source or replaces a preexisting data source with the same name.

        Args:
            name (str): Name of the data source.
            data (Union[TData, DataSource]): Local data of type `TData`, or a preexisting
             `DataSource`.
            replace_ok (bool): Default=False. Whether to replace a preexisting Data Source of the
            same name, if such a Data Source exists.
        """
        if (name in self.data_sources) and (not replace_ok):
            raise ValueError(
                f"A conflicting data source with identifier {name} is already set "
                f"in this draft. Data sources on a dataset draft can be replaced "
                f"after it's been constructed."
            )
        # TODO: verify same namespace?
        if isinstance(data, DataSource):
            self.data_sources[name] = data
        else:
            remote_path = self._path_for_upload(self.metadata, name)
            self.data_sources[name] = self._upload_data_source(name, data, remote_path)

    def update_index(self, index):
        """
        Updates the index of the manifest.
        """
        self.index = self._create_index(index, self.metadata)

    # TODO this is framework specific
    def load_index(self) -> datasets.Dataset:
        """Function that loads in the index"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_parquet_path = STORAGE_HANDLER.copy_file(self.index.location, tmp_dir)

            # we specify "train" here to get a `Dataset` instead of a `DatasetDict`
            dataset = load_dataset(
                "parquet", data_files=local_parquet_path, split="train"
            )

            return dataset

    def load(self, data_source: str, index: Optional[IndexT] = None, **kwargs) -> DataT:
        """
        Load data from a named data source.

        Args:
            data_source (str): A named data source from the input manifest.
            for_index (Optional[TIndex]): Pass in an index to filter the data on. By default, the
             original Dataset index is used. This argument can be used to use a different
              index instead.
            kwargs (dict): Additional keyword arguments forwarded to the _load_data_source method.

        Returns:
            TData: Data of type TData
        """
        if data_source not in self.data_sources:
            raise ValueError(
                f"Named source {data_source} not part of recognised data sources: "
                f"{self.data_sources.keys()}."
            )
        if index is None:
            index = self.load_index()
        return self._load_data_source(self.data_sources[data_source], index, **kwargs)

    # TODO this is actually framework specific
    @staticmethod
    def _load_data_source(
        data_source: DataSource,
        index_filter: datasets.Dataset,
        **kwargs,
    ) -> datasets.Dataset:
        """Function that loads in a data source"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_source_location = data_source.location

            local_parquet_path = STORAGE_HANDLER.copy_parquet(
                data_source_location, tmp_dir
            )

            if "columns" in kwargs:
                if "index" not in kwargs["columns"]:
                    raise ValueError(
                        "Please also include the index when specifying columns"
                    )

            dataset = load_dataset(
                "parquet",
                data_files=local_parquet_path,
                split="train",
                **kwargs,
            )

            if index_filter:
                index = index_filter["index"]
                return dataset.filter(lambda example: example["index"] in index)

            return dataset

    def _create_metadata(self, metadata_args: dict) -> Metadata:
        """
        Create the manifest metadata
        Args:
            metadata_args (dict): a dictionary containing metadata information

        Returns:
            Metadata: the initial metadata
        """

        initial_metadata = Metadata()
        return self._update_metadata(initial_metadata, metadata_args)

    def _update_metadata(
        self,
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
        metadata_dict["git branch"] = os.environ.get("GIT_BRANCH")
        metadata_dict["commit sha"] = os.environ.get("COMMIT_SHA")
        metadata_dict["build timestamp"] = os.environ.get("BUILD_TIMESTAMP")

        return Metadata.from_dict(metadata_dict)

    def upload(self, save_path):
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(save_path).write_text(self.to_json(), encoding="utf-8")
        return None

    def to_json(self):
        # based on https://stackoverflow.com/questions/3768895/how-to-make-a-class-json-serializable
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    @classmethod
    def from_json(cls, json_file):
        # TODO perhaps use this instead
        # manifest = Path(manifest_path).read_text(encoding="utf-8")

        with open(json_file, encoding="utf-8") as file_:
            manifest_dict = json.load(file_)

        index = DataSource.from_dict(manifest_dict["index"])
        data_sources = {
            name: DataSource.from_dict(data_source)
            for name, data_source in manifest_dict["data_sources"].items()
        }

        return cls(index, data_sources, manifest_dict["metadata"])


class ExpressTransformComponent(Generic[IndexT, DataT]):
    """
    An abstract component that facilitates end-to-end transformation of Express Datasets.
    It can be subclassed or used with a mixin to support reading and writing of a specific data
     type, and to implement specific dataset transformations.
    """

    @classmethod
    def run(cls) -> Manifest:
        """
        Parses input data, executes the transform, and creates output manifest.

        Returns:
            Manifest: the output manifest
        """
        args = cls._parse_args()
        # load manifest
        input_manifest = Manifest.from_json(args.input_manifest)
        # update metadata based on args.metadata
        input_manifest.metadata = input_manifest._update_metadata(
            input_manifest.metadata, json.loads(args.metadata)
        )
        # transform
        output_manifest = cls.transform(
            manifest=input_manifest,
            args=json.loads(args.args),
        )
        # create output manifest
        output_manifest.upload(save_path=args.output_manifest)

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
            help="Path to the input data manifest",
        )
        parser.add_argument(
            "--args",
            type=str,
            required=False,
            help="Extra arguments for the component, passed as a json dict string",
        )
        parser.add_argument(
            "--metadata",
            type=str,
            required=True,
            help="The metadata associated with this pipeline run",
        )
        parser.add_argument(
            "--output-manifest",
            type=str,
            required=True,
            help="Path to store the output data manifest",
        )
        return parser.parse_args()

    @classmethod
    @abstractmethod
    def transform(
        cls,
        manifest: Manifest,
        extra_args: Optional[Dict[str, Union[str, int, float, bool]]] = None,
    ) -> Manifest:
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
            Manifest[TIndex, TData]: draft of output dataset, to be uploaded after this
             transform completes. Can be created by calling `extend` on an existing dataset, or by
              directly calling the constructor.
        """


class ExpressLoaderComponent(Generic[IndexT, DataT]):
    """
    An abstract component that facilitates creation of a new output manifest.
    This will commonly be the first component in an Express Pipeline. It can be subclassed or used
    with a mixin to support loading of a specific data type, and to implement specific dataset
    loaders.
    """

    @classmethod
    def run(cls) -> Manifest:
        """
        Parses input data, executes the data loader, and creates output manifest.

        Returns:
            Manifest: the output manifest
        """
        args = cls._parse_args()
        # create manifest
        output_manifest = cls.load(
            args=json.loads(args.args), metadata=json.loads(args.metadata)
        )
        # upload manifest
        output_manifest.upload(save_path=args.output_manifest)

    @classmethod
    def _parse_args(cls):
        """
        Parse component arguments
        """
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--args",
            type=str,
            required=False,
            help="Extra arguments, passed as a json dict string",
        )
        parser.add_argument(
            "--metadata",
            type=str,
            required=True,
            help="Metadata, passed as a json dict string",
        )
        parser.add_argument(
            "--output-manifest",
            type=str,
            required=True,
            help="Path to store the output manifest",
        )
        return parser.parse_args()

    @classmethod
    @abstractmethod
    def load(
        cls,
        args: Optional[Dict[str, Union[str, int, float, bool]]] = None,
        metadata=None,
    ) -> Manifest:
        """
        Loads data from an arbitrary source to create an output manifest.

        Args:
            args (Optional[Dict[str, Union[str, int, float, bool]]): an optional dictionary
             of additional arguments passed in by the pipeline run
            metadata (Optional[Dict[str, Union[str, int, float, bool]]): an optional dictionary
                of metadata passed in by the pipeline run

        Returns:
            Manifest: output manifest, to be uploaded after this loader completes.
        """
