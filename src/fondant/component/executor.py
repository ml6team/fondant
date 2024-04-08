"""
This Python module defines abstract base class for components in the Fondant data processing
framework, providing a standardized interface for extending loading and transforming components.
The loading component is the first component that loads the initial dataset and the transform
components take care of processing, filtering and extending the data.
"""
import argparse
import json
import logging
import typing as t
from abc import abstractmethod
from distutils.util import strtobool
from pathlib import Path

import dask.dataframe as dd
import pandas as pd
from fsspec import open as fs_open

from fondant.component import (
    Component,
    DaskLoadComponent,
    DaskTransformComponent,
    DaskWriteComponent,
    PandasTransformComponent,
)
from fondant.component.data_io import DaskDataLoader, DaskDataWriter
from fondant.core.component_spec import Argument, OperationSpec
from fondant.core.manifest import Manifest, Metadata

logger = logging.getLogger(__name__)


class Executor(t.Generic[Component]):
    """
    An executor executes a Component.

    Args:
        operation_spec: The operation spec of the component to be executed.
        cache: Flag indicating whether to use caching for intermediate results.
        input_manifest_path: The path to the input manifest file.
        output_manifest_path: The path to the output manifest file.
        metadata: Components metadata dict
        user_arguments: User-defined component arguments.
        input_partition_rows: The number of rows to process in each
        partition of dataframe.
        Partitions are divided based on this number (n rows per partition).
        Set to None for no row limit.
        initialise the dask client, allowing for advanced configuration.
        previous_index: The name of the index column of the previous component.
            Used to remove all previous fields if the component changes the index
    """

    def __init__(
        self,
        operation_spec: OperationSpec,
        *,
        cache: bool,
        input_manifest_path: t.Union[str, Path],
        output_manifest_path: t.Union[str, Path],
        metadata: t.Dict[str, t.Any],
        user_arguments: t.Dict[str, t.Any],
        input_partition_rows: int,
        previous_index: t.Optional[str] = None,
        working_directory: str,
    ) -> None:
        self.operation_spec = operation_spec
        self.cache = cache
        self.input_manifest_path = input_manifest_path
        self.output_manifest_path = output_manifest_path
        self.metadata = Metadata.from_dict(metadata)
        self.user_arguments = user_arguments
        self.input_partition_rows = input_partition_rows
        self.previous_index = previous_index
        self.working_directory = working_directory

    @classmethod
    def from_args(cls) -> "Executor":
        """Create an executor from a passed argument containing the specification as a dict."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--operation_spec", type=json.loads)
        parser.add_argument("--cache", type=lambda x: bool(strtobool(x)))
        parser.add_argument("--input_partition_rows", type=int)
        parser.add_argument("--working_directory", type=str)

        args, _ = parser.parse_known_args()

        if "operation_spec" not in args:
            msg = "Error: The --operation_spec argument is required."
            raise ValueError(msg)

        operation_spec = OperationSpec.from_dict(args.operation_spec)

        working_directory = args.working_directory

        return cls.from_spec(
            operation_spec,
            cache=args.cache,
            input_partition_rows=args.input_partition_rows,
            working_directory=working_directory,
        )

    @classmethod
    def from_spec(
        cls,
        operation_spec: OperationSpec,
        *,
        cache: bool,
        input_partition_rows: int,
        working_directory: str,
    ) -> "Executor":
        """Create an executor from a component spec."""
        args_dict = vars(cls._add_and_parse_args(operation_spec))

        for argument in [
            "operation_spec",
            "input_partition_rows",
            "cache",
            "consumes",
            "produces",
            "working_directory",
        ]:
            args_dict.pop(argument, None)

        input_manifest_path = args_dict.pop("input_manifest_path")
        output_manifest_path = args_dict.pop("output_manifest_path")
        metadata = args_dict.pop("metadata")
        metadata = json.loads(metadata) if metadata else {}

        return cls(
            operation_spec,
            input_manifest_path=input_manifest_path,
            output_manifest_path=output_manifest_path,
            cache=cache,
            metadata=metadata,
            user_arguments=args_dict,
            input_partition_rows=input_partition_rows,
            previous_index=operation_spec.previous_index,
            working_directory=working_directory,
        )

    @classmethod
    def _add_and_parse_args(cls, operation_spec: OperationSpec):
        parser = argparse.ArgumentParser()
        component_arguments = cls._get_component_arguments(operation_spec)

        for arg in component_arguments.values():
            if arg.name in cls.optional_fondant_arguments():
                input_required = False
                default = None
            elif arg.default is not None and arg.optional is False:
                input_required = False
                default = arg.default
            elif arg.default is None and arg.optional is True:
                input_required = False
                default = None
            else:
                input_required = True
                default = None

            parser.add_argument(
                f"--{arg.name}",
                type=arg.parser,
                required=input_required,
                default=default,
                help=arg.description,
            )

        args, _ = parser.parse_known_args()
        args.__dict__ = {
            k: v if v != "None" else None for k, v in args.__dict__.items()
        }

        return args

    @staticmethod
    def optional_fondant_arguments() -> t.List[str]:
        return []

    @staticmethod
    def _get_component_arguments(
        operation_spec: OperationSpec,
    ) -> t.Dict[str, Argument]:
        """
        Get the component arguments as a dictionary representation containing both input and output
            arguments of a component
        Args:
            operation_spec: the operation spec
        Returns:
            Input and output arguments of the component.
        """
        component_arguments: t.Dict[str, Argument] = {}
        component_arguments.update(operation_spec.args)
        return component_arguments

    @abstractmethod
    def _load_or_create_manifest(self) -> Manifest:
        """Abstract method that returns the dataset manifest."""

    @abstractmethod
    def _execute_component(
        self,
        component: Component,
        *,
        manifest: Manifest,
    ) -> t.Union[None, dd.DataFrame]:
        """
        Abstract method to execute a component with the provided manifest.

        Args:
            component: Component instance to execute
            manifest: Manifest describing the input data

        Returns:
            A Dask DataFrame containing the output data
        """

    def _write_data(
        self,
        dataframe: dd.DataFrame,
        *,
        manifest: Manifest,
    ):
        """Create a data writer given a manifest and writes out the index and subsets."""
        data_writer = DaskDataWriter(
            manifest=manifest,
            operation_spec=self.operation_spec,
        )

        data_writer.write_dataframe(dataframe)

    def _get_cache_reference_content(self) -> t.Union[str, None]:
        """
        Get the content of the cache reference file. This file contains the path to the cached
        manifest or empty string if the component is cached without producing any manifest.

        Returns:
            The content of the cache reference file.
        """
        manifest_reference_path = (
            f"{self.working_directory}/{self.metadata.dataset_name}/cache/"
            f"{self.metadata.cache_key}.txt"
        )

        try:
            with fs_open(
                manifest_reference_path,
                mode="rt",
                encoding="utf-8",
            ) as file_:
                return file_.read()

        except FileNotFoundError:
            logger.info("No matching execution for component detected")
            return None

    def _is_previous_cached(self, input_manifest: Manifest) -> bool:
        """
        Checks whether the previous component's output is cached based on its run ID.

        This function compares the run ID of the input manifest
         (representing the previous component) with the run ID of the current component metadata.
        If the run IDs are different, it indicates that the previous component's output belongs to
        another workflow run, implying that it is cached. Otherwise, if the run IDs match, it
        suggests that the previous component was not cached and had to execute to produce the
         current output.

        Args:
            input_manifest: The manifest representing the output of the previous component.

        Returns:
            True if the previous component's output is cached, False otherwise.
        """
        previous_component_id = input_manifest.component_id

        if input_manifest.run_id == self.metadata.run_id:
            logger.info(
                f"Previous component `{previous_component_id}` is not cached."
                f" Invalidating cache for current and subsequent components",
            )
            return False

        logger.info(
            f"Previous component `{previous_component_id}` run was cached. "
            f"Cached workflow id: {input_manifest.run_id}",
        )
        return True

    def _run_execution(
        self,
        component_cls: t.Type[Component],
        input_manifest: Manifest,
    ) -> Manifest:
        logging.info("Executing component")

        component: Component = component_cls(**self.user_arguments)

        component.consumes = self.operation_spec.operation_consumes
        component.produces = self.operation_spec.operation_produces

        state = component.setup()

        output_df = self._execute_component(
            component,
            manifest=input_manifest,
        )

        output_manifest = input_manifest.evolve(
            operation_spec=self.operation_spec,
            run_id=self.metadata.run_id,
            working_directory=self.working_directory,
        )
        self._write_data(dataframe=output_df, manifest=output_manifest)

        component.teardown(state)

        return output_manifest

    def execute(self, component_cls: t.Type[Component]) -> None:
        """
        Execute a component.

        Args:
            component_cls: The class of the component to execute
            working_directory: The working directory where the dataset artifacts will be stored
        """
        input_manifest = self._load_or_create_manifest()

        if self.cache and self._is_previous_cached(input_manifest):
            logger.info("Caching is currently temporarily disabled.")
            cache_reference_content = self._get_cache_reference_content()

            if cache_reference_content is not None:
                logger.info("Skipping component execution")

                if cache_reference_content:
                    output_manifest = Manifest.from_file(cache_reference_content)

                    logger.info(
                        f"Matching execution detected for component. The last execution of the"
                        f" component originated from `{output_manifest.run_id}`.",
                    )
                else:
                    logger.info("Component is cached without producing a manifest")
                    output_manifest = None
            else:
                output_manifest = self._run_execution(component_cls, input_manifest)
        else:
            logger.info("Caching disabled for the component")
            output_manifest = self._run_execution(component_cls, input_manifest)

        if output_manifest:
            self.upload_manifest(output_manifest, save_path=self.output_manifest_path)

        self._upload_cache_reference_content(
            working_directory=self.working_directory,
            dataset_name=self.metadata.dataset_name,
        )

    def _upload_cache_reference_content(
        self,
        working_directory: str,
        dataset_name: str,
    ):
        """
        Write the cache key containing the reference to the location of the written manifest.

        This function creates a file with the format "<cache_key>.txt" at the specified
        'manifest_save_path' to store the manifest location for future retrieval of
        cached component executions.

        Args:
            working_directory: Working directory where the dataset artifacts are stored.
            dataset_name: The name of the dataset.
        """
        cache_reference_path = (
            f"{working_directory}/{dataset_name}/cache/{self.metadata.cache_key}.txt"
        )

        logger.info(
            f"Writing cache key with manifest reference to {cache_reference_path}",
        )

        with fs_open(
            cache_reference_path,
            mode="wt",
            encoding="utf-8",
            auto_mkdir=True,
        ) as file_:
            file_.write(self.cache_reference_content)

    @property
    def cache_reference_content(self) -> str:
        return str(self.output_manifest_path)

    def upload_manifest(self, manifest: Manifest, save_path: t.Union[str, Path]):
        """
        Uploads the manifest to the specified destination.

        Args:
            manifest: The Manifest object to be uploaded.
            save_path: The path where the Manifest object will be saved.

        """
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        manifest.to_file(save_path)
        logger.info(f"Saving output manifest to {save_path}")


class DaskLoadExecutor(Executor[DaskLoadComponent]):
    """Base class for a Fondant load component."""

    def _is_previous_cached(self, input_manifest: Manifest) -> bool:
        return True

    @staticmethod
    def optional_fondant_arguments() -> t.List[str]:
        return ["input_manifest_path", "input_partition_rows"]

    def _load_or_create_manifest(self) -> Manifest:
        return Manifest.create(
            dataset_name=self.metadata.dataset_name,
            run_id=self.metadata.run_id,
            component_id=self.metadata.component_id,
            cache_key=self.metadata.cache_key,
        )

    def _execute_component(
        self,
        component: DaskLoadComponent,
        *,
        manifest: Manifest,
    ) -> dd.DataFrame:
        """This function loads the initial dataframe using the user-provided `load` method.

        Returns:
            A `dd.DataFrame` instance with initial data.
        """
        return component.load()


class TransformExecutor(Executor[Component]):
    """Base class for a Fondant transform component."""

    def _load_or_create_manifest(self) -> Manifest:
        return Manifest.from_file(self.input_manifest_path)

    def _execute_component(
        self,
        component: Component,
        *,
        manifest: Manifest,
    ) -> dd.DataFrame:
        raise NotImplementedError


class DaskTransformExecutor(TransformExecutor[DaskTransformComponent]):
    def _execute_component(
        self,
        component: DaskTransformComponent,
        *,
        manifest: Manifest,
    ) -> dd.DataFrame:
        """
        Load the data based on the manifest using a DaskDataloader and call the transform method to
        process it.

        Returns:
            A `dd.DataFrame` instance with updated data based on the applied data transformations.
        """
        data_loader = DaskDataLoader(
            manifest=manifest,
            operation_spec=self.operation_spec,
            input_partition_rows=self.input_partition_rows,
        )
        dataframe = data_loader.load_dataframe()
        return component.transform(dataframe)


class PandasTransformExecutor(TransformExecutor[PandasTransformComponent]):
    @staticmethod
    def optional_fondant_arguments() -> t.List[str]:
        return ["input_manifest_path", "input_partition_rows"]

    @staticmethod
    def wrap_transform(
        transform: t.Callable,
        *,
        operation_spec: OperationSpec,
    ) -> t.Callable:
        """Factory that creates a function to wrap the component transform function. The wrapper:
        - Skips the transformation if the received partition is empty
        - Removes extra columns from the returned dataframe which are not defined in the component
          spec `produces` section
        - Sorts the columns from the returned dataframe according to the order in the component
          spec `produces` section to match the order in the `meta` argument passed to Dask's
          `map_partitions`.

        Args:
            transform: Transform method to wrap
            operation_spec: Operation specification to base behavior on
        """

        def wrapped_transform(dataframe: pd.DataFrame) -> pd.DataFrame:
            # Columns of operation specification
            columns = [
                name for name, field in operation_spec.operation_produces.items()
            ]

            if not dataframe.empty:
                dataframe = transform(dataframe)
            else:
                logger.info("Received empty partition, skipping transformation.")

            # Drop columns not in specification
            return dataframe[columns]

        return wrapped_transform

    def _execute_component(
        self,
        component: PandasTransformComponent,
        *,
        manifest: Manifest,
    ) -> dd.DataFrame:
        """
        Load the data based on the manifest using a DaskDataloader and call the component's
        transform method for each partition of the data.

        Returns:
            A `dd.DataFrame` instance with updated data based on the applied data transformations.
        """
        data_loader = DaskDataLoader(
            manifest=manifest,
            operation_spec=self.operation_spec,
            input_partition_rows=self.input_partition_rows,
        )
        dataframe = data_loader.load_dataframe()

        # Create meta dataframe with expected format
        meta_dict = {"id": pd.Series(dtype="object")}
        for field_name, field in self.operation_spec.operation_produces.items():
            meta_dict[field_name] = pd.Series(dtype=pd.ArrowDtype(field.type.value))
        meta_df = pd.DataFrame(meta_dict).set_index("id")

        wrapped_transform = self.wrap_transform(
            component.transform,
            operation_spec=self.operation_spec,
        )

        # Call the component transform method for each partition
        dataframe = dataframe.map_partitions(
            wrapped_transform,
            meta=meta_df,
        )

        # Clear divisions if component spec indicates that the index is changed
        if self.previous_index is not None:
            dataframe.clear_divisions()

        return dataframe


class DaskWriteExecutor(Executor[DaskWriteComponent]):
    """Base class for a Fondant write component."""

    @staticmethod
    def optional_fondant_arguments() -> t.List[str]:
        return ["input_partition_rows", "output_manifest_path"]

    def _load_or_create_manifest(self) -> Manifest:
        return Manifest.from_file(self.input_manifest_path)

    def _execute_component(
        self,
        component: DaskWriteComponent,
        *,
        manifest: Manifest,
    ) -> None:
        data_loader = DaskDataLoader(
            manifest=manifest,
            operation_spec=self.operation_spec,
            input_partition_rows=self.input_partition_rows,
        )
        dataframe = data_loader.load_dataframe()
        component.write(dataframe)

    def _write_data(self, dataframe: dd.DataFrame, *, manifest: Manifest):
        """Create a data writer given a manifest and writes out the index and subsets."""

    def upload_manifest(self, manifest: Manifest, save_path: t.Union[str, Path]):
        pass

    @property
    def cache_reference_content(self) -> str:
        return ""


class ExecutorFactory:
    def __init__(self, component: t.Type[Component]):
        self.component = component
        self.component_executor_mapping: t.Dict[str, t.Type[Executor]] = {
            "DaskLoadComponent": DaskLoadExecutor,
            "DaskTransformComponent": DaskTransformExecutor,
            "DaskWriteComponent": DaskWriteExecutor,
            "PandasTransformComponent": PandasTransformExecutor,
        }

    def get_executor(self) -> Executor:
        component_type = self.component.__bases__[0].__name__
        try:
            executor = self.component_executor_mapping[component_type].from_args()
        except KeyError:
            msg = (
                f"The component `{self.component.__name__}` of type `{component_type}` has no"
                f" corresponding executor.\n "
                f"Component executor mapping:"
                f" {json.dumps(self.component_executor_mapping, indent=4)}"
            )
            raise ValueError(msg)
        return executor
