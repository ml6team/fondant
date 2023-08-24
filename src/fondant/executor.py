"""
This Python module defines abstract base class for components in the Fondant data processing
framework, providing a standardized interface for extending loading and transforming components.
The loading component is the first component that loads the initial dataset and the transform
components take care of processing, filtering and extending the data.
"""

import argparse
import ast
import json
import logging
import typing as t
from abc import abstractmethod
from pathlib import Path

import dask.dataframe as dd
import pandas as pd

from fondant.component import (
    Component,
    DaskLoadComponent,
    DaskTransformComponent,
    DaskWriteComponent,
    PandasTransformComponent,
)
from fondant.component_spec import Argument, ComponentSpec, kubeflow2python_type
from fondant.data_io import DaskDataLoader, DaskDataWriter
from fondant.manifest import Manifest, Metadata
from fondant.schema import validate_partition_number

logger = logging.getLogger(__name__)


class Executor(t.Generic[Component]):
    """An executor executes a Component."""

    def __init__(
        self,
        spec: ComponentSpec,
        *,
        cache: bool,
        input_manifest_path: t.Union[str, Path],
        output_manifest_path: t.Union[str, Path],
        metadata: t.Dict[str, t.Any],
        user_arguments: t.Dict[str, t.Any],
        input_partition_rows: t.Optional[t.Union[str, int]] = None,
    ) -> None:
        self.spec = spec
        self.cache = cache
        self.input_manifest_path = input_manifest_path
        self.output_manifest_path = output_manifest_path
        self.metadata = Metadata.from_dict(metadata)
        self.user_arguments = user_arguments
        self.input_partition_rows = input_partition_rows

    @classmethod
    def from_args(cls) -> "Executor":
        """Create an executor from a passed argument containing the specification as a dict."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--component_spec", type=json.loads)
        parser.add_argument("--cache", type=ast.literal_eval)
        parser.add_argument("--input_partition_rows", type=validate_partition_number)
        args, _ = parser.parse_known_args()

        if "component_spec" not in args:
            msg = "Error: The --component_spec argument is required."
            raise ValueError(msg)

        component_spec = ComponentSpec(args.component_spec)
        input_partition_rows = args.input_partition_rows
        cache = args.cache

        return cls.from_spec(
            component_spec,
            cache=cache,
            input_partition_rows=input_partition_rows,
        )

    @classmethod
    def from_spec(
        cls,
        component_spec: ComponentSpec,
        *,
        cache: bool,
        input_partition_rows: t.Optional[t.Union[str, int]],
    ) -> "Executor":
        """Create an executor from a component spec."""
        args_dict = vars(cls._add_and_parse_args(component_spec))

        if "component_spec" in args_dict:
            args_dict.pop("component_spec")

        if "input_partition_rows" in args_dict:
            args_dict.pop("input_partition_rows")

        if "cache" in args_dict:
            args_dict.pop("cache")

        input_manifest_path = args_dict.pop("input_manifest_path")
        output_manifest_path = args_dict.pop("output_manifest_path")
        metadata = args_dict.pop("metadata")
        metadata = json.loads(metadata) if metadata else {}

        return cls(
            component_spec,
            input_manifest_path=input_manifest_path,
            output_manifest_path=output_manifest_path,
            cache=cache,
            metadata=metadata,
            user_arguments=args_dict,
            input_partition_rows=input_partition_rows,
        )

    @classmethod
    def _add_and_parse_args(cls, spec: ComponentSpec):
        parser = argparse.ArgumentParser()
        component_arguments = cls._get_component_arguments(spec)

        for arg in component_arguments.values():
            if arg.name in cls.optional_fondant_arguments():
                input_required = False
                default = None
            elif arg.default is not None:
                input_required = False
                default = arg.default
            else:
                input_required = True
                default = None

            parser.add_argument(
                f"--{arg.name}",
                type=kubeflow2python_type(arg.type),  # type: ignore
                required=input_required,
                default=default,
                help=arg.description,
            )

        args, _ = parser.parse_known_args()
        return args

    @staticmethod
    def optional_fondant_arguments() -> t.List[str]:
        return []

    @staticmethod
    def _get_component_arguments(spec: ComponentSpec) -> t.Dict[str, Argument]:
        """
        Get the component arguments as a dictionary representation containing both input and output
            arguments of a component
        Args:
            spec: the component spec
        Returns:
            Input and output arguments of the component.
        """
        component_arguments: t.Dict[str, Argument] = {}
        kubeflow_component_spec = spec.kubeflow_specification
        component_arguments.update(kubeflow_component_spec.input_arguments)
        component_arguments.update(kubeflow_component_spec.output_arguments)
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

    def _write_data(self, dataframe: dd.DataFrame, *, manifest: Manifest):
        """Create a data writer given a manifest and writes out the index and subsets."""
        data_writer = DaskDataWriter(
            manifest=manifest,
            component_spec=self.spec,
        )

        data_writer.write_dataframe(dataframe)

    def _load_cached_output_manifest(self) -> "Manifest":
        """Function that returns the cached output manifest."""
        raise NotImplementedError

    def _has_matching_execution(self) -> bool:
        """Function that checks if there is an existing previous matching execution."""
        # TODO: implement
        return False

    def execute(self, component_cls: t.Type[Component]) -> None:
        """Execute a component.

        Args:
            component_cls: The class of the component to execute
        """
        matching_execution_exists = self._has_matching_execution()

        if matching_execution_exists:
            logger.info("Previous matching execution found")
        else:
            logger.info("No previous matching execution found")

        if self.cache:
            logger.info("Caching for the component is disabled")
        else:
            logger.info("Caching for the component is enabled")

        if self.cache is False and matching_execution_exists:
            logging.info("Cached component run. Skipping component execution")
            output_manifest = self._load_cached_output_manifest()
        else:
            logging.info("Executing component")
            input_manifest = self._load_or_create_manifest()
            component = component_cls(self.spec, **self.user_arguments)
            output_df = self._execute_component(component, manifest=input_manifest)
            output_manifest = input_manifest.evolve(component_spec=self.spec)
            self._write_data(dataframe=output_df, manifest=output_manifest)

        self.upload_manifest(output_manifest, save_path=self.output_manifest_path)

    def upload_manifest(self, manifest: Manifest, save_path: t.Union[str, Path]):
        """
        Uploads the manifest to the specified destination.

        If the save_path points to the kubeflow output artifact temporary path,
        it will be saved both in a specific base path and the native kfp artifact path.

        Args:
            manifest: The Manifest object to be uploaded.
            save_path: The path where the Manifest object will be saved.

        """
        is_kubeflow_output = (
            str(save_path) == "/tmp/outputs/output_manifest_path/data"  # nosec
        )

        if is_kubeflow_output:
            # Save to the expected base path directory
            save_path_base_path = (
                f"{manifest.base_path}/{manifest.pipeline_name}/{manifest.run_id}/"
                f"{manifest.component_id}/manifest.json"
            )
            Path(save_path_base_path).parent.mkdir(parents=True, exist_ok=True)
            manifest.to_file(save_path_base_path)
            logger.info(f"Saving output manifest to {save_path_base_path}")
            # Write manifest to the native kfp artifact path that will be passed as an artifact
            # and read by the next component
            manifest.to_file(save_path)
        else:
            # Local runner
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            manifest.to_file(save_path)
            logger.info(f"Saving output manifest to {save_path}")


class DaskLoadExecutor(Executor[DaskLoadComponent]):
    """Base class for a Fondant load component."""

    @staticmethod
    def optional_fondant_arguments() -> t.List[str]:
        return ["input_manifest_path"]

    def _load_or_create_manifest(self) -> Manifest:
        component_id = self.spec.name.lower().replace(" ", "_")
        return Manifest.create(
            pipeline_name=self.metadata.pipeline_name,
            base_path=self.metadata.base_path,
            run_id=self.metadata.run_id,
            component_id=component_id,
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
            component_spec=self.spec,
            input_partition_rows=self.input_partition_rows,
        )
        dataframe = data_loader.load_dataframe()
        return component.transform(dataframe)


class PandasTransformExecutor(TransformExecutor[PandasTransformComponent]):
    @staticmethod
    def wrap_transform(transform: t.Callable, *, spec: ComponentSpec) -> t.Callable:
        """Factory that creates a function to wrap the component transform function. The wrapper:
        - Converts the columns to hierarchical format before passing the dataframe to the
          transform function
        - Removes extra columns from the returned dataframe which are not defined in the component
          spec `produces` section
        - Sorts the columns from the returned dataframe according to the order in the component
          spec `produces` section to match the order in the `meta` argument passed to Dask's
          `map_partitions`.
        - Flattens the returned dataframe columns.

        Args:
            transform: Transform method to wrap
            spec: Component specification to base behavior on
        """

        def wrapped_transform(dataframe: pd.DataFrame) -> pd.DataFrame:
            # Switch to hierarchical columns
            dataframe.columns = pd.MultiIndex.from_tuples(
                tuple(column.split("_")) for column in dataframe.columns
            )

            # Call transform method
            dataframe = transform(dataframe)

            # Drop columns not in specification
            columns = [
                (subset_name, field)
                for subset_name, subset in spec.produces.items()
                for field in subset.fields
            ]
            dataframe = dataframe[columns]

            # Switch to flattened columns
            dataframe.columns = [
                "_".join(column) for column in dataframe.columns.to_flat_index()
            ]
            return dataframe

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
            component_spec=self.spec,
            input_partition_rows=self.input_partition_rows,
        )
        dataframe = data_loader.load_dataframe()

        # Create meta dataframe with expected format
        meta_dict = {"id": pd.Series(dtype="object")}
        for subset_name, subset in self.spec.produces.items():
            for field_name, field in subset.fields.items():
                meta_dict[f"{subset_name}_{field_name}"] = pd.Series(
                    dtype=pd.ArrowDtype(field.type.value),
                )
        meta_df = pd.DataFrame(meta_dict).set_index("id")

        wrapped_transform = self.wrap_transform(component.transform, spec=self.spec)

        # Call the component transform method for each partition
        dataframe = dataframe.map_partitions(
            wrapped_transform,
            meta=meta_df,
        )

        # Clear divisions if component spec indicates that the index is changed
        if self._infer_index_change():
            dataframe.clear_divisions()

        return dataframe

    def _infer_index_change(self) -> bool:
        """Infer if this component changes the index based on its component spec."""
        if not self.spec.accepts_additional_subsets:
            return True
        if not self.spec.outputs_additional_subsets:
            return True
        for subset in self.spec.consumes.values():
            if not subset.additional_fields:
                return True
        return any(
            not subset.additional_fields for subset in self.spec.produces.values()
        )


class DaskWriteExecutor(Executor[DaskWriteComponent]):
    """Base class for a Fondant write component."""

    @staticmethod
    def optional_fondant_arguments() -> t.List[str]:
        return ["output_manifest_path"]

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
            component_spec=self.spec,
            input_partition_rows=self.input_partition_rows,
        )
        dataframe = data_loader.load_dataframe()
        component.write(dataframe)

    def _write_data(self, dataframe: dd.DataFrame, *, manifest: Manifest):
        """Create a data writer given a manifest and writes out the index and subsets."""

    def upload_manifest(self, manifest: Manifest, save_path: t.Union[str, Path]):
        pass


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
