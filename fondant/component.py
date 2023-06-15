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
from abc import ABC, abstractmethod
from pathlib import Path

import dask.dataframe as dd
import pandas as pd

from fondant.component_spec import Argument, ComponentSpec, kubeflow2python_type
from fondant.data_io import DaskDataLoader, DaskDataWriter
from fondant.manifest import Manifest

logger = logging.getLogger(__name__)


class Component(ABC):
    """Abstract base class for a Fondant component."""

    def __init__(
        self,
        spec: ComponentSpec,
        *,
        input_manifest_path: t.Union[str, Path],
        output_manifest_path: t.Union[str, Path],
        metadata: t.Dict[str, t.Any],
        user_arguments: t.Dict[str, Argument],
    ) -> None:
        self.spec = spec
        self.input_manifest_path = input_manifest_path
        self.output_manifest_path = output_manifest_path
        self.metadata = metadata
        self.user_arguments = user_arguments

    @classmethod
    def from_file(
        cls, path: t.Union[str, Path] = "../fondant_component.yaml"
    ) -> "Component":
        """Create a component from a component spec file.

        Args:
            path: Path to the component spec file
        """
        component_spec = ComponentSpec.from_file(path)
        return cls.from_spec(component_spec)

    @classmethod
    def from_args(cls) -> "Component":
        """Create a component from a passed argument containing the specification as a dict."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--component_spec", type=json.loads)
        args, _ = parser.parse_known_args()

        if "component_spec" not in args:
            raise ValueError("Error: The --component_spec argument is required.")

        component_spec = ComponentSpec(args.component_spec)

        return cls.from_spec(component_spec)

    @classmethod
    def from_spec(cls, component_spec: ComponentSpec) -> "Component":
        """Create a component from a component spec."""
        args_dict = vars(cls._add_and_parse_args(component_spec))

        if "component_spec" in args_dict:
            args_dict.pop("component_spec")
        input_manifest_path = args_dict.pop("input_manifest_path")
        output_manifest_path = args_dict.pop("output_manifest_path")
        metadata = args_dict.pop("metadata")

        metadata = json.loads(metadata) if metadata else {}

        return cls(
            component_spec,
            input_manifest_path=input_manifest_path,
            output_manifest_path=output_manifest_path,
            metadata=metadata,
            user_arguments=args_dict,
        )

    @classmethod
    def _add_and_parse_args(cls, spec: ComponentSpec):
        parser = argparse.ArgumentParser()
        component_arguments = cls._get_component_arguments(spec)

        for arg in component_arguments.values():
            if arg.name in cls.optional_fondant_arguments():
                input_required = False
                default = None
            elif arg.default:
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

        return parser.parse_args()

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
    def _process_dataset(self, manifest: Manifest) -> t.Union[None, dd.DataFrame]:
        """Abstract method that processes the manifest and
        returns another dataframe.
        """

    def _write_data(self, dataframe: dd.DataFrame, *, manifest: Manifest):
        """Create a data writer given a manifest and writes out the index and subsets."""
        data_writer = DaskDataWriter(manifest=manifest, component_spec=self.spec)
        data_writer.write_dataframe(dataframe)

    def run(self):
        """Runs the component."""
        input_manifest = self._load_or_create_manifest()

        output_df = self._process_dataset(manifest=input_manifest)

        output_manifest = input_manifest.evolve(component_spec=self.spec)

        self._write_data(dataframe=output_df, manifest=output_manifest)

        self.upload_manifest(output_manifest, save_path=self.output_manifest_path)

    def upload_manifest(self, manifest: Manifest, save_path: str):
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        manifest.to_file(save_path)


class LoadComponent(Component):
    """Base class for a Fondant load component."""

    @staticmethod
    def optional_fondant_arguments() -> t.List[str]:
        return ["input_manifest_path"]

    def _load_or_create_manifest(self) -> Manifest:
        # create initial manifest
        # TODO ideally get rid of args.metadata by including them in the storage args

        component_id = self.spec.name.lower().replace(" ", "_")
        manifest = Manifest.create(
            base_path=self.metadata["base_path"],
            run_id=self.metadata["run_id"],
            component_id=component_id,
        )

        return manifest

    @abstractmethod
    def load(self, *args, **kwargs) -> dd.DataFrame:
        """Abstract method that loads the initial dataframe."""

    def _process_dataset(self, manifest: Manifest) -> dd.DataFrame:
        """This function loads the initial dataframe sing the user-provided `load` method.

        Returns:
            A `dd.DataFrame` instance with initial data'.
        """
        # Load the dataframe according to the custom function provided to the user
        df = self.load(**self.user_arguments)

        return df


class TransformComponent(Component):
    """Base class for a Fondant transform component."""

    def _load_or_create_manifest(self) -> Manifest:
        return Manifest.from_file(self.input_manifest_path)

    @abstractmethod
    def transform(self, *args, **kwargs) -> dd.DataFrame:
        """
        Abstract method for applying data transformations to the input dataframe.

        Args:
            args: The dataframe will be passed in as a positional argument
            kwargs: Arguments provided to the component are passed as keyword arguments
        """

    def _process_dataset(self, manifest: Manifest) -> dd.DataFrame:
        """
        Load the data based on the manifest using a DaskDataloader and call the transform method to
        process it.

        Returns:
            A `dd.DataFrame` instance with updated data based on the applied data transformations.
        """


class DaskTransformComponent(TransformComponent):
    @abstractmethod
    def transform(self, *args, **kwargs) -> dd.DataFrame:
        """
        Abstract method for applying data transformations to the input dataframe.

        Args:
            args: A Dask dataframe will be passed in as a positional argument
            kwargs: Arguments provided to the component are passed as keyword arguments
        """

    def _process_dataset(self, manifest: Manifest) -> dd.DataFrame:
        """
        Load the data based on the manifest using a DaskDataloader and call the transform method to
        process it.

        Returns:
            A `dd.DataFrame` instance with updated data based on the applied data transformations.
        """
        data_loader = DaskDataLoader(manifest=manifest, component_spec=self.spec)
        df = data_loader.load_dataframe()
        df = self.transform(dataframe=df, **self.user_arguments)
        return df


class PandasTransformComponent(TransformComponent):
    def setup(self, *args, **kwargs):
        """Called once for each instance of the Component class. Use this to set up resources
        such as database connections.
        """
        return

    @abstractmethod
    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method for applying data transformations to the input dataframe.
        Called once for each partition of the data.

        Args:
            dataframe: A Pandas dataframe containing a partition of the data
        """

    def _process_dataset(self, manifest: Manifest) -> dd.DataFrame:
        """
        Load the data based on the manifest using a DaskDataloader and call the transform method to
        process it.

        Returns:
            A `dd.DataFrame` instance with updated data based on the applied data transformations.
        """
        data_loader = DaskDataLoader(manifest=manifest, component_spec=self.spec)
        df = data_loader.load_dataframe()

        # Call the component setup method with user provided argument
        self.setup(**self.user_arguments)

        # Create meta dataframe with expected format
        meta_dict = {"id": pd.Series(dtype="object")}
        for subset_name, subset in self.spec.produces.items():
            for field_name, field in subset.fields.items():
                print(field.type.value)
                meta_dict[f"{subset_name}_{field_name}"] = pd.Series(
                    # dtype=f"{field.type.value}[pyarrow]"
                    dtype=pd.ArrowDtype(field.type.value)
                )
        meta_df = pd.DataFrame(meta_dict).set_index("id")

        # Call the component transform method for each partition
        df = df.map_partitions(
            self.transform,
            meta=meta_df,
        )

        # Clear divisions if component spec indicates that the index is changed
        if self._infer_index_change():
            df.clear_divisions()

        return df

    def _infer_index_change(self) -> bool:
        """Infer if this component changes the index based on its component spec."""
        if not self.spec.accepts_additional_subsets:
            return True
        if not self.spec.outputs_additional_subsets:
            return True
        for subset in self.spec.consumes.values():
            if not subset.additional_fields:
                return True
        for subset in self.spec.produces.values():
            if not subset.additional_fields:
                return True
        return False


class WriteComponent(Component):
    """Base class for a Fondant write component."""

    @staticmethod
    def optional_fondant_arguments() -> t.List[str]:
        return ["output_manifest_path"]

    def _load_or_create_manifest(self) -> Manifest:
        return Manifest.from_file(self.input_manifest_path)

    @abstractmethod
    def write(self, *args, **kwargs):
        """
        Abstract method to write a dataframe to a final custom location.

        Args:
            args: The dataframe will be passed in as a positional argument
            kwargs: Arguments provided to the component are passed as keyword arguments
        """

    def _process_dataset(self, manifest: Manifest) -> None:
        """
        Creates a DataLoader using the provided manifest and loads the input dataframe using the
        `load_dataframe` instance, and  applies data transformations to it using the `transform`
        method implemented by the derived class. Returns a single dataframe.

        Returns:
            A `dd.DataFrame` instance with updated data based on the applied data transformations.
        """
        data_loader = DaskDataLoader(manifest=manifest, component_spec=self.spec)
        df = data_loader.load_dataframe()
        self.write(dataframe=df, **self.user_arguments)

    def _write_data(self, dataframe: dd.DataFrame, *, manifest: Manifest):
        """Create a data writer given a manifest and writes out the index and subsets."""
        pass

    def upload_manifest(self, manifest: Manifest, save_path: str):
        pass
