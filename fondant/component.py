"""
This Python module defines abstract base class for components in the Fondant data processing
framework, providing a standardized interface for extending loading and transforming components.
The loading component is the first component that loads the initial dataset and the transform
components take care of processing, filtering and extending the data.
"""

import argparse
import json
import logging
from pathlib import Path
import typing as t
from abc import ABC, abstractmethod

import dask.dataframe as dd

from fondant.component_spec import FondantComponentSpec, kubeflow2python_type, Argument
from fondant.dataset import FondantDataset
from fondant.manifest import Manifest

logger = logging.getLogger(__name__)


class FondantComponent(ABC):
    """Abstract base class for a Fondant component"""

    def __init__(self):
        self.spec = FondantComponentSpec.from_file("fondant_component.yaml")
        self.args = self._add_and_parse_args()

    def _get_component_arguments(self) -> t.Mapping[str, Argument]:
        """
        Get the component arguments as a dictionary representation containing both input and output
            arguments of a component
        Returns:
            Input and output arguments of the component.
        """
        component_arguments = {}
        kubeflow_component_spec = self.spec.kubeflow_specification
        component_arguments.update(kubeflow_component_spec.input_arguments)
        component_arguments.update(kubeflow_component_spec.output_arguments)
        return component_arguments

    def _add_and_parse_args(self) -> argparse.Namespace:
        """Add and parses the component arguments"""
        parser = argparse.ArgumentParser()
        component_arguments = self._get_component_arguments()

        for arg in component_arguments.values():
            # Input manifest is not required for loading component
            if arg.name == "input_manifest_path":
                input_required = False
            else:
                input_required = True

            parser.add_argument(
                f"--{arg.name}",
                type=kubeflow2python_type[arg.type],
                required=input_required,
                help=arg.description,
            )

        # add metadata
        parser.add_argument(
            "--metadata",
            type=str,
            required=True,
            help="The metadata associated with the pipeline run",
        )

        return parser.parse_args()

    @abstractmethod
    def _load_or_create_manifest(self) -> Manifest:
        """Abstract method that returns the dataset manifest"""

    @abstractmethod
    def _process_dataset(self, dataset: FondantDataset) -> dd.DataFrame:
        """Abstract method that processes the input dataframe of the `FondantDataset` and
        returns another dataframe"""

    def _get_custom_arguments(self):
        """Gets the custom argument names and their values as a dictionary"""

        args = vars(self.args)
        # only keep the custom args of the component
        custom_args = [arg.name for arg in self.spec.args]
        args = {key: value for key, value in args.items() if key in custom_args}

        return args

    def run(self):
        """
        Runs the component.
        """
        input_manifest = self._load_or_create_manifest()
        input_dataset = FondantDataset(input_manifest)

        df = self._process_dataset(input_dataset)

        output_manifest = input_manifest.evolve(component_spec=self.spec)
        output_dataset = FondantDataset(output_manifest)

        # write index and output subsets to remote storage
        output_dataset.write_index(df)
        output_dataset.write_subsets(df, self.spec)

        # write output manifest
        self._upload_manifest(output_manifest, save_path=self.args.output_manifest_path)

    def _upload_manifest(self, manifest: Manifest, save_path: str):
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        manifest.to_file(save_path)


class FondantLoadComponent(FondantComponent):
    """Abstract base class for a Fondant load component"""

    def _load_or_create_manifest(self) -> Manifest:
        # create initial manifest
        # TODO ideally get rid of args.metadata by including them in the storage args, getting
        # run_id based on args.output_manifest_path
        metadata = json.loads(self.args.metadata)
        manifest = Manifest.create(
            base_path=metadata["base_path"],
            run_id=metadata["run_id"],
            component_id=metadata["component_id"],
        )

        return manifest

    @abstractmethod
    def load(self, **kwargs) -> dd.DataFrame:
        """Abstract method that loads the initial dataframe"""

    def _process_dataset(self, dataset: FondantDataset) -> dd.DataFrame:
        """This function loads the initial dataframe sing the user-provided `load` method.

        Returns:
            A `dd.DataFrame` instance with initial data'.
        """
        # Load the dataframe according to the custom function provided to the user
        custom_args = self._get_custom_arguments()
        df = self.load(**custom_args)

        return df


class FondantTransformComponent(FondantComponent):
    """Abstract base class for a Fondant transform component"""

    def _load_or_create_manifest(self) -> Manifest:
        return Manifest.from_file(self.args.input_manifest_path)

    @abstractmethod
    def transform(self, dataframe: dd.DataFrame, **kwargs) -> dd.DataFrame:
        """Abstract method for applying data transformations to the input dataframe"""

    def _process_dataset(self, dataset: FondantDataset) -> dd.DataFrame:
        """
        Loads the input dataframe using the `load_dataframe` method of the provided `FondantDataset`
        instance, and  applies data transformations to it using the `transform` method implemented
        by the derived class. Returns a single dataframe.

        Returns:
            A `dd.DataFrame` instance with updated data based on the applied data transformations.
        """
        df = dataset.load_dataframe(self.spec)
        custom_args = self._get_custom_arguments()
        df = self.transform(dataframe=df, **custom_args)

        return df
