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
    def _process_dataset(self, dataset: FondantDataset) -> FondantDataset:
        """Abstract method that processes the input dataframe and updates the Fondant Dataset
        with the new or loaded subsets"""

    def run(self):
        """
        Runs the component.
        """
        manifest = self._load_or_create_manifest()
        dataset = FondantDataset(manifest)
        dataset = self._process_dataset(dataset)
        dataset.upload(save_path=self.args.output_manifest_path)


class FondantLoadComponent(FondantComponent):
    """Abstract base class for a Fondant load component"""

    def _load_or_create_manifest(self) -> Manifest:
        # create initial manifest
        # TODO ideally get rid of args.metadata by including them in the storage args, getting
        #  run_id based on args.output_manifest_path
        metadata = json.loads(self.args.metadata)
        manifest = Manifest.create(
            base_path=metadata["base_path"],
            run_id=metadata["run_id"],
            component_id=metadata["component_id"],
        )

        # evolve manifest based on component spec
        manifest = manifest.evolve(self.spec)

        return manifest

    @abstractmethod
    def load(self, args: argparse.Namespace) -> dd.DataFrame:
        """Abstract method that loads the initial dataframe"""

    def _process_dataset(self, dataset: FondantDataset) -> FondantDataset:
        """This function takes in a FondantDataset object and processes the initial input dataframe
         by loading it using the user-provided load function. It then adds an initial index and
        subsets to the dataset, as specified by the component specification, and uploads
         the updated dataset to remote storage. The function returns the updated FondantDataset
         object
        Returns:
            A `FondantDataset` instance with updated data based on the applied data transformations.
        """
        # Load the dataframe according to the custom function provided to the user
        df = self.load(self.args)

        # Write index and output subsets to remote storage
        dataset.write_index(df)
        dataset.write_subsets(df, self.spec)

        return dataset


class FondantTransformComponent(FondantComponent):
    """Abstract base class for a Fondant transform component"""

    def _load_or_create_manifest(self) -> Manifest:
        return Manifest.from_file(self.args.input_manifest_path)

    @abstractmethod
    def transform(
        self, args: argparse.Namespace, dataframe: dd.DataFrame
    ) -> dd.DataFrame:
        """Abstract method for applying data transformations to the input dataframe"""

    def _process_dataset(self, dataset: FondantDataset) -> FondantDataset:
        """Applies data transformations to the input dataframe and updates the Fondant Dataset with
         the new or loaded subsets.
        Loads the input dataframe using the `load_data` method of the provided `FondantDataset`
         instance, and  applies data transformations to it using the `transform` method implemented
          by the derived class.

        After data transformations, the index of the dataset may need to be updated or new subsets
        may need to be added, depending on the component specification. Once the necessary updates
        have been performed, the updated dataset is returned.

        Returns:
            A `FondantDataset` instance with updated data based on the applied data transformations.
        """
        df = dataset.load_dataframe(self.spec)
        df = self.transform(args=self.args, dataframe=df)
        # evolve manifest
        dataset.manifest = dataset.manifest.evolve(self.spec)

        # TODO update index, potentially add new subsets (functionality still missing)
        # Write index and output subsets and write them to remote storage

        return dataset
