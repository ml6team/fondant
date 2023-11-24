"""This component writes an image dataset to the hub."""
import logging
import typing as t
from io import BytesIO

import dask.dataframe as dd
import datasets

# Define the schema for the struct using PyArrow
import huggingface_hub
from datasets.features.features import generate_from_arrow_type
from fondant.component import DaskWriteComponent
from fondant.core.component_spec import ComponentSpec
from PIL import Image

logger = logging.getLogger(__name__)


def convert_bytes_to_image(
    image_bytes: bytes,
    feature_encoder: datasets.Image,
) -> t.Dict[str, t.Any]:
    """
    Function that converts image bytes to hf image format
    Args:
        image_bytes: the images as a bytestring
        feature_encoder: hf image feature encoder
    Returns:
        HF image representation.
    """
    image = Image.open(BytesIO(image_bytes))
    image = feature_encoder.encode_example(image)
    return image


class WriteToHubComponent(DaskWriteComponent):
    def __init__(
        self,
        spec: ComponentSpec,
        *,
        hf_token: str,
        username: str,
        dataset_name: str,
        image_column_names: t.Optional[list],
        column_name_mapping: t.Optional[dict],
    ):
        """
        Args:
            spec: Dynamic component specification describing the dataset to write
            hf_token: The hugging face token used to write to the hub
            username: The username under which to upload the dataset
            dataset_name: The name of the dataset to upload
            image_column_names: A list containing the subset image column names. Used to format the
            image fields to HF hub format
            column_name_mapping: Mapping of the consumed fondant column names to the written hub
             column names.
        """
        huggingface_hub.login(token=hf_token)

        repo_id = f"{username}/{dataset_name}"
        self.repo_path = f"hf://datasets/{repo_id}"

        logger.info(f"Creating HF dataset repository under ID: '{repo_id}'")
        huggingface_hub.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

        self.spec = spec
        self.image_column_names = image_column_names
        self.column_name_mapping = column_name_mapping

    def write(
        self,
        dataframe: dd.DataFrame,
    ):
        # Get columns to write and schema
        write_columns = []
        schema_dict = {}
        for field_name, field in self.spec.consumes.items():
            column_name = field.name
            write_columns.append(column_name)
            if self.image_column_names and column_name in self.image_column_names:
                schema_dict[column_name] = datasets.Image()
            else:
                schema_dict[column_name] = generate_from_arrow_type(
                    field.type.value,
                )

        schema = datasets.Features(schema_dict).arrow_schema
        dataframe = dataframe[write_columns]

        # Map image column to hf data format
        feature_encoder = datasets.Image(decode=True)

        if self.image_column_names is not None:
            for image_column_name in self.image_column_names:
                dataframe[image_column_name] = dataframe[image_column_name].map(
                    lambda x: convert_bytes_to_image(x, feature_encoder),
                    meta=(image_column_name, "object"),
                )

        # Map column names to hf data format
        if self.column_name_mapping:
            dataframe = dataframe.rename(columns=self.column_name_mapping)

        # Write dataset to the hub
        dd.to_parquet(dataframe, path=f"{self.repo_path}/data", schema=schema)
