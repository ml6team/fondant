"""This component writes an image dataset to the hub."""
import logging
import typing as t
from io import BytesIO

import dask.dataframe as dd
import datasets

# Define the schema for the struct using PyArrow
import huggingface_hub
from PIL import Image

from fondant.component import WriteComponent
from fondant.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def convert_bytes_to_image(image_bytes: bytes, feature_encoder: datasets.Image) -> t.Dict[
    str, t.Any]:
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


class WriteToHubComponent(WriteComponent):
    def write(
            self,
            dataframe: dd.DataFrame,
            *,
            hf_token: str,
            username: str,
            dataset_name: str,
            image_column_name: t.Optional[str],
            column_name_mapping: t.Optional[dict]
    ):
        """
        Args:
            dataframe: Dask dataframe
            hf_token: The hugging face token used to write to the hub
            username: The username under which to upload the dataset
            dataset_name: The name of the dataset to upload
            image_column_name: the name of the image column. Used to format to HF hub format
            column_name_mapping: mapping between the subset columns and the mapping of the final
            column names.
        """
        # login
        huggingface_hub.login(token=hf_token)

        # Create HF dataset repository
        repo_id = f"{username}/{dataset_name}"
        repo_path = f"hf://datasets/{repo_id}"
        logger.info(f"Creating HF dataset repository under ID: '{repo_id}'")
        huggingface_hub.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

        # Get columns to write and schema
        write_columns = []
        schema_dict = {}
        for subset_name, subset in self.spec.consumes.items():
            for field in subset.fields.values():
                column_name = f"{subset_name}_{field.name}"
                write_columns.append(column_name)
                if column_name == image_column_name:
                    schema_dict[column_name] = datasets.Image()
                else:
                    schema_dict[column_name] = datasets.Value(str(field.type.value))

        schema = datasets.Features(schema_dict).arrow_schema
        dataframe = dataframe[write_columns]

        # Map image column to hf data format
        feature_encoder = datasets.Image(decode=True)

        if image_column_name:
            if column_name_mapping:
                invert_column_name_mapping = \
                    {final_dataset_column: subset_column for subset_column, final_dataset_column
                     in column_name_mapping.items()}

                if image_column_name in invert_column_name_mapping:
                    image_column_name = invert_column_name_mapping[image_column_name]

            dataframe[image_column_name] = dataframe[image_column_name].map(
                lambda x: convert_bytes_to_image(x, feature_encoder),
                meta=(image_column_name, schema)
            )

        # Map column names to hf data format
        if column_name_mapping:
            dataframe = dataframe.rename(columns=column_name_mapping)

        # Write dataset to the hub
        dd.to_parquet(dataframe, path=f"{repo_path}/data", schema=schema)


if __name__ == "__main__":
    component = WriteToHubComponent.from_args()
    component.run()
