"""
This component writes an image dataset to the hub.
"""
import logging

import huggingface_hub
import dask.dataframe as dd

from fondant.component import TransformComponent
from fondant.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


class WriteToHubComponent(TransformComponent):
    def transform(
        self,
        dataframe: dd.DataFrame,
        *,
        hf_token: str,
        username: str,
        dataset_name: str,
    ) -> dd.DataFrame:
        """
        Args:
            dataframe: Dask dataframe
            hf_token: The hugging face token used to write to the hub
            username: The username under which to upload the dataset
            dataset_name: The name of the dataset to upload

        Returns:
            dataset
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
        schema = {}
        for subset_name, subset in self.spec.consumes.items():
            write_columns.extend([f"{subset_name}_{field}" for field in subset.fields])
            # Get schema
            subset_schema = {
                f"{subset_name}_{field.name}": field.type.value
                for field in subset.fields.values()
            }

            schema.update(subset_schema)

        dataframe_hub = dataframe[write_columns]
        dd.to_parquet(dataframe_hub, path=f"{repo_path}/data", schema=schema)

        return dataframe


if __name__ == "__main__":
    component = WriteToHubComponent.from_file()
    component.run()
