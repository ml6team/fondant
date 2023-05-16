"""
This component write an images dataset to the hub.
"""
import logging

import huggingface_hub
import dask.dataframe as dd

from fondant.component import FondantTransformComponent
from fondant.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


class WriteToHubComponent(FondantTransformComponent):
    def transform(
        self,
        *,
        dataframe: dd.DataFrame,
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

        # Create hub
        repo_id = f"{username}/{dataset_name}"
        repo_path = f"hf://datasets/{repo_id}"
        logger.info(f"Creating HF dataset repository under ID: '{repo_id}'")
        huggingface_hub.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

        dataframe_hub = dataframe[["images_data", "captions_data"]]
        dataframe_hub = dataframe_hub.rename(
            columns={"images_data": "images", "captions_data": "captions"}
        )

        schema = {"images": "binary", "captions": "string"}
        dd.to_parquet(dataframe_hub, path=f"{repo_path}/data", schema=schema)

        return dataframe


if __name__ == "__main__":
    component = WriteToHubComponent.from_file()
    component.run()
