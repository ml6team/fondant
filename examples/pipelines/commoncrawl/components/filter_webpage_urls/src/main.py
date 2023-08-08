import os
import logging
from typing import List, Set

import dask.dataframe as dd
from urllib.parse import urlparse

from fondant.component import DaskTransformComponent
from fondant.executor import DaskTransformExecutor

from utils.file_utils import download_tar_file, get_urls_from_file

logger = logging.getLogger(__name__)

BLACKLIST_DIR = "/tmp/blacklists"
BLACKLIST_URL = "http://dsi.ut-capitole.fr/blacklists/download/blacklists.tar.gz"


def get_blacklisted_domains(blacklist_dir: str, categories: List[str]) -> Set[str]:
    result = set()
    for category in categories:
        path = os.path.join(blacklist_dir, "blacklists", category, "domains")
        urls = get_urls_from_file(path)
        result.update(urls)

    return result


class FilterWebpageUrls(DaskTransformComponent):
    def __init__(self, *_, categories: List[str] = None):
        self.categories = categories

    def transform(self, dataframe: dd.DataFrame) -> dd.DataFrame:
        logger.info("Length of dataframe before filtering: %s", len(dataframe))

        download_tar_file(BLACKLIST_URL, BLACKLIST_DIR)
        blacklisted_domains = get_blacklisted_domains(BLACKLIST_DIR, self.categories)

        dataframe["webpage_base_url"] = dataframe["webpage_url"].apply(
            lambda x: urlparse(x).netloc, meta=("webpage_url", "object")
        )

        filtered_df = dataframe[
            ~dataframe["webpage_base_url"].isin(blacklisted_domains)
        ]
        filtered_df = filtered_df.drop(columns=["webpage_base_url"])

        logger.info("Length of dataframe after filtering: %s", len(filtered_df))
        return filtered_df


if __name__ == "__main__":
    executor = DaskTransformExecutor.from_args()
    executor.execute(FilterWebpageUrls)
