"""A component that downloads common crawl files."""
import logging
import typing as t

import dask
import dask.dataframe as dd
import pandas as pd
from bs4 import BeautifulSoup
from fondant.component import DaskTransformComponent
from fastwarc.warc import ArchiveIterator, WarcRecordType

from utils.download_utils import download_warc_file
from utils.license_utils import get_license_type, get_license_location
from utils.image_utils import get_images_from_soup, get_unique_images

logger = logging.getLogger(__name__)

dask.config.set(scheduler="processes")

CC_BASE_URL = "http://data.commoncrawl.org"


class CommonCrawlDownloadComponent(DaskTransformComponent):
    """Component that download common crawl files."""

    @staticmethod
    def get_image_info_from_webpage(
        webpage_url: str, webpage_html: str
    ) -> t.Optional[t.List[t.Tuple[str, str, str, str, str]]]:
        """Extracts image urls and license metadata from the parsed html code.
        Args:
            webpage_url: The url of the webpage.
            webpage_html: The html content of the webpage.
        Returns:
            A list of image urls and license metadata.
        """
        try:
            soup = BeautifulSoup(webpage_html, "html.parser")
            for a_tag in soup.find_all("a"):
                if a_tag.has_attr("href"):
                    license_type = get_license_type(a_tag)
                    if license_type is not None:
                        license_location = get_license_location(a_tag)

                        if license_location is None:
                            continue
                        logger.info(
                            f"Found license type: {license_type} at {license_location}"
                        )
                        images = get_images_from_soup(
                            soup, webpage_url, license_type, license_location
                        )
                        logger.info(f"Found {len(images)} images.")

                        unique_images = get_unique_images(images)
                        logger.info(f"Found {len(unique_images)} unique images.")

                        return unique_images

        except Exception as e:
            logger.error(f"Error parsing HTML: {e}")
            return None

    def extract_images(self, file) -> t.List[t.Tuple[str, str, str, str, str]]:
        """Extracts records from a WARC file, optionally converting HTML to plain text.
        Args:
            file: The WARC file.
        Returns:
            A list of webpage records, where each record is a url and content.
        """
        images = []

        def filter_(record):
            if record.headers.get("WARC-Identified-Payload-Type") != "text/html":
                return False
            return True

        for record in ArchiveIterator(
            file,
            record_types=WarcRecordType.response,
            func_filter=filter_,
        ):
            url = record.headers.get("WARC-Target-URI")
            content = record.reader.read().decode("utf-8", "replace")
            if content:
                image_info = self.get_image_info_from_webpage(url, content)
                if image_info:
                    images.extend(image_info)

        return images

    def download_and_extract_warc(
        self, warc_file: str
    ) -> t.List[t.Tuple[str, str, str, str, str]]:
        """Downloads a WARC file and extracts the webpages.
        Args:
            warc_file: The path to the WARC file.
        Returns:
            The extracted images with their licenses and other metadata.
        """
        logger.warning(f"Processing WARC file: {warc_file}...")

        response = download_warc_file(warc_file)
        return self.extract_images(response.raw)

    def download_and_extract_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Download and extract all warc files in a dataframe."""
        return (
            dataframe.apply(
                lambda row: self.download_and_extract_warc(row["warc_url"]),
                axis=1,
            )
            .explode()
            .apply(pd.Series)
        )

    def transform(self, dataframe: dd.DataFrame) -> dd.DataFrame:
        """Concurrently download and extract the WARC files referenced in the provided dataframe."""
        content_df = dataframe.map_partitions(
            self.download_and_extract_dataframe,
            meta=pd.DataFrame(columns=[0, 1, 2, 3, 4]),
        )

        content_df = content_df.dropna()

        content_df.columns = [
            "image_image_url",
            "image_alt_text",
            "image_webpage_url",
            "image_license_type",
            "image_license_location",
        ]

        return content_df
