import re
import logging

import pandas as pd
from bs4 import BeautifulSoup
from typing import List

from fondant.component import PandasTransformComponent

from utils.license_utils import get_license_type, get_license_location
from utils.image_utils import get_images_from_soup, get_unique_images

logger = logging.getLogger(__name__)


def get_image_info_from_webpage(webpage_url: str, webpage_html: str) -> List[List[str]]:
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


class ExtractImageLicenses(PandasTransformComponent):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extracts image url and license from the HTML content.
        Args:
            df: A pandas dataframe with the webpage url and html content.
        Returns:
            A pandas dataframe with the image url and license metadata.
        """
        df = (
            df.apply(
                lambda row: get_image_info_from_webpage(
                    row[("webpage", "url")], row[("webpage", "html")]
                ),
                axis=1,
                result_type="expand",
            )
            .explode(0)
            .apply(pd.Series)
        )

        df = df.dropna()

        df.columns = [
            ("image", "image_url"),
            ("image", "alt_text"),
            ("image", "webpage_url"),
            ("image", "license_type"),
            ("image", "license_location"),
        ]

        return df


if __name__ == "__main__":
    component = ExtractImageLicenses.from_args()
    component.run()
