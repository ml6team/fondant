import logging
from urllib.parse import urlparse
from typing import Any, List

logger = logging.getLogger(__name__)


def get_full_image_url(image_url: str, webpage_url: str) -> str:
    """Returns the full image url if not already provided.
    Args:
        image_url: The image url.
        webpage_url: The url of the webpage.
    Returns:
        The full image url.
    """
    if "http" not in image_url or image_url[0] == "/":
        parsed_webpage_url = urlparse(webpage_url)
        image_url = (
            f"{parsed_webpage_url.scheme}://{parsed_webpage_url.netloc}{image_url}"
        )

        try:
            pos = image_url.index("?")
            image_url = image_url[:pos]
        except:
            logger.info("No query parameter found in the image URL.")

    return image_url


def get_image_info(
    a_tag: Any, webpage_url: str, license_type: str, license_location: str
) -> List[str]:
    """Returns the image url, alt text, webpage url, and license type.
    Args:
        a_tag: The parsed html code.
        webpage_url: The url of the webpage.
        license_type: The license type.
    Returns:
        A list of image url, alt text, webpage url, and license type.
    """
    img_tag = a_tag.find("img")

    if img_tag and img_tag.has_attr("src"):
        img_src = get_full_image_url(img_tag["src"], webpage_url)
        img_alt = img_tag.get("alt", "")
        return [img_src, img_alt, webpage_url, license_type, license_location]

    return None


def get_images_from_soup(
    soup: Any, webpage_url: str, license_type: str, license_location: str
) -> List[List[str]]:
    """Returns a list of image urls from the parsed html code.
    Args:
        soup: The parsed html code.
        webpage_url: The url of the webpage.
        license_type: The license type.
    Returns:
        A list of image urls."""
    image_info = []
    for a_tag in soup.find_all("a"):
        img_info = get_image_info(a_tag, webpage_url, license_type, license_location)
        if img_info:
            image_info.append(img_info)

    logger.info(f"Found {len(image_info)} images.")
    return image_info


def get_unique_images(images: List[List[str]]) -> List[List[str]]:
    """Returns a list of unique images.
    Args:
        images: A list of images.
    Returns:
        A list of unique images.
    """
    unique_images = []
    for image in images:
        if image not in unique_images:
            unique_images.append(image)
    return unique_images
