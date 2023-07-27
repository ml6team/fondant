import re
import logging
from typing import Any

from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def get_license_location(element: Any) -> str:
    """Returns the license location from the parsed html code.
    Args:
        element: The parsed html code.
    Returns:
        The license location.
    """
    parent = element.parent

    if parent is None:  # could not find an apprioriate tag
        return None

    if (
        parent.name == "footer"
        or parent.find("div", id="footer")
        or parent.find("div", class_="footer")
    ):
        return "footer"
    elif (
        parent.name == "aside"
        or parent.find("div", id="aside")
        or parent.find("div", class_="aside")
    ):
        return "aside"
    elif (
        parent.name == "sidebar"
        or parent.find("div", id="sidebar")
        or parent.find("div", class_="sidebar")
    ):
        return "sidebar"
    else:
        return get_license_location(parent)


def get_license_type_from_creative_commons_url(license_url: str) -> str:
    """Returns the license type from the creative commons url.
    Args:
        license_url: The creative commons url.
    Returns:
        The license type.
    """
    license_split = urlparse(license_url).path.split("/")
    logger.info(f"license_split: {license_split}")

    if "publicdomain" in license_split:
        return "public domain"
    else:
        license = [l for l in license_split if "by" in l]
        return license[0]


def get_license_type_from_fandom_url(a_tag: Any) -> str:
    return a_tag.text


def get_license_type(a_tag: Any) -> str:
    """Returns the license type from the parsed html code.
    Args:
        a_tag: The parsed html code.
    Returns:
        The license type.
    """
    href = a_tag.get("href")

    if "fandom.com/licensing" in href:
        return get_license_type_from_fandom_url(a_tag)
    elif "creativecommons.org" in href:
        return get_license_type_from_creative_commons_url(href)
    else:
        return None
