import logging
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

BASE_URL = "https://data.commoncrawl.org/"


def convert_to_plain_text(html: str) -> str:
    """Converts an HTML string to plain text.
    Args:
        html: The HTML string.
    Returns:
        The plain text string.
    """
    try:
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text()
    except Exception as e:
        logger.error(f"Error converting HTML to plain text: {e}")
        return None
