import logging
import html_text

logger = logging.getLogger(__name__)


def convert_to_plain_text(html: str) -> str:
    try:
        return html_text.extract_text(html)
    except Exception as e:
        logger.error(f"Error converting HTML to plain text: {e}")
        return None
