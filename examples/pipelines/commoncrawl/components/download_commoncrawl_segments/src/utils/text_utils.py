import logging
from configparser import ConfigParser
import trafilatura

logger = logging.getLogger(__name__)

# Disable trafilatura warnings and logs.
# Some files from the common crawl are empty or do not contain valid HTML.
# Trafilatura handles these scenarios and logs an error message consistently.
trafilatura_logger = logging.getLogger("trafilatura")
trafilatura_logger.setLevel(logging.CRITICAL)


def convert_to_plain_text(
    html: str, extraction_config: ConfigParser, target_language=None
) -> str:
    try:
        return trafilatura.extract(
            html,
            no_fallback=True,
            include_tables=False,
            deduplicate=True,
            target_language=target_language,
            config=extraction_config,
        )
    except Exception as e:
        logger.error(f"Error converting HTML to plain text: {e}")
        return None
