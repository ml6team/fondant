import io
import logging
import typing as t

import trafilatura
from configs import VALID_COMMONCRAWL_INDEX_FILTERS, VALID_OPERATORS
from trafilatura.settings import use_config
from warcio.archiveiterator import ArchiveIterator

logger = logging.getLogger(__name__)

TRAFILATURE_CONFIG = use_config()
TRAFILATURE_CONFIG.set("DEFAULT", "EXTRACTION_TIMEOUT", "0")


def read_warc_content(warc_stream: io.BytesIO) -> t.Optional[str]:
    try:
        for record in ArchiveIterator(warc_stream):
            if "response" in record.content_type and record.format == "warc":
                return record.content_stream().read().decode()
    except Exception as e:
        logger.info(f"Error during WARC file read: {e}")
    return None


def extract_html(content: str) -> str:
    """Extracts structured data from HTML content using Trafilatura."""
    return trafilatura.extract(
        content,
        no_fallback=True,
        include_tables=False,
        deduplicate=True,
        config=TRAFILATURE_CONFIG,
    )


def parse_commoncrawl_index_filters(filters: t.List) -> t.List:
    """Parse and validate the provided filters."""
    logger.info(f"Validating filters: {filters}")

    try:
        if not isinstance(filters, list):
            msg = "filters must be a list"
            raise TypeError(msg)

        filters = [(d["field"], d["operator"], d["value"]) for d in filters]

        for filter_tuple in filters:
            if not isinstance(filter_tuple, tuple):
                msg = "filters must be a list of tuples"
                raise TypeError(msg)

            field, operator, value = filter_tuple
            if field not in VALID_COMMONCRAWL_INDEX_FILTERS:
                msg = f"Invalid field: {field} in filter expression: {filter_tuple}"
                raise ValueError(msg)

            if operator not in VALID_OPERATORS:
                msg = f"Invalid operator: {operator} in filter expression: {filter_tuple}"
                raise ValueError(msg)

        return filters
    except Exception as e:
        raise e
