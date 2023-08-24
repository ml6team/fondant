import logging
from configparser import ConfigParser
from io import BytesIO
from typing import List

import httpx
import trafilatura
from configs import VALID_COMMONCRAWL_INDEX_FILTERS, VALID_OPERATORS
from warcio.archiveiterator import ArchiveIterator

logger = logging.getLogger(__name__)

def extract_bytes_from_warc_file_http(warc_filename: str,
                                      warc_record_offset: int,
                                      warc_record_length: int,
                                      prefix: str = "https://data.commoncrawl.org/"):
    """Extracts a specified range of bytes from a WARC file over HTTP."""
    url = f"{prefix}{warc_filename}"
    headers = {"Range": f"bytes={warc_record_offset}-{warc_record_offset + warc_record_length - 1}"}

    with httpx.Client() as client:
        response = client.get(url, headers=headers)

    with BytesIO(response.content) as warc_stream:
        try:
            for record in ArchiveIterator(warc_stream):
                if "response" in record.content_type and record.format == "warc":
                    return record.content_stream().read().decode()
        except Exception as e:
            logger.info(f"Error during WARC file read: {e}")
            return None
    return None


def extract_html(content: str, trafilatura_config: ConfigParser) -> str:
    """Extracts structured data from HTML content using Trafilatura."""
    logger.info(f"Extracting HTML...{content[:10]}")
    return trafilatura.extract(
        content,
        no_fallback=True,
        include_tables=False,
        deduplicate=True,
        config=trafilatura_config,
    )

def validate_commoncrawl_index_filters(filters: List) -> List:
    """Validates that the filters are correct."""
    logger.info(f"Validating filters: {filters}")
    filters = [(d["field"], d["operator"], d["value"]) for d in filters]

    try:
        if not isinstance(filters, list):
            msg = "filters must be a list"
            raise TypeError(msg)

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
