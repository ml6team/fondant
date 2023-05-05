"""Utils file with useful methods."""

import logging
import os
import sys

LOG_LEVEL = os.environ.get("LOG_LEVEL", default="INFO")


def configure_logging(log_level=LOG_LEVEL) -> None:
    """Configure the root logger based on config settings."""
    logger = logging.getLogger()

    # set loglevel
    level = logging.getLevelName(log_level)
    logger.setLevel(level)

    # logging stdout handler
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(
        logging.Formatter(fmt="[%(asctime)s | %(name)s | %(levelname)s] %(message)s")
    )
    logger.addHandler(handler)
