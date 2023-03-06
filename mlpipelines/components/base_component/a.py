from helpers.logger import get_logger
import logging
logger = get_logger(name=__name__, level=logging.INFO)


logger.info(
    "A total of %s images out of %s (%s%%) were filtered",
    1,
    2,
    3)