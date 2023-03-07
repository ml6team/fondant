"""
Test scripts for logger functionalities
"""
import logging
import pytest

from express_components.helpers.logger import configure_logging


@pytest.mark.parametrize("log_level, expected_level", [
    ("DEBUG", logging.DEBUG),
    ("INFO", logging.INFO),
    ("WARNING", logging.WARNING),
    ("ERROR", logging.ERROR),
    ("CRITICAL", logging.CRITICAL),
])
def test_configure_logging(log_level, expected_level):
    configure_logging(log_level)

    logger = logging.getLogger(__name__)

    assert logger.root.level == expected_level
