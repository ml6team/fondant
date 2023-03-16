"""
Test scripts for io utils
"""
import pytest
from typing import List, Union
from express.utils import is_module_available


@pytest.mark.parametrize("module_name,expected", [
    ("datasets", True),
    (["datasets"], True),
    (["datasets", "pandas"], True),
    ("invalid_module_name", False),
    (["datasets", "invalid_module_name"], False),
])
def test_is_module_available(module_name: Union[str, List[str]],
                             expected: bool):
    """
    Test function for is_module_available
    """
    assert is_module_available(module_name) == expected
