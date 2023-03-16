"""
Import utils
"""
import logging
import importlib.util
import importlib.metadata as importlib_metadata
from typing import Union, List

logger = logging.getLogger(__name__)


def is_module_available(module_name: Union[str, List[str]]) -> bool:
    """
    Function that checks if a given module or modules is/are available
    Args:
        module_name (Union[List[str], str]): the name of the module(s) to check
    Returns:
        bool: whether the module(s) is available
    """
    if isinstance(module_name, str):
        module_name = [module_name]

    for module in module_name:
        module_available = importlib.util.find_spec(module) is not None
        if module_available:
            module_version = importlib_metadata.version(module)
            logger.info(f"{module} version {module_version} available.")
        else:
            logger.error(f"Module {module} not found")
            return False
    return True
