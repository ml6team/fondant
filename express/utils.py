"""
Import utils
"""
import logging
import importlib.util
import importlib.metadata as importlib_metadata
from typing import Union, List

logger = logging.getLogger(__name__)

import importlib.util
import importlib.metadata
from typing import Sequence


def is_module_available(module_name: Union[str, Sequence[str]]) -> List[str]:
    """
    Function that checks if given modules are available
    Args:
        module_name (Sequence[str]): the name of the modules to check
    Returns:
        Sequence[str]: list of missing modules, empty list if all modules are available
    """
    if isinstance(module_name, str):
        module_name = [module_name]

    missing_modules = []
    for module in module_name:
        if importlib.util.find_spec(module) is None:
            missing_modules.append(module)
        else:
            module_version = importlib_metadata.version(module)
            logger.info(f"{module} version {module_version} available.")
    return missing_modules
