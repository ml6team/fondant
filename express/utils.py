"""
Import utils
"""
import logging
import importlib.util
import importlib.metadata
from typing import Sequence, Union, List

import pynvml

logger = logging.getLogger(__name__)


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
            module_version = importlib.metadata.version(module)
            logger.info(f"{module} version {module_version} available.")
    return missing_modules


def get_cuda_availability():
    """Function that checks if a cuda device is available"""

    def _round_bytes_to_gb(byte_size):
        return round(byte_size / 1024**3, 1)

    try:
        logger.info("Driver Version: %s", pynvml.nvmlSystemGetDriverVersion())
        device_cnt = pynvml.nvmlDeviceGetCount()
        logger.info("Found %s cuda device(s)", device_cnt)
        for device_idx in range(device_cnt):
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            logger.info("Device: %s", pynvml.nvmlDeviceGetName(handle))
            logger.info("Total memory: %s GB", _round_bytes_to_gb(info.total))
            logger.info("Free memory: %s GB", _round_bytes_to_gb(info.free))
            logger.info("Used memory: %s GB", _round_bytes_to_gb(info.used))
    except pynvml.NVMLError:
        logger.warning("Cuda device(s) not found.")
