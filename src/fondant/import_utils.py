"""Import utils."""
import importlib.metadata
import importlib.util
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

KFP_IMPORT_ERROR = """
`{0}` requires the kubeflow pipelines (kfp) library but it was not found in your environment.
Please install fondant using pip install fondant [pipelines].
"""


def is_package_available(package_name: str, import_error_msg: str) -> bool:
    """
    Function that checks if a given package is available
    Args:
        package_name (str): the name of the package
        import_error_msg (str): the error message to return if the package is not found
    Returns:
        bool: check if package is available.
    """
    package_available = importlib.util.find_spec(package_name) is not None

    try:
        package_version = importlib.metadata.version(package_name)
        logger.debug(f"Successfully imported {package_name} version {package_version}")
    except importlib.metadata.PackageNotFoundError:
        package_available = False

    if package_available:
        return package_available

    raise ModuleNotFoundError(import_error_msg.format(Path(sys.argv[0]).stem))


def is_kfp_available():
    """Check if 'pandas' is available."""
    return is_package_available("kfp", KFP_IMPORT_ERROR)
