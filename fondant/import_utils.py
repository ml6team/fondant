"""Import utils."""
import importlib.metadata
import importlib.util
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

PANDAS_IMPORT_ERROR = """
`{0}` requires the pandas library but it was not found in your environment. Please install fondant
 using the 'pandas' extra.
"""

DATASETS_IMPORT_ERROR = """
`{0}` requires the 🤗 Datasets library but it was not found in your environment.
Please install fondant using the 'datasets' extra.
Note that if you have a local folder named `datasets` or a local python file named
 `datasets.py` in your current working directory, python may try to import this instead of the 🤗 
 Datasets library. You should rename this folder or that python file if that's the case.
  Please note that you may need to restart your runtime after installation.
"""

KFP_IMPORT_ERROR = """
`{0}` requires the kubeflow pipelines (kfp) library but it was not found in your environment.
Please install fondant using the 'pipelines' extra.
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
    else:
        raise ModuleNotFoundError(import_error_msg.format(Path(sys.argv[0]).stem))


def is_datasets_available():
    """Check if 'datasets' is available."""
    return is_package_available("datasets", DATASETS_IMPORT_ERROR)


def is_pandas_available():
    """Check if 'pandas' is available."""
    return is_package_available("pandas", PANDAS_IMPORT_ERROR)


def is_kfp_available():
    """Check if 'pandas' is available."""
    return is_package_available("kfp", KFP_IMPORT_ERROR)
