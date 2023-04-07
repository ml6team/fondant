"""Storage mounting functionalities"""
import subprocess  # nosec
import logging
from typing import Sequence, List, NamedTuple, Dict, Optional
from enum import Enum
from pathlib import Path

from express.io import create_subprocess_arguments

logger = logging.getLogger(__name__)


class CloudStorageConfig(NamedTuple):
    """
    Defines the configuration for a specific cloud provider's storage system.

    Attributes:
        storage_prefix (str): The prefix used to access the storage system.
        mount_command (str): The command used to mount the storage system.
    """

    storage_prefix: str
    mount_command: str


class CloudProvider(Enum):
    """
    Enumerates the different cloud providers that can be used for remote storage.

    Attributes:
        GCP (CloudStorageConfig): Google Cloud Platform storage configuration.
        AWS (CloudStorageConfig): Amazon Web Services storage configuration.
        AZURE (CloudStorageConfig): Microsoft Azure storage configuration.
    """

    GCP = CloudStorageConfig("gs://", "gcsfuse")
    AWS = CloudStorageConfig("s3://", "s3fs")
    # TODO: double check azure storage prefix
    AZURE = CloudStorageConfig("blob.core.windows.net://", "blobfuse")


def mount_remote_storage(
    *,
    mount_buckets: Sequence[str],
    mount_dir: str,
    mount_command: str,
    mount_args: Optional[List[str]] = None,
    mount_kwargs: Optional[Dict[str, any]] = None,
):
    """
    Function that mounts a remote bucket to a local directory using FUSE
    Args:
        mount_buckets (str): the name of the bucket(s) to be mount
        mount_dir (str): the local directory where to mount the buckets to
        mount_command (str): the command used to mount the storage system
        mount_args (Optional[List[str]]): Additional arguments to pass to the FUSE command.
        mount_kwargs (Optional[Dict[str, any]]): Additional key word arguments to pass
         to the FUSE command.
    """
    # TODO: figure out if we want to pass credentials for accounts to authenticate or whether we
    #  expect the orchestrator service account to have a service accounts with valid permissions
    #  (current scenario)

    if isinstance(mount_buckets, str):
        mount_buckets = [mount_buckets]
    elif not isinstance(mount_buckets, list):
        raise ValueError("The 'mount_buckets' argument must be a list or string.")

    for mount_bucket in mount_buckets:
        mount_path = str(Path(mount_dir, mount_bucket))
        try:
            Path(mount_path).mkdir(parents=True, exist_ok=True)
            extra_args_list = create_subprocess_arguments(
                args=mount_args, kwargs=mount_kwargs
            )
            extra_args_string = " ".join(extra_args_list)
            command = [mount_command, extra_args_string, mount_bucket, mount_path]
            subprocess.run(command, check=True)  # nosec
            logger.info(f"Mounted {mount_bucket} to {mount_path}")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"{mount_command} was not found. Please make sure that you have "
                f"configured the proper installation in your component."
            )
        except Exception as e:
            raise Exception(f"Failed to mount {mount_bucket} to {mount_path}: {e}")


def unmount_remote_storage(mount_dir: str):
    """
    Function that mounts a remote GCS bucket to a local mount path
    Args:
         mount_dir (str): the local directory where to mount the buckets to
    """
    for path in Path(mount_dir).iterdir():
        if path.is_dir():
            try:
                # TODO: double check to see if fusermount is applicable for all cloud providers
                subprocess.run(["fusermount", "-u", path], check=True)  # nosec
                logger.info(f"Unmount content from {path}")
            except subprocess.CalledProcessError as e:
                raise Exception(f"Failed to unmount GCS bucket {path}: {e}")
