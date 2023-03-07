"""Checkpoint utils"""
import os
import subprocess  # nosec
import logging
from shutil import rmtree
from typing import List

from helpers.logger import get_logger
from helpers import storage_helpers

logger = get_logger(name=__name__, level=logging.INFO)


def sorted_checkpoints(gcs_blobs: List[str]) -> List[str]:
    """
    Function to sort training checkpoints
    Args:
        gcs_blobs(List[str]): list of gcs blobs containing checkpoints
    Returns:
        List[str]: a list of available checkpoints in descending order
         (e.g. [checkpoint-1000, checkpoint-500, ...])
    """
    available_checkpoints = []
    for p in gcs_blobs:
        if not p:
            continue
        p = p.strip('/').split('/')[-1]
        if p.startswith('checkpoint-'):
            available_checkpoints.append(p)
    available_checkpoints.sort(key=lambda p: int(p.split('checkpoint-')[1]), reverse=True)
    return available_checkpoints


def download_checkpoint_to_resume_from(resume_from_checkpoint: str, pretrained_model_gcs_path: str,
                                       finetuned_model_path: str) -> str:
    """
    Function that downloads the checkpoint to resume from to a local gcs directory
    Args:
        resume_from_checkpoint (str):  Whether training should be resumed from a previous
        checkpoint. Use a path saved in `pretrained_model_gcs_path` by `--checkpointing_steps`,
          or `"latest"` to automatically select the last available checkpoint.
        pretrained_model_gcs_path (str): the gcs path of the pretrained model where the
          checkpoints are created
        finetuned_model_path (str): the path to the finetuned model where the checkpoints are
         created
    Returns:
        str: the checkpoint prefix (e.g. 'checkpoint-500') to resume from
    """

    if resume_from_checkpoint == "latest":
        possible_checkpoints = \
            subprocess.run(["gsutil", "ls", pretrained_model_gcs_path],  # nosec
                           capture_output=True, check=True).stdout.decode().split('\n')
        checkpoint_prefix = \
            sorted_checkpoints(possible_checkpoints)[0]
    else:
        checkpoint_prefix = resume_from_checkpoint

    logger.info(
        f"Downloading checkpoint {resume_from_checkpoint} from {pretrained_model_gcs_path}.")
    downloaded_checkpoint_path = os.path.join(finetuned_model_path, resume_from_checkpoint)
    os.makedirs(downloaded_checkpoint_path, exist_ok=True)
    storage_helpers.copy_folder_bulk(
        os.path.join(pretrained_model_gcs_path, resume_from_checkpoint),
        downloaded_checkpoint_path)

    return checkpoint_prefix


def sync_training_checkpoints(finetuned_model_path: str, finetuned_model_gcs_uri: str):
    """
    Function that sync local created checkpoints during training with gcs
    Args:
        finetuned_model_path (str): the path to the finetuned model where the checkpoints are
         created
        finetuned_model_gcs_uri (str): the gcs uri where the model will be uploaded
    """
    if os.path.exists(finetuned_model_path):
        logger.info(f"Syncing checkpoints to {finetuned_model_gcs_uri}. "
                    f"Folder contents: {os.listdir(finetuned_model_path)}")
        subprocess.run(['gsutil', '-m', 'rsync', '-r', finetuned_model_path,  # nosec
                        finetuned_model_gcs_uri])

        # Clean up older checkpoints. They'll remain available in cloud storage, but we want to
        # avoid running out of ephemeral storage.
        available_checkpoints = sorted_checkpoints(os.listdir(finetuned_model_path))
        for to_delete in available_checkpoints[1:]:
            logger.info(f"Local cleanup of old checkpoint: {to_delete}")
            rmtree(os.path.join(finetuned_model_path, to_delete), onerror=logger.info)
    else:
        logger.info(
            f"Checked {finetuned_model_path} for checkpoints, but path doesn't exist yet.")


def download_model_without_checkpoints(pretrained_model_gcs_path: str, finetuned_model_path: str):
    """
    Function that copies a pretrained model (without checkpoints)
    Args:
        pretrained_model_gcs_path (str): the gcs path to the pretrained model which may contain
         checkpoints
        finetuned_model_path (str): the local path to the finetuned model where to download the
         model
    """
    subprocess.run(['gsutil', '-m', 'rsync', '-r', '-x', 'checkpoint*',  # nosec
                    pretrained_model_gcs_path, finetuned_model_path])
    logger.info(
        f"Model downloaded from '{pretrained_model_gcs_path}' to '{finetuned_model_path}'.")
