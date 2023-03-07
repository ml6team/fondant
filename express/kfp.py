"""General kfp helpers"""
import ast
import logging

import torch

LOGGER = logging.getLogger(__name__)


def parse_kfp_list(kfp_parsed_string: str) -> list:
    """
    This is mainly to resolve issues in passing a list to kfp components. kfp will return a json
    string object instead of a list. This function parses the json string to return the
    original list
    Reference: https://stackoverflow.com/questions/57806505/in-kubeflow-pipelines-how-to
    -send-a-list-of-elements-to-a-lightweight-python-co
    Args:
        kfp_parsed_string (str): the list string to parse with format: '[',l',i','s','t']'
    Returns:
        list: the list representation of the json string
    """
    return ast.literal_eval("".join(kfp_parsed_string))


def get_cuda_availability():
    """Function that checks if a cuda device is available"""
    cuda_available = torch.cuda.is_available()
    LOGGER.info("CUDA device availability:%s", cuda_available)

    if cuda_available:
        LOGGER.info(torch.cuda.get_device_name(0))
        LOGGER.info("CUDA device: %s", torch.cuda.get_device_name(0))
        LOGGER.info("Num of GPUs: %s", torch.cuda.device_count())
        LOGGER.info("Memory Usage:")
        LOGGER.info("Allocated: %s GB", round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1))
        LOGGER.info("Cached: %s GB", round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1))
