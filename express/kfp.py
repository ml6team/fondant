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
