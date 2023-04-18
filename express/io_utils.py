"""This module defines commpon I/O utilities."""
import yaml
import typing as t


def load_yaml(yaml_path: str) -> t.Dict:
    """
    Loads a YAML file and returns a dictionary.
    Args:
        yaml_path (str): the path to the yaml file
    """
    try:
        with open(yaml_path) as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise e
