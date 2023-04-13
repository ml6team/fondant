"""This module defines commpon I/O utilities."""
import yaml
import typing as t


def load_yaml(yaml_path: str) -> t.Dict:
    """
    Loads a YAML file and returns a dictionary.
    Args:
        yaml_path (str): the path to the yaml path
    """
    try:
        with open(yaml_path) as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise ValueError(f"Failed to load YAML file: {e}")
