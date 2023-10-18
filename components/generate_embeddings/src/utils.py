import os


def to_env_vars(api_keys: dict):
    for key, value in api_keys.items():
        os.environ[key] = value
