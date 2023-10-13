import os


def secrets_to_env_vars():
    #adjust here to get the secrets from wherever they are stored
    secrets ={"api":"key"}
    for key, value in secrets.items():
        os.environ[key] = value