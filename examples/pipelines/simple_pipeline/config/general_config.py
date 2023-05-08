"""General config"""

import os

from dataclasses import dataclass


@dataclass
class GeneralConfig:
    """
    General configs
    Params:
        GCP_PROJECT_ID (str): GCP project ID
        ENV (str): the project run environment (sbx, dev, prd)
    """

    GCP_PROJECT_ID = "soy-audio-379412"
    ENV = os.environ.get("ENV", "dev")


@dataclass
class KubeflowConfig(GeneralConfig):
    """
    Configs for the Kubeflow cluster
    Params:
        BASE_PATH (str): base path to store the artifacts
        CLUSTER_NAME (str): name of the k8 cluster hosting KFP
        CLUSTER_ZONE (str): zone of the k8 cluster hosting KFP
        HOST (str): kfp host url
    """

    BASE_PATH = "gcs://soy-audio-379412_kfp-artifacts/custom_artifact"
    CLUSTER_NAME = "kfp-fondant"
    CLUSTER_ZONE = "europe-west4-a"
    HOST = "https://52074149b1563463-dot-europe-west1.pipelines.googleusercontent.com"
