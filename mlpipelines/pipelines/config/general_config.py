"""Dataset creation pipeline config"""
import os

from dataclasses import dataclass


# pylint: disable=invalid-name
@dataclass
class GeneralConfig:
    """
    General configs
    Params:
        GCP_PROJECT_ID (str): the id of the gcp project
        ENV (str): the project run environment (sbx, dev, prd)
    """
    GCP_PROJECT_ID = "storied-landing-366912"
    ENV = os.environ.get('ENV', 'dev')


@dataclass
class KubeflowConfig(GeneralConfig):
    """
    Configs for the Kubeflow cluster
    Params:
        ARTIFACT_BUCKET (str): the GCS bucket used to store the artifacts
        CLUSTER_NAME (str): the name of the k8 cluster hosting KFP
        CLUSTER_ZONE (str): the zone of the k8 cluster hosting KFP
        HOST (str): the kfp host url
    """
    ARTIFACT_BUCKET = f"{GeneralConfig.GCP_PROJECT_ID}-kfp-output"
    CLUSTER_NAME = "kfp-cf"
    CLUSTER_ZONE = "europe-west4-a"
    HOST = "https://2fe14e47821a6eb7-dot-europe-west1.pipelines.googleusercontent.com"
