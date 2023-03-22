from dataclasses import dataclass

@dataclass
class DatasetLoaderConfig:
    """
    Configs for the dataset loader component
    Params:
        BUCKET_NAME (str): The GCS bucket containing the initial dataset to load
        BLOB_NAME (str): the zone of the k8 cluster hosting KFP
        NAMESPACE (str): The dataset namespace (abbreviation for data source)
    """
    BUCKET_NAME = "express-datasets"
    BLOB_NAME = "initial-clean-cut-dataset"
    NAMESPACE = "cf"