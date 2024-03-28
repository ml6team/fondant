from fondant.dataset import Dataset


def create_dataset_with_args(name):
    return Dataset.create("load_from_parquet", dataset_name=name)


def create_dataset():
    return Dataset.create("load_from_parquet", dataset_name="test_dataset")


def not_implemented():
    raise NotImplementedError


workspace = create_dataset()


number = 1
