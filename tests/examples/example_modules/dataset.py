from fondant.dataset import Dataset


def create_dataset_with_args(name):
    return Dataset(name)


def create_dataset():
    return Dataset("test_dataset")


def not_implemented():
    raise NotImplementedError


workspace = create_dataset()


number = 1
