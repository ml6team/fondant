from fondant.dataset import Dataset


def create_pipeline_with_args(name):
    return Dataset(name=name, base_path="some/path")


def create_pipeline():
    return Dataset(name="test_pipeline", base_path="some/path")


def not_implemented():
    raise NotImplementedError


pipeline = create_pipeline()


number = 1
