from fondant.pipeline import Pipeline


def create_pipeline_with_args(name):
    return Pipeline(pipeline_name=name, base_path="some/path")


def create_pipeline():
    return Pipeline(pipeline_name="test_pipeline", base_path="some/path")


pipeline = create_pipeline()
