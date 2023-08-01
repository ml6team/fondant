import json
from glob import glob

import pandas
from fondant.component import Component
from fondant.executor import Executor


def load_fixtures(path):
    test_configurations = []
    fixture_list = glob(path)
    for fixture in fixture_list:
        with open(fixture) as file:
            fixture_dict = json.load(file)

        user_argmuments = fixture_dict["user_arguments"]
        input_data = {
            tuple(key.split("_")): value for key, value in fixture_dict["input"].items()
        }
        expected_out = {
            tuple(key.split("_")): value
            for key, value in fixture_dict["output"].items()
        }

        test_configurations.append((user_argmuments, input_data, expected_out))

    return test_configurations

class TestComponentExecuter(Executor[Component]):
    def __init__(self, user_arguments: t.Dict[str, t.Any], input_data: t.Dict):
        self.user_arguments = user_arguments
        self.input_data = input_data

    def execute(self, component_cls: t.Type[Component]) -> pandas.DataFrame:
        """Execute a component.

        Args:
            component_cls: The class of the component to execute.
        """
        component = component_cls(None, **self.user_arguments)

        input_dataframe = dd.from_dict(self.input_data, npartitions=2)

        if isinstance(component, PandasTransformComponent):
            output_df = component.transform(input_dataframe.compute())

        elif isinstance(component, DaskTransformComponent):
            output_df = component.transform(input_dataframe()).compute()

        else:
            msg = "Non support component type."
            raise NotImplementedError(msg)

        return output_df
