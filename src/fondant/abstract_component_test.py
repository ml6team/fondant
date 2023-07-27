from abc import ABC, abstractmethod

import pytest


class AbstractComponentTest(ABC):
    @abstractmethod
    def create_component(self):
        """
        This method should be implemented by concrete test classes
        to create the specific component
        that needs to be tested.
        """
        raise NotImplementedError

    @abstractmethod
    def create_input_data(self):
        """This method should be implemented by concrete test classes
        to create the specific input data.
        """
        raise NotImplementedError

    @abstractmethod
    def create_output_data(self):
        """This method should be implemented by concrete test classes
        to create the specific output data.
        """
        raise NotImplementedError

    @pytest.fixture(autouse=True)
    def __setUp(self):
        """
        This method will be run before each test method.
        Add any common setup steps for your components here.
        """
        self.component = self.create_component()
        self.input_data = self.create_input_data()
        self.expected_output_data = self.create_output_data()

    def test_transform(self):
        """
        Default test for the transform method.
        Tests if the transform method executes without errors.
        """
        output = self.component.transform(self.input_data)
        if not output.equals(self.expected_output_data):
            msg = f"Output is not matching expected data\n{output} != {self.expected_output_data}"
            raise Exception(
                msg,
            )
