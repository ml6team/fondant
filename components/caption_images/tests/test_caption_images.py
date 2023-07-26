import inspect

import pytest
from caption_images.src.main import CaptionImagesComponent
from fondant.component import (
    BaseComponent,
    DaskLoadComponent,
    DaskTransformComponent,
    DaskWriteComponent,
    PandasTransformComponent,
)


def get_base_class_name(class_object):
    return class_object.__base__.__name__
component_class_map = {
    "BaseComponent": BaseComponent,
    "DaskLoadComponent": DaskLoadComponent,
    "DaskTransformComponent": DaskTransformComponent,
    "DaskWriteComponent": DaskWriteComponent,
    "PandasTransformComponent": PandasTransformComponent,
}


class Test_AbstractComponent:
    @pytest.fixture(
        scope="class", params=[CaptionImagesComponent],
    )  # move the component classes here
    def get_base_component(self, request):
        """
        Method to get the base component class
        of the component_object.
        """
        component_object = request.param
        base_component_object = component_class_map[
            get_base_class_name(component_object)
        ]
        return base_component_object, component_object

    def test_component(self, get_base_component):
        base_component_object, component_object = get_base_component
        methods = [
            method
            for method in dir(base_component_object)
            if not method.startswith("_")
            and callable(getattr(base_component_object, method))
            and not inspect.isbuiltin(getattr(base_component_object, method))
        ]
        # check if component object has these methods
        for method_name in methods:
            assert callable(
                getattr(component_object, method_name),
            ), f"{component_object.__class__.__name__} should have {method_name} method"

        attributes = [
            attr
            for attr in dir(base_component_object)
            if not attr.startswith("_")
            and not callable(getattr(base_component_object, attr))
        ]

        # check if component object has these attributes
        for attribute in attributes:
            assert getattr(
                component_object, attribute,
            ), f"{component_object.__class__.__name__} should have {attribute} attribute"
