import typing as t

import pytest
from fondant.component import PandasTransformComponent
from fondant.core.component_spec import Argument
from fondant.core.exceptions import UnsupportedTypeAnnotation
from fondant.pipeline.argument_inference import infer_arguments


def test_no_init():
    class TestComponent(PandasTransformComponent):
        pass

    assert infer_arguments(TestComponent) == {}


def test_no_arguments():
    class TestComponent(PandasTransformComponent):
        def __init__(self, **kwargs):
            pass

    assert infer_arguments(TestComponent) == {}


def test_missing_types():
    class TestComponent(PandasTransformComponent):
        def __init__(self, *, argument, **kwargs):
            pass

    assert infer_arguments(TestComponent) == {
        "argument": Argument(
            name="argument",
            type=str,
            optional=False,
        ),
    }


def test_types():
    class TestComponent(PandasTransformComponent):
        def __init__(
            self,
            *,
            str_argument: str,
            int_argument: int,
            float_argument: float,
            bool_argument: bool,
            dict_argument: dict,
            list_argument: list,
            **kwargs,
        ):
            pass

    assert infer_arguments(TestComponent) == {
        "str_argument": Argument(
            name="str_argument",
            type=str,
            optional=False,
        ),
        "int_argument": Argument(
            name="int_argument",
            type=int,
            optional=False,
        ),
        "float_argument": Argument(
            name="float_argument",
            type=float,
            optional=False,
        ),
        "bool_argument": Argument(
            name="bool_argument",
            type=bool,
            optional=False,
        ),
        "dict_argument": Argument(
            name="dict_argument",
            type=dict,
            optional=False,
        ),
        "list_argument": Argument(
            name="list_argument",
            type=list,
            optional=False,
        ),
    }


def test_optional_types():
    class TestComponent(PandasTransformComponent):
        def __init__(
            self,
            *,
            str_argument: t.Optional[str] = "",
            int_argument: t.Optional[int] = 1,
            float_argument: t.Optional[float] = 1.0,
            bool_argument: t.Optional[bool] = False,
            dict_argument: t.Optional[dict] = None,
            list_argument: t.Optional[list] = None,
            **kwargs,
        ):
            pass

    assert infer_arguments(TestComponent) == {
        "str_argument": Argument(
            name="str_argument",
            type=str,
            optional=True,
            default="",
        ),
        "int_argument": Argument(
            name="int_argument",
            type=int,
            optional=True,
            default=1,
        ),
        "float_argument": Argument(
            name="float_argument",
            type=float,
            optional=True,
            default=1.0,
        ),
        "bool_argument": Argument(
            name="bool_argument",
            type=bool,
            optional=True,
            default=False,
        ),
        "dict_argument": Argument(
            name="dict_argument",
            type=dict,
            optional=True,
            default=None,
        ),
        "list_argument": Argument(
            name="list_argument",
            type=list,
            optional=True,
            default=None,
        ),
    }


def test_parametrized_types_old():
    class TestComponent(PandasTransformComponent):
        def __init__(
            self,
            *,
            dict_argument: t.Dict[str, t.Any],
            list_argument: t.Optional[t.List[int]] = None,
            **kwargs,
        ):
            pass

    assert infer_arguments(TestComponent) == {
        "dict_argument": Argument(
            name="dict_argument",
            type=dict,
            optional=False,
            default=None,
        ),
        "list_argument": Argument(
            name="list_argument",
            type=list,
            optional=True,
            default=None,
        ),
    }


def test_parametrized_types_new():
    class TestComponent(PandasTransformComponent):
        def __init__(
            self,
            *,
            dict_argument: dict[str, t.Any],
            list_argument: list[int] | None = None,
            **kwargs,
        ):
            pass

    assert infer_arguments(TestComponent) == {
        "dict_argument": Argument(
            name="dict_argument",
            type=dict,
            optional=False,
            default=None,
        ),
        "list_argument": Argument(
            name="list_argument",
            type=list,
            optional=True,
            default=None,
        ),
    }


def test_unsupported_complex_type():
    class TestComponent(PandasTransformComponent):
        def __init__(
            self,
            *,
            union_argument: t.Union[str, int],
            **kwargs,
        ):
            pass

    with pytest.raises(
        UnsupportedTypeAnnotation,
        match="Fondant only supports simple types",
    ):
        infer_arguments(TestComponent)


def test_unsupported_custom_type():
    class CustomClass:
        pass

    class TestComponent(PandasTransformComponent):
        def __init__(
            self,
            *,
            class_argument: CustomClass,
            **kwargs,
        ):
            pass

    with pytest.raises(
        UnsupportedTypeAnnotation,
        match="Fondant only supports builtin types",
    ):
        infer_arguments(TestComponent)


def test_consumes_produces():
    class TestComponent(PandasTransformComponent):
        def __init__(self, *, argument, consumes, **kwargs):
            pass

    assert infer_arguments(TestComponent) == {
        "argument": Argument(
            name="argument",
            type=str,
            optional=False,
        ),
    }
