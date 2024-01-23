import inspect
import itertools
import textwrap
import typing as t
from dataclasses import asdict, dataclass
from functools import wraps

import pyarrow as pa

from fondant.component import BaseComponent, Component
from fondant.core.schema import Field


@dataclass
class Image:
    base_image: str = "fondant:latest"
    extra_requires: t.Optional[t.List[str]] = None
    script: t.Optional[str] = None

    def __post_init__(self):
        if self.base_image is None:
            # TODO: link to Fondant version
            self.base_image = "fondant:latest"

    def to_dict(self):
        return asdict(self)


class PythonComponent(BaseComponent):
    @classmethod
    def image(cls) -> Image:
        raise NotImplementedError

    @classmethod
    def consumes(cls) -> t.Optional[list]:
        pass

    @classmethod
    def get_consumes_spec(
        cls,
        dataset_fields: t.Mapping[str, Field],
        apply_consumes: t.Optional[t.Dict[str, t.Union[str, pa.DataType]]],
    ):
        pass


def lightweight_component(
    *args,
    extra_requires: t.Optional[t.List[str]] = None,
    base_image: t.Optional[str] = None,
    consumes: t.Optional[list] = None,
):
    """Decorator to enable a python component."""

    def wrapper(cls):
        script = build_python_script(cls)
        image = Image(
            base_image=base_image,
            extra_requires=extra_requires,
            script=script,
        )

        def get_base_cls(cls):
            """
            Returns the BaseComponent. If the implementation inherits from several classes,
            the Fondant base class is selected. If more than one Fondant base component
            is implemented, an exception is raised.
            """
            base_component_module = inspect.getmodule(Component).__name__
            base_component_cls_list = [
                base
                for base in cls.__bases__
                if base.__module__ == base_component_module
            ]
            if len(base_component_cls_list) > 1:
                msg = (
                    f"Multiple base classes detected. Only one component should be inherited or"
                    f" implemented."
                    f"Found classes: {', '.join([cls.__name__ for cls in base_component_cls_list])}"
                )
                raise ValueError(
                    msg,
                )
            return base_component_cls_list[0]

        def validate_signatures(base_component_cls, cls_implementation):
            """
            Compare the signature of overridden methods in a class with their counterparts
            in the BaseComponent classes.
            """
            for function_name in dir(cls_implementation):
                if not function_name.startswith("__") and function_name in dir(
                    base_component_cls,
                ):
                    type_cls_implementation = inspect.signature(
                        getattr(cls_implementation, function_name, None),
                    )
                    type_base_cls = inspect.signature(
                        getattr(base_component_cls, function_name, None),
                    )
                    if type_cls_implementation != type_base_cls:
                        msg = (
                            f"Invalid function definition of function {function_name}. "
                            f"The expected function signature is {type_base_cls}"
                        )
                        raise ValueError(
                            msg,
                        )

        def validate_abstract_methods_are_implemented(cls):
            """
            Function to validate that a class has overridden every required function marked as
            abstract.
            """
            abstract_methods = [
                name
                for name, value in inspect.getmembers(cls)
                if getattr(value, "__isabstractmethod__", False)
            ]
            if len(abstract_methods) >= 1:
                msg = (
                    f"Every required function must be overridden in the PythonComponent. "
                    f"Missing implementations for the following functions: {abstract_methods}"
                )
                raise ValueError(
                    msg,
                )

        def modify_consumes_spec(apply_consumes, consumes_spec):
            """Modify fields based on the consumes argument in the 'apply' method."""
            if apply_consumes:
                for k, v in apply_consumes.items():
                    if isinstance(v, str):
                        consumes_spec[k] = consumes_spec.pop(v)
                    elif isinstance(v, pa.DataType):
                        pass
                    else:
                        msg = (
                            f"Invalid data type for field `{k}` in the `apply_consumes` "
                            f"argument. Only string and pa.DataType are allowed."
                        )
                        raise ValueError(
                            msg,
                        )
            return consumes_spec

        def filter_consumes_spec(python_component_consumes, consumes_spec):
            """Filter for values that are not in the user defined consumes list."""
            if python_component_consumes:
                for field_to_consume in python_component_consumes:
                    if field_to_consume not in consumes_spec.keys():
                        msg = f"Field `{field_to_consume}` is not available in the dataset."
                        raise ValueError(
                            msg,
                        )

                    consumes_spec = {
                        k: v
                        for k, v in consumes_spec.items()
                        if k in python_component_consumes
                    }
            return consumes_spec

        validate_abstract_methods_are_implemented(cls)
        base_component_cls = get_base_cls(cls)
        validate_signatures(base_component_cls, cls)

        # updated=() is needed to prevent an attempt to update the class's __dict__
        @wraps(cls, updated=())
        class PythonComponentOp(cls, PythonComponent):
            @classmethod
            def image(cls) -> Image:
                return image

            @classmethod
            def consumes(cls) -> t.Optional[list]:
                return consumes

            @classmethod
            def get_consumes_spec(
                cls,
                dataset_fields: t.Mapping[str, Field],
                apply_consumes: t.Optional[t.Dict[str, t.Union[str, pa.DataType]]],
            ):
                python_component_consumes = cls.consumes()

                # Get consumes spec from the dataset
                consumes_spec = {k: v.type.to_dict() for k, v in dataset_fields.items()}

                # Modify naming based on the consumes argument in the 'apply' method
                consumes_spec = modify_consumes_spec(apply_consumes, consumes_spec)

                # Filter for values that are not in the user defined consumes list
                consumes_spec = filter_consumes_spec(
                    python_component_consumes,
                    consumes_spec,
                )

                return consumes_spec

        return PythonComponentOp

    # Call wrapper with function (`args[0]`) when no additional arguments were passed
    if args:
        return wrapper(args[0])

    return wrapper


def build_python_script(component_cls: t.Type[Component]) -> str:
    """Build a self-contained python script for the provided component class, which will act as
    the `src/main.py` script to execute the component.
    """
    imports_source = textwrap.dedent(
        """\
        from typing import *
        import typing as t

        import dask.dataframe as dd
        import fondant
        import pandas as pd
        from fondant.component import *
        from fondant.core import *
    """,
    )

    component_source = inspect.getsource(component_cls)
    component_source = textwrap.dedent(component_source)
    component_source_lines = component_source.split("\n")

    # Removing possible decorators (can be multiline) until the class
    # definition is found
    component_source_lines = list(
        itertools.dropwhile(
            lambda x: not x.startswith("class"),
            component_source_lines,
        ),
    )

    if not component_source_lines:
        msg = (
            f'Failed to dedent and clean up the source of function "{component_cls.__name__}". '
            f"Its probably not properly indented."
        )
        raise ValueError(
            msg,
        )

    component_source = "\n".join(component_source_lines)

    return "\n\n".join([imports_source, component_source])
