import inspect
import itertools
import logging
import sys
import textwrap
import typing as t
from dataclasses import asdict, dataclass
from functools import wraps
from importlib.metadata import version

from fondant.component import BaseComponent, Component

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MIN_PYTHON_VERSION = (3, 8)
MAX_PYTHON_VERSION = (3, 11)


@dataclass
class Image:
    base_image: str
    extra_requires: t.Optional[t.List[str]] = None
    script: t.Optional[str] = None

    def __post_init__(self):
        if self.base_image is None:
            self.base_image = self.resolve_fndnt_base_image()

        # log info when custom image without Fondant is defined
        elif self.extra_requires and not any(
            dependency.startswith("fondant") for dependency in self.extra_requires
        ):
            msg = (
                "You are not using a Fondant default base image, and Fondant is not part of"
                "your extra requirements. Please make sure that you have installed fondant "
                "inside your container. Alternatively, you can should add Fondant to "
                "the extra requirements. \n"
                "E.g. \n"
                '@lightweight_component(..., extra_requires=["fondant"])'
            )

            logger.warning(msg)

    @staticmethod
    def resolve_fndnt_base_image():
        """Resolve the correct fndnt base image using python version and fondant version."""
        # Set python version to latest supported version
        python_version = sys.version_info
        if MIN_PYTHON_VERSION <= python_version < MAX_PYTHON_VERSION:
            python_version = f"{python_version.major}.{python_version.minor}"
        else:
            python_version = f"{MAX_PYTHON_VERSION[0]}.{MAX_PYTHON_VERSION[1]}"

        fondant_version = version("fondant")
        if fondant_version == "0.1.dev0":
            fondant_version = "dev"
        basename = "fndnt/fondant"
        return f"{basename}:{fondant_version}-py{python_version}"

    def to_dict(self):
        return asdict(self)


class PythonComponent(BaseComponent):
    @classmethod
    def image(cls) -> Image:
        raise NotImplementedError


def lightweight_component(
    *args,
    extra_requires: t.Optional[t.List[str]] = None,
    base_image: t.Optional[str] = None,
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

        validate_abstract_methods_are_implemented(cls)
        base_component_cls = get_base_cls(cls)
        validate_signatures(base_component_cls, cls)

        # updated=() is needed to prevent an attempt to update the class's __dict__
        @wraps(cls, updated=())
        class PythonComponentOp(cls, PythonComponent):
            @classmethod
            def image(cls) -> Image:
                return image

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
