import inspect
import itertools
import textwrap
import typing as t
from dataclasses import dataclass
from functools import wraps

from fondant.component import Component


@dataclass
class Image:
    base_image: str = "fondant:latest"
    extra_requires: t.Optional[t.List[str]] = None
    script: t.Optional[str] = None

    def __post_init__(self):
        if self.base_image is None:
            # TODO: link to Fondant version
            self.base_image = "fondant:latest"


class PythonComponent:
    @classmethod
    def image(cls) -> Image:
        raise NotImplementedError


def lightweight_component(
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

        base_component_cls = cls.__bases__[0]

        for function_name in dir(cls):
            if not function_name.startswith("__") and function_name in dir(
                base_component_cls,
            ):
                type_cls_implementation = inspect.signature(
                    getattr(cls, function_name, None),
                ).return_annotation
                type_base_cls = inspect.signature(
                    getattr(base_component_cls, function_name, None),
                ).return_annotation
                if type_cls_implementation != type_base_cls:
                    msg = (
                        f"Invalid function definition of function {function_name}. The return type "
                        f"has to be {type_base_cls}"
                    )
                    raise ValueError(
                        msg,
                    )

        # updated=() is needed to prevent an attempt to update the class's __dict__
        @wraps(cls, updated=())
        class AppliedPythonComponent(cls, PythonComponent):
            @classmethod
            def image(cls) -> Image:
                return image

        return AppliedPythonComponent

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
