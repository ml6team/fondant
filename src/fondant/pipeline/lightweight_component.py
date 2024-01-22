import inspect
import itertools
import sys
import textwrap
import typing as t
from dataclasses import dataclass
from functools import wraps
from importlib.metadata import version

from fondant.component import Component

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

    @staticmethod
    def resolve_fndnt_base_image(use_ecr_registry=False):
        """Resolve the correct fndnt base image using python version and fondant version."""
        # Set python version to latest supported version
        python_version = sys.version_info
        if MIN_PYTHON_VERSION <= python_version < MAX_PYTHON_VERSION:
            python_version = f"{python_version.major}.{python_version.minor}"
        else:
            python_version = f"{MAX_PYTHON_VERSION[0]}.{MAX_PYTHON_VERSION[1]}"

        fondant_version = version("fondant")
        basename = (
            "fndnt/fondant-base"
            if not use_ecr_registry
            else "public.ecr.aws/fndnt/fondant-base"
        )
        return f"{basename}:{fondant_version}-python{python_version}"


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
