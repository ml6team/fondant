import typing as t
from dataclasses import dataclass
from functools import wraps


@dataclass
class Image:
    base_image: str = "fondant:latest"
    extra_requires: t.Optional[t.List[str]] = None
    script: t.Optional[str] = None


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
        kwargs = {}
        if base_image:
            kwargs["base_image"] = base_image
        if extra_requires:
            kwargs["extra_requires"] = extra_requires
        image = Image(**kwargs)

        # updated=() is needed to prevent an attempt to update the class's __dict__
        @wraps(cls, updated=())
        class AppliedPythonComponent(cls, PythonComponent):
            @classmethod
            def image(cls) -> Image:
                return image

        return AppliedPythonComponent

    return wrapper
