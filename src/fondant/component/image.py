import typing as t
from dataclasses import dataclass


@dataclass
class Image:
    base_image: t.Optional[str] = "python:3.8-slim"
    extra_requires: t.Optional[t.List[str]] = None
    script: t.Optional[str] = None
