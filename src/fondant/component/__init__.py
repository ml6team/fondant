try:
    pass
except ImportError:
    msg = (
        "You need to install fondant using the `component` extra to develop or run a component."
        "You can install it with `pip install fondant[component]`"
    )
    raise SystemExit(
        msg,
    )

from .component import (  # noqa
    BaseComponent,
    Component,
    DaskLoadComponent,
    DaskTransformComponent,
    DaskWriteComponent,
    PandasTransformComponent,
)
