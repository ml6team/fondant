import inspect
import typing as t

from fondant.component import Component
from fondant.core.component_spec import Argument
from fondant.core.exceptions import UnsupportedTypeAnnotation

BUILTIN_TYPES = [str, int, float, bool, dict, list]


def annotation_to_type(annotation: t.Any) -> t.Type:
    """Extract the simple built-in type from an Annotation.

    Examples:
        dict[str, int] -> dict
        t.Optional[str] -> str

    Args:
        annotation: Annotation of an argument as returned by inspect.signature

    Raises:
        UnsupportedTypeAnnotation: If the annotation is not simple or not based on a built-in type

    """
    # If no type annotation is present, default to str
    if annotation == inspect.Parameter.empty:
        return str

    # Unpack the annotation until we get a simple type.
    # This removes complex structures such as Optional
    while t.get_origin(annotation) not in [*BUILTIN_TYPES, None]:
        # Filter out NoneType values (Optional[x] is represented as Union[x, NoneType]
        annotation_args = [
            arg for arg in t.get_args(annotation) if arg is not type(None)
        ]

        # Multiple arguments remaining (eg. Union[str, int])
        # Raise error since we cannot infer type unambiguously
        if len(annotation_args) > 1:
            msg = (
                f"Fondant only supports simple types for component arguments."
                f"Expected one of {BUILTIN_TYPES}, received {annotation} instead."
            )
            raise UnsupportedTypeAnnotation(msg)

        annotation = annotation_args[0]

    # Remove any subscription (eg. dict[str, int] -> dict)
    annotation = t.get_origin(annotation) or annotation

    # Check for classes not supported as argument
    if annotation not in BUILTIN_TYPES:
        msg = (
            f"Fondant only supports builtin types for component arguments."
            f"Expected one of {BUILTIN_TYPES}, received {annotation} instead."
        )
        raise UnsupportedTypeAnnotation(msg)

    return annotation


def is_optional(parameter: inspect.Parameter) -> bool:
    """Check if an inspect.Parameter is optional. We check this based on the presence of a
    default value instead of based on the type, since this is more trustworthy.
    """
    return parameter.default != inspect.Parameter.empty


def get_default(parameter: inspect.Parameter) -> t.Any:
    """Get the default value from an inspect.Parameter."""
    if parameter.default == inspect.Parameter.empty:
        return None
    return parameter.default


def parameter_to_argument(parameter: inspect.Parameter) -> Argument:
    """Translate an inspect.Parameter into a Fondant Argument."""
    return Argument(
        name=parameter.name,
        type=annotation_to_type(parameter.annotation),
        optional=is_optional(parameter),
        default=get_default(parameter),
    )


def infer_arguments(component: t.Type[Component]) -> t.Dict[str, Argument]:
    """Infer the user arguments from a Python Component class.
    Default arguments are skipped.

    Args:
        component: Component class to inspect.
    """
    signature = inspect.signature(component)

    arguments = {}
    for name, parameter in signature.parameters.items():
        # Skip non-user arguments
        if name in ["self", "consumes", "produces", "kwargs"]:
            continue

        arguments[name] = parameter_to_argument(parameter)

    return arguments
