"""This module defines exceptions thrown by Fondant."""

from jsonschema.exceptions import ValidationError


class FondantException(Exception):
    """All custom Fondant exception should subclass this one."""


class InvalidManifest(ValidationError, FondantException):
    """Thrown when a manifest cannot be validated against the schema."""


class InvalidComponentSpec(ValidationError, FondantException):
    """Thrown when a component spec cannot be validated against the schema."""


class InvalidDatasetDefinition(ValidationError, FondantException):
    """Thrown when a dataset definition is invalid."""


class InvalidTypeSchema(ValidationError, FondantException):
    """Thrown when a Type schema definition is invalid."""


class UnsupportedTypeAnnotation(FondantException):
    """Thrown when an unsupported type annotation is encountered during type inference."""


class InvalidLightweightComponent(FondantException):
    """Thrown when a component is not a valid lightweight component."""
