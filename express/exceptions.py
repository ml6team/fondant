"""This module defines exceptions thrown by Express."""

from jsonschema.exceptions import ValidationError


class ExpressException(Exception):
    """All custom Express exception should subclass this one."""


class InvalidManifest(ValidationError, ExpressException):
    """Thrown when a manifest cannot be validated against the schema."""
