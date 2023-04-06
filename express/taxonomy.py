"This module defines the taxonomy of the various modalities."


class Image:
    """
    A single image
    """

    def __init__(self):
        pass

    def required_columns(self):
        return ["width", "height"]


class Text:
    """
    A single text
    """

    def __init__(self):
        pass

    def required_columns(self):
        return ["len"]

    def __repr__(self) -> str:
        return "Text"


class Vector:
    """
    A single vector
    """

    def __init__(self):
        pass

    def required_columns(self):
        return ["size"]
