"""
This component generates a set of initial prompts that will be used to retrieve images from the LAION-5B dataset.
"""
import itertools
import logging
import typing as t

import dask.dataframe as dd
import pandas as pd

from fondant.component import DaskLoadComponent
from fondant.executor import DaskLoadExecutor

logger = logging.getLogger(__name__)

interior_styles = [
    "art deco",
    "bauhaus",
    "bouclé",
    "maximalist",
    "brutalist",
    "coastal",
    "minimalist",
    "rustic",
    "hollywood regency",
    "midcentury modern",
    "modern organic",
    "contemporary",
    "modern",
    "scandinavian",
    "eclectic",
    "bohemiam",
    "industrial",
    "traditional",
    "transitional",
    "farmhouse",
    "country",
    "asian",
    "mediterranean",
    "rustic",
    "southwestern",
    "coastal",
]

interior_prefix = [
    "comfortable",
    "luxurious",
    "simple",
]

rooms = [
    "Bathroom",
    "Living room",
    "Hotel room",
    "Lobby",
    "Entrance hall",
    "Kitchen",
    "Family room",
    "Master bedroom",
    "Bedroom",
    "Kids bedroom",
    "Laundry room",
    "Guest room",
    "Home office",
    "Library room",
    "Playroom",
    "Home Theater room",
    "Gym room",
    "Basement room",
    "Garage",
    "Walk-in closet",
    "Pantry",
    "Gaming room",
    "Attic",
    "Sunroom",
    "Storage room",
    "Study room",
    "Dining room",
    "Loft",
    "Studio room",
    "Appartement",
]


def make_interior_prompt(room: str, prefix: str, style: str) -> str:
    """Generate a prompt for the interior design model.

    Args:
        room: room name
        prefix: prefix for the room
        style: interior style

    Returns:
        prompt for the interior design model
    """
    return f"{prefix.lower()} {room.lower()}, {style.lower()} interior design"


class GeneratePromptsComponent(DaskLoadComponent):
    def __init__(self, *args, n_rows_to_load: t.Optional[int]) -> None:
        """
        Generate a set of initial prompts that will be used to retrieve images from the LAION-5B
        dataset.

        Args:
            n_rows_to_load: Optional argument that defines the number of rows to load. Useful for
             testing pipeline runs on a small scale
        """
        self.n_rows_to_load = n_rows_to_load

    def load(self) -> dd.DataFrame:
        room_tuples = itertools.product(rooms, interior_prefix, interior_styles)
        prompts = map(lambda x: make_interior_prompt(*x), room_tuples)

        pandas_df = pd.DataFrame(prompts, columns=["prompts_text"])

        if self.n_rows_to_load:
            pandas_df = pandas_df.head(self.n_rows_to_load)

        df = dd.from_pandas(pandas_df, npartitions=1)

        return df
