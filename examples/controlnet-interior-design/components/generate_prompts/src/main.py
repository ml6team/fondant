"""
This component generates a set of initial prompts that will be used to retrieve images from the LAION-5B dataset.
"""
import itertools
import logging

import dask.dataframe as dd
import dask.array as da
import pandas as pd

from fondant.component import FondantLoadComponent
from fondant.logger import configure_logging

configure_logging()
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
        room (str): room name
        prefix (str): prefix for the room
        style (str): interior style

    Returns:
        str: prompt for the interior design model
    """
    # TODO: add more variation in the prompt, multiple variants, ...
    return f"{prefix.lower()} {room.lower()}, {style.lower()} interior design"


class LoadPromptsComponent(FondantLoadComponent):
    def load(self):
        """
        Generate a set of initial prompts that will be used to retrieve images from the LAION-5B dataset.

        Returns:
            Dask dataframe
        """
        room_tuples = list(itertools.product(rooms, interior_prefix, interior_styles))
        prompts = list(map(lambda x: make_interior_prompt(*x), room_tuples))

        pandas_df = pd.DataFrame(prompts, columns=["prompts_text"])

        df = dd.from_pandas(pandas_df, npartitions=1)

        index_list = [idx for idx in range(len(df))]
        source_list = ["seed" for _ in range(len(df))]

        df["id"] = da.array(index_list)
        df["source"] = da.array(source_list)

        return df


if __name__ == "__main__":
    component = LoadPromptsComponent.from_file()
    component.run()
