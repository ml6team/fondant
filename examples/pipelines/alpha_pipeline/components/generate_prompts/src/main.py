"""
This component generates prompts that will be used to retrieve images
 from the LAION-5B dataset.
"""
import os
import itertools
import pandas as pd
import dask as dd


def load_prompt_seeds(path_to_data):
    """
    Load seed lists for generating prompts to retrieve images from the
    LAION-5B dataset.

    Args:
        path_to_data (str): Path to the directory containing text files with
                            prompt seeds. The path should lead to three files:
                            room_types.txt, interior_styles.txt, and
                            interior_prefix.txt

    Returns:
        tuple: A tuple containing three lists of lowercase strings:
            - rooms: A list of types of rooms.
            - interior_styles: A list interior styles.
            - interior_prefix: A list of prefixes for interior prompts.
    """
    with open(os.path.join(path_to_data, "room_types.txt")) as f:
        rooms = [line.strip().lower() for line in f.readlines()]

    with open(os.path.join(path_to_data, "interior_styles.txt")) as f:
        interior_styles = [line.strip().lower() for line in f.readlines()]

    with open(os.path.join(path_to_data, "interior_prefix.txt")) as f:
        interior_prefix = [line.strip().lower() for line in f.readlines()]

    return rooms, interior_styles, interior_prefix


def generate_prompts(path_to_data):
    """
    Generate prompts for retrieving images from the LAION-5B dataset using the
    given seed lists"

    Args:
        path_to_data (str): Path to the directory containing text files with
        prompt seeds.

    Returns:
        dask.dataframe: A dask dataframe containing generated prompts.
    """
    rooms, interior_styles, interior_prefix = load_prompt_seeds(path_to_data)

    room_tuples = list(itertools.product(interior_prefix,
                                         interior_styles,
                                         rooms))

    prompts_list = [" ".join(i) for i in room_tuples]

    prompts_pdf = pd.DataFrame(prompts_list, columns=['prompts_data'])

    prompts_ddf = dd.from_pandas(prompts_pdf)

    return prompts_ddf
