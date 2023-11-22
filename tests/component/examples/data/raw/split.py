"""
This is a small script to split the raw data into different subsets to be used while testing.

The data is the 151 first pokemon and the following fields are available:

'id', 'Name', 'Type 1', 'Type 2', 'Total', 'HP', 'Attack', 'Defense',
'Sp. Atk', 'Sp. Def', 'Speed', 'source', 'Legendary'


"""
from pathlib import Path

import dask.dataframe as dd

data_path = Path(__file__).parent
output_path = Path(__file__).parent.parent


def split_into_subsets():
    # read in complete dataset
    master_df = dd.read_parquet(path=data_path / "testset.parquet")
    master_df = master_df.set_index("id", sorted=True)
    master_df = master_df.repartition(divisions=[0, 50, 100, 151], force=True)

    # create properties subset
    properties_df = master_df[["Name", "HP"]]
    properties_df.to_parquet(output_path / "component_1")

    # create types subset
    types_df = master_df[["Type 1", "Type 2"]]
    types_df.to_parquet(output_path / "component_2")


if __name__ == "__main__":
    split_into_subsets()
