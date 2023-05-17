"""
This is a small script to split the raw data into different subsets to be used while testing.

The data is the 151 first pokemon and the following fields are available:

'id', 'Name', 'Type 1', 'Type 2', 'Total', 'HP', 'Attack', 'Defense',
'Sp. Atk', 'Sp. Def', 'Speed', 'source', 'Legendary'


"""
from pathlib import Path

import dask.dataframe as dd

data_path = Path(__file__).parent
output_path = Path(__file__).parent.parent / "subsets_input/"


def split_into_subsets():
    # read in complete dataset
    master_df = dd.read_parquet(path=data_path / "testset.parquet")
    master_df = master_df.set_index("id", drop=False, sorted=True)
    master_df = master_df.repartition(divisions=[0, 50, 100, 151], force=True)
    master_df = master_df.astype({"source": "string", "id": "string"})
    master_df["uid"] = master_df["source"] + "_" + master_df["id"].astype("str")

    # create index subset
    index_df = master_df[["uid", "id", "source"]]
    index_df = index_df.set_index("uid")
    index_df.to_parquet(output_path / "index")

    # create properties subset
    properties_df = master_df[["uid", "Name", "HP"]]
    properties_df = properties_df.set_index("uid")
    properties_df.to_parquet(output_path / "properties")

    # create types subset
    types_df = master_df[["uid", "Type 1", "Type 2"]]
    types_df = types_df.set_index("uid")
    types_df.to_parquet(output_path / "types")


if __name__ == "__main__":
    split_into_subsets()
