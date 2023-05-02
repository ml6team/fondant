"""
This is a small script to split the raw data into different subsets to be used while testing.

The data is the 151 first pokemon and the following field are available:

'id', 'Name', 'Type 1', 'Type 2', 'Total', 'HP', 'Attack', 'Defense',
'Sp. Atk', 'Sp. Def', 'Speed', 'source', 'Legendary'


"""
from pathlib import Path
import dask.dataframe as dd

data_path = Path(__file__).parent
output_path = Path(__file__).parent.parent / "subsets/"


def split_into_subsets():
    # read in complete dataset
    master_df = dd.read_parquet(path=data_path / "testset.parquet")

    # create index subset
    index_df = master_df[["id", "source"]]
    index_df.set_index("id")
    index_df.to_parquet(output_path / "index")

    # create properties subset
    properies_df = master_df[["id", "source", "Name", "HP"]]
    properies_df.set_index("id")
    properies_df.to_parquet(output_path / "properties")

    # create types subset
    types_df = master_df[["id", "source", "Type 1", "Type 2"]]
    types_df.set_index("id")
    types_df.to_parquet(output_path / "types")


if __name__ == "__main__":
    split_into_subsets()
