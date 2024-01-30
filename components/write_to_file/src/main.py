import dask.dataframe as dd
from fondant.component import DaskWriteComponent


class WriteToFile(DaskWriteComponent):
    def __init__(self, *, path: str, format: str):
        """Initialize the write to file component."""
        self.path = path
        self.format = format

    def write(self, dataframe: dd.DataFrame) -> None:
        """
        Writes the data from the given Dask DataFrame to a file either locally or
        to a remote storage bucket.

        Args:
            dataframe (dd.DataFrame): The Dask DataFrame containing the data to be written.
        """
        if self.format.lower() == "csv":
            self.path = self.path + "/export-*.csv"
            dataframe.to_csv(self.path)
        elif self.format.lower() == "parquet":
            dataframe.to_parquet(self.path)
        else:
            msg = (
                f"Not supported file format {self.format}. Write to file only "
                f"supports csv and parquet."
            )
            raise ValueError(msg)
