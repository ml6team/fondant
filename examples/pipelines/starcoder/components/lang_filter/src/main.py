"""
Custom component to implement
"""
import dask.dataframe as dd

from fondant.component import TransformComponent


class CustomComponent(TransformComponent):
    """
    Custom component
    """

    def transform(self, dataframe: dd.DataFrame, *, lang: str) -> dd.DataFrame:
        """
        Implement this function to do the actual filtering

        Args:
            dataframe: Dask dataframe
            Arguments: ...

        Returns:
            Filtered dask dataframe
        """
        f_df = dataframe[dataframe["code_lang"] == lang]
        return f_df


if __name__ == "__main__":
    component = CustomComponent.from_args()
    component.run()
