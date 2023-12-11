import dask.dataframe as dd
from fondant.component import DaskTransformComponent


class AggregateResults(DaskTransformComponent):
    def __init__(self, **kwargs) -> None:
        return None

    def transform(self, dataframe: dd.DataFrame) -> dd.DataFrame:
        metrics = list(dataframe.select_dtypes(["float", "int"]).columns)
        agg = dataframe[metrics].mean()
        agg_df = agg.to_frame(name="score")
        agg_df["metric"] = agg.index
        agg_results_df = agg_df[["metric", "score"]]
        agg_results_df = agg_results_df.reset_index(drop=True)

        return agg_results_df
