import pandas as pd
from datasets import Dataset
from fondant.component import PandasTransformComponent
from ragas import evaluate
from ragas.llms import LangchainLLM


class RetrieverEval(PandasTransformComponent):
    def __init__(
        self,
        *,
        module: str,
        llm_name: str,
        llm_kwargs: dict,
        metrics: list,
        **kwargs,
    ) -> None:
        """
        Args:
            module: Module from which the LLM is imported. Defaults to langchain.llms
            llm_name: Name of the selected llm
            llm_kwargs: Arguments of the selected llm
            metrics: RAGAS metrics to compute.
            kwargs: Unhandled keyword arguments passed in by Fondant.
        """
        self.llm = self.extract_llm(
            module=module,
            model_name=llm_name,
            model_kwargs=llm_kwargs,
        )
        self.gpt_wrapper = LangchainLLM(llm=self.llm)
        self.metric_functions = self.extract_metric_functions(metrics=metrics)
        self.set_llm(self.metric_functions)

    # import the metric functions selected
    @staticmethod
    def import_from(module, name):
        module = __import__(module, fromlist=[name])
        return getattr(module, name)

    def extract_llm(self, module, model_name, model_kwargs):
        module = self.import_from(module, model_name)
        return module(**model_kwargs)

    def extract_metric_functions(self, metrics: list):
        functions = []
        for metric in metrics:
            functions.append(self.import_from("ragas.metrics", metric))
        return functions

    def set_llm(self, metric_functions: list):
        for metric_function in metric_functions:
            metric_function.llm = self.gpt_wrapper

    # evaluate the retriever
    @staticmethod
    def create_hf_ds(dataframe: pd.DataFrame):
        dataframe = dataframe.rename(
            columns={"text": "question", "retrieved_chunks": "contexts"},
        )
        return Dataset.from_pandas(dataframe)

    def ragas_eval(self, dataset):
        return evaluate(dataset=dataset, metrics=self.metric_functions)

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        hf_dataset = self.create_hf_ds(
            dataframe=dataframe[["text", "retrieved_chunks"]],
        )
        if "id" in hf_dataset.column_names:
            hf_dataset = hf_dataset.remove_columns("id")

        result = self.ragas_eval(dataset=hf_dataset)
        results_df = result.to_pandas()
        results_df = results_df.set_index(dataframe.index)

        return results_df
