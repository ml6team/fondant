import typing as t

import pandas as pd
from datasets import Dataset
from fondant.component import PandasTransformComponent
from ragas import evaluate
from ragas.llms import LangchainLLM


class RetrieverEval(PandasTransformComponent):
    def __init__(
        self,
        *,
        llm_module_name: str,
        llm_class_name: str,
        llm_kwargs: dict,
        produces: t.Dict[str, t.Any],
    ) -> None:
        """
        Args:
            llm_module_name: Module from which the LLM is imported. Defaults to
             langchain.chat_models
            llm_class_name: Name of the selected llm. Defaults to ChatOpenAI
            llm_kwargs: Arguments of the selected llm
            produces: RAGAS metrics to compute.
        """
        self.llm = self.extract_llm(
            llm_module_name=llm_module_name,
            llm_class_name=llm_class_name,
            llm_kwargs=llm_kwargs,
        )
        self.gpt_wrapper = LangchainLLM(llm=self.llm)
        self.metric_functions = self.extract_metric_functions(
            metrics=list(produces.keys()),
        )
        self.set_llm(self.metric_functions)

    # import the metric functions selected
    @staticmethod
    def import_from(module_name: str, element_name: str):
        module = __import__(module_name, fromlist=[element_name])
        return getattr(module, element_name)

    def extract_llm(self, llm_module_name: str, llm_class_name: str, llm_kwargs: dict):
        module = self.import_from(
            module_name=llm_module_name,
            element_name=llm_class_name,
        )
        return module(**llm_kwargs)

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
            columns={"retrieved_chunks": "contexts"},
        )
        return Dataset.from_pandas(dataframe)

    def ragas_eval(self, dataset):
        return evaluate(dataset=dataset, metrics=self.metric_functions)

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        hf_dataset = self.create_hf_ds(
            dataframe=dataframe[["question", "retrieved_chunks"]],
        )
        if "id" in hf_dataset.column_names:
            hf_dataset = hf_dataset.remove_columns("id")

        result = self.ragas_eval(dataset=hf_dataset)
        results_df = result.to_pandas()
        results_df = results_df.set_index(dataframe.index)

        return results_df
