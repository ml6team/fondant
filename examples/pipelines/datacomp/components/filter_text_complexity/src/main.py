"""This component filters text based on complexity of the dependency parse tree.

As proposed in [Radenovic et al., 2023](https://arxiv.org/abs/2301.02280).
"""
import logging

import pandas as pd
import spacy

from fondant.component import PandasTransformComponent
from fondant.executor import PandasTransformExecutor

logger = logging.getLogger(__name__)


def get_text_complexity(doc: spacy.tokens.doc.Doc):
    complexity = 0
    for token in doc:
        num_children = len([child for child in token.children])
        if num_children > complexity:
            complexity = num_children

    return complexity


class FilterTextComplexity(PandasTransformComponent):
    """Component that filters text based on complexity of the dependency parse tree."""

    def __init__(
        self,
        *args,
        spacy_pipeline,
        batch_size: int,
        min_complexity: int,
    ) -> None:
        self.nlp = spacy.load(
            spacy_pipeline, exclude=["tagger", "ner", "lemmatizer", "textcat"]
        )
        self.batch_size = batch_size
        self.min_complexity = min_complexity

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        texts = dataframe["text"]["data"]

        logger.info("Creating SpaCy docs...")
        docs = list(self.nlp.pipe(texts, batch_size=self.batch_size))
        docs = pd.Series(docs)

        logger.info("Calculating text complexity...")
        caption_complexity = docs.apply(lambda doc: get_text_complexity(doc))

        mask = caption_complexity >= self.min_complexity
        mask = mask.to_numpy()

        return dataframe[mask]
