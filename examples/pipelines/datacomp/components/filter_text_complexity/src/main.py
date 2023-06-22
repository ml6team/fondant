"""This component filters text based on:

- complexity of the dependency parse tree
- number of actions.

As proposed in [Radenovic et al., 2023](https://arxiv.org/abs/2301.02280).
"""
import logging

import pandas as pd
import spacy
from spacy.symbols import nsubj, VERB

from fondant.component import PandasTransformComponent
from fondant.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def get_text_complexity(doc: spacy.tokens.doc.Doc):
    complexity = 0
    for token in doc:
        num_children = len([child for child in token.children])
        if num_children > complexity:
            complexity = num_children

    return complexity


def get_num_actions(doc: spacy.tokens.doc.Doc):
    verbs = set()
    for possible_subject in doc:
        if possible_subject.dep == nsubj and possible_subject.head.pos == VERB:
            verbs.add(possible_subject.head)

    return len(verbs)


class FilterTextComplexity(PandasTransformComponent):
    """Component that filters text based on:

    - complexity of the dependency parse tree
    - number of actions"""

    def setup(
        self,
        *,
        spacy_pipeline,
        batch_size: int,
        min_complexity: int,
        min_num_actions: int
    ) -> None:
        self.nlp = spacy.load(spacy_pipeline, exclude=["ner"])
        self.batch_size = batch_size
        self.min_complexity = min_complexity
        self.min_num_actions = min_num_actions

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        texts = dataframe["text"]["data"]

        docs = list(self.nlp.pipe(texts, batch_size=self.batch_size))
        docs = pd.Series(docs)

        caption_complexity = docs.apply(lambda doc: get_text_complexity(doc))
        num_actions = docs.apply(lambda doc: get_num_actions(doc))

        mask = (caption_complexity >= self.min_complexity) & (
            num_actions >= self.min_num_actions
        )
        mask = mask.to_numpy()

        dataframe = dataframe[mask]

        print("Shape of the dataframe after filtering:", dataframe.shape)
        print("Columns of final dataframe:", dataframe.columns)

        return dataframe


if __name__ == "__main__":
    component = FilterTextComplexity.from_args()
    component.run()
