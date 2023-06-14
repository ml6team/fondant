"""A component that detects and redacts Personal Identifiable Information (PII) in code."""

import json
import logging

import dask.dataframe as dd
from pii_detection import scan_pii
from pii_redaction import redact_pii

from fondant.component import TransformComponent
from fondant.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


class RemovePIIComponent(TransformComponent):
    """Component that detects and redacts PII from code."""

    def transform(
        self,
        *,
        dataframe: dd.DataFrame,
    ) -> dd.DataFrame:
        """
        Args:
            dataframe: Dask dataframe.

        Returns:
            Dask dataframe
        """
        # detect PII
        result = dataframe.apply(
            lambda example: scan_pii(text=example.code_content),
            axis=1,
            result_type="expand",
            meta={0: object, 1: bool, 2: int},
        )
        result.columns = ["code_secrets", "code_has_secrets", "code_number_secrets"]

        dataframe = dataframe.merge(result, left_index=True, right_index=True)

        # redact PII
        # we use random replacements by default
        with open("replacements.json", "r") as f:
            replacements = json.load(f)

        dataframe["code_content"] = dataframe.apply(
            lambda example: redact_pii(
                text=example.code_content,
                secrets=example.code_secrets,
                has_secrets=example.code_has_secrets,
                replacements=replacements,
            ),
            axis=1,
            meta=(None, "str"),
        )
        dataframe = dataframe.drop(
            ["code_secrets", "code_has_secrets", "code_number_secrets"], axis=1
        )

        return dataframe


if __name__ == "__main__":
    component = RemovePIIComponent.from_args()
    component.run()
