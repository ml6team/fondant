"""A component that detects and redacts Personal Identifiable Information (PII) in code."""

import json
import logging

import pandas as pd
from fondant.component import PandasTransformComponent
from pii_detection import scan_pii
from pii_redaction import redact_pii

logger = logging.getLogger(__name__)


class RemovePIIComponent(PandasTransformComponent):
    """Component that detects and redacts PII from code."""

    def transform(
            self,
            dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        # detect PII
        result = dataframe.apply(
            lambda example: scan_pii(text=example.code_content),
            axis=1,
            result_type="expand",
            meta={0: object, 1: bool, 2: int},
        )
        result.columns = [("code", "secrets"), ("code", "has_secrets"), ("code", "number_secrets")]

        dataframe = dataframe.merge(result, left_index=True, right_index=True)

        # redact PII
        # we use random replacements by default
        with open("replacements.json") as f:
            replacements = json.load(f)

        dataframe["code"]["content"] = dataframe.apply(
            lambda example: redact_pii(
                text=example.code_content,
                secrets=example.code_secrets,
                has_secrets=example.code_has_secrets,
                replacements=replacements,
            ),
            axis=1,
            meta=(None, "str"),
        )
        return dataframe.drop(
            [("code", "secrets"), ("code", "has_secrets"), ("code", "number_secrets")], axis=1,
        )
