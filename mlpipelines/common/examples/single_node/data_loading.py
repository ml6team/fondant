"""Example script showcasing data loading with Express. Make sure to install express with
`setup.py develop` command
run with command
`python3 data_loading.py --extra-args '{"project_id": "storied-landing-366912"}' \
 --output-manifest <local_path> \
  --metadata-args '{"run_id":"test","component_name":"test_component", \
  "artifact_bucket":"storied-landing-366912-kfp-output"}'

"""
from typing import Optional, Dict, Union

import pandas as pd

from express_components.pandas_components import PandasLoaderComponent, PandasDatasetDraft
from express_components.helpers.logger import configure_logging


# pylint: disable=too-few-public-methods
class SeedDatasetLoader(PandasLoaderComponent):
    """Class that inherits from Pandas data loading """

    @classmethod
    def load(cls, extra_args: Optional[Dict[str, Union[str, int, float, bool]]] = None) -> PandasDatasetDraft:
        """
        An example function showcasing the data loader component using Express functionalities
        Args:
            extra_args (Optional[Dict[str, Union[str, int, float, bool]]): optional args to pass to the function (e.g.
             seed data source)
        Returns:
            PandasDatasetDraft: a dataset draft that creates a plan for an output datasets/manifest
        """
        configure_logging()

        # 1) Create an example index
        index_list = ['index_1', 'index_2', 'index_3', 'index_4']

        # 2) Create example datasources
        # 2.1.1) metadata
        metadata = {'index': index_list,
                    'uri': ['uri_1', 'uri_2', 'uri_3', 'uri4'],
                    'size': [300, 400, 500, 600],
                    'format': ['jpeg', 'jpeg', 'jpeg', 'jpeg']}

        df_metadata = pd.DataFrame(metadata).set_index('index')

        # 2.1.2) Caption]

        captions = {'index': index_list,
                    'captions': ['dog', 'cat', 'bear', 'duck']}

        df_captions = pd.DataFrame(captions).set_index('index')

        # 2.2) Create data_source dictionary
        data_sources = {"metadata": df_metadata,
                        "captions": df_captions}

        # Create dataset draft from index and additional data sources
        dataset_draft = PandasDatasetDraft(index=df_metadata.index, data_sources=data_sources)

        return dataset_draft


if __name__ == '__main__':
    SeedDatasetLoader.run()
