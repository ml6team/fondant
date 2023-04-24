from typing import Optional, Dict, Union
import pandas as pd

from fondant.components.pandas_components import PandasLoaderComponent, PandasDatasetDraft
from fondant.logger import configure_logging


# pylint: disable=too-few-public-methods
class SeedDatasetLoader(PandasLoaderComponent):
    """Class that inherits from Pandas data loading """

    @classmethod
    def load(cls, extra_args: Optional[
        Dict[str, Union[str, int, float, bool]]] = None) -> PandasDatasetDraft:
        """
        An example function showcasing the data loader component using Fondant functionalities
        Args:
            extra_args (Optional[Dict[str, Union[str, int, float, bool]]): optional args to pass to
             the function (e.g. seed data source)
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
        # 2.1.2) Caption
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
