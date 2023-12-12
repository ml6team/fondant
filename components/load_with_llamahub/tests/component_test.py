from pathlib import Path

import yaml
from fondant.core.component_spec import ComponentSpec

from src.main import LlamaHubReader


def test_arxiv_reader():
    """Test the component with the ArxivReader.

    This test requires a stable internet connection, both to download the loader, and to download
    the papers from Arxiv.
    """
    with open(Path(__file__).with_name("fondant_component.yaml")) as f:
        spec = yaml.safe_load(f)
    spec = ComponentSpec(spec)

    component = LlamaHubReader(
        spec=spec,
        loader_class="ArxivReader",
        loader_kwargs={},
        load_kwargs={
            "search_query": "jeff dean",
            "max_results": 5,
        },
        additional_requirements=["pypdf"],
        n_rows_to_load=None,
        index_column=None,
    )

    output_dataframe = component.load().compute()

    assert len(output_dataframe) > 0
    assert output_dataframe.columns.tolist() == ["text", "URL", "Title of this paper"]
