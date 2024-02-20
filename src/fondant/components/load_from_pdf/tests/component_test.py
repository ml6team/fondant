from pathlib import Path

import yaml
from fondant.core.component_spec import ComponentSpec, OperationSpec

from src.main import PDFReader


def test_pdf_reader():
    """Test the component with the ArxivReader.

    This test requires a stable internet connection, both to download the loader, and to download
    the papers from Arxiv.
    """
    with open(Path(__file__).with_name("fondant_component.yaml")) as f:
        spec = ComponentSpec(yaml.safe_load(f))

    spec = OperationSpec(spec)

    pdf_path = ["tests/test_file/dummy.pdf", "tests/test_folder"]

    for path in pdf_path:
        component = PDFReader(
            produces=dict(spec.operations_produces),
            pdf_path=path,
            n_rows_to_load=None,
            index_column=None,
        )

        output_dataframe = component.load().compute()

        assert output_dataframe.columns.tolist() == ["pdf_path", "file_name", "text"]

        if path == "tests/test_file/dummy.pdf":
            assert output_dataframe.shape == (1, 3)
            assert output_dataframe["file_name"].tolist() == ["dummy.pdf"]
            assert output_dataframe["text"].tolist() == ["Dummy PDF file\n"]
        else:
            assert output_dataframe.shape == (2, 3)
            file_names = output_dataframe["file_name"].tolist()
            file_names.sort()
            assert file_names == [
                "dummy_1.pdf",
                "dummy_2.pdf",
            ]
            assert output_dataframe["text"].tolist() == [
                "Dummy PDF file\n",
                "Dummy PDF file\n",
            ]
