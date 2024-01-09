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
        print(f.name)
        spec = ComponentSpec(yaml.safe_load(f))
    spec = OperationSpec(spec)

    pdf_path = ["test_file/dummy.pdf", "test_folder"]

    for path in pdf_path:
        component = PDFReader(
            spec=spec,
            pdf_path=path,
            n_rows_to_load=None,
            index_column=None,
        )

        output_dataframe = component.load().compute()

        assert output_dataframe.columns.tolist() == ["file_name", "text"]

        if path == "test_file/dummy.pdf":
            assert output_dataframe.shape == (1, 2)
            assert output_dataframe["file_name"].tolist() == ["dummy.pdf"]
            assert output_dataframe["text"].tolist() == ["Dumm y PDF file"]
        else:
            assert output_dataframe.shape == (2, 2)
            assert output_dataframe["file_name"].tolist() == [
                "dummy_2.pdf",
                "dummy_1.pdf",
            ]
            assert output_dataframe["text"].tolist() == [
                "Dumm y PDF file",
                "Dumm y PDF file",
            ]
