"""Unit test for text length filter component."""
import pandas as pd

from components.text_length_filter.src.main import TextLengthFilterComponent
from fondant.component_spec import ComponentSpec


def test_run_component_test():
    """Test text length filter component."""
    # Given: Dataframe with text with different lengths
    data = [{"data": "To less words"},
            {"data": "Still to less chars"},
            {"data": "This a valid sentence which should be still there"}]

    dataframe = pd.concat({"text": pd.DataFrame(data)}, axis=1, names=["text", "data"])

    # When: The text filter component proceed the dataframe
    spec = ComponentSpec.from_file("../fondant_component.yaml")

    component = TextLengthFilterComponent(spec, input_manifest_path="./dummy_input_manifest.json",
                                        output_manifest_path="./dummy_input_manifest.json",
                                        metadata={},
                                        user_arguments={"min_characters_length": 20,
                                                        "min_words_length": 4},
                                        )
    component.setup(min_characters_length=20, min_words_length=4)
    dataframe = component.transform(dataframe=dataframe)

    # Then: dataframe only contains one row
    assert len(dataframe) == 1
    assert dataframe.loc[2]["text"]["data"] == "This a valid sentence which should be still there"
