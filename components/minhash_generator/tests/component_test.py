"""Unit test for minhash generation component."""
import pandas as pd
from fondant.component_spec import ComponentSpec

from components.minhash_generator.src.main import MinHashGeneratorComponent


def test_run_component_test():
    """Test MinHash generation."""
    # Given: Dataframe with text, one duplicate in
    data = [{"data": "This is my first sentence"},
            {"data": "This is my first sentence"},
            {"data": "This is a different sentence"}]

    dataframe = pd.concat({"text": pd.DataFrame(data)}, axis=1, names=["text", "data"])

    # When: The text filter component proceed the dataframe
    spec = ComponentSpec.from_file("../fondant_component.yaml")

    component = MinHashGeneratorComponent(spec, input_manifest_path="./dummy_input_manifest.json",
                                        output_manifest_path="./dummy_input_manifest.json",
                                        metadata={},
                                        user_arguments={},
                                        )
    component.setup()
    dataframe = component.transform(dataframe=dataframe)

    # Then: dataframe contain minhashes for each entry
    assert any(dataframe.loc[0]["text"]["minhash"] == dataframe.loc[1]["text"]["minhash"])
    assert not any(dataframe.loc[0]["text"]["minhash"] == dataframe.loc[2]["text"]["minhash"])
