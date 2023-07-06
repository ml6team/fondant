"""Unit test for language filter component."""
import pandas as pd

from components.language_filter.src.main import LanguageFilterComponent
from fondant.component_spec import ComponentSpec


def test_run_component_test():
    """Test language filter component."""
    # Given: Dataframe with text in different languages
    data = [{"text": "Das hier ist ein Satz in deutscher Sprache"},
            {"text": "This is a sentence in English"},
            {"text": "Dit is een zin in het Nederlands"}]
    dataframe = pd.DataFrame(data)

    # When: The language filter component proceed the dataframe
    # and filter out all entries which are not written in german
    spec = ComponentSpec.from_file("../fondant_component.yaml")

    component = LanguageFilterComponent(spec, input_manifest_path="./dummy_input_manifest.json",
                                        output_manifest_path="./dummy_input_manifest.json",
                                        metadata={},
                                        user_arguments={"language": "de"},
                                        )
    component.setup(language="de")
    dataframe = component.transform(dataframe=dataframe)

    # Then: dataframe only contains one german row
    assert len(dataframe) == 1
    assert dataframe.loc[0]["text"] == "Das hier ist ein Satz in deutscher Sprache"


def test_run_component_test_filter_out_all():
    """Test language filter component."""
    # Given: Dataframe with text in different languages
    data = [{"text": "Das hier ist ein Satz in deutscher Sprache"},
            {"text": "This is a sentence in English"},
            {"text": "Dit is een zin in het Nederlands"}]
    dataframe = pd.DataFrame(data)

    # When: The language filter component proceed the dataframe
    # and filter out all entries which are not written in french
    spec = ComponentSpec.from_file("../fondant_component.yaml")

    component = LanguageFilterComponent(spec, input_manifest_path="./dummy_input_manifest.json",
                                        output_manifest_path="./dummy_input_manifest.json",
                                        metadata={},
                                        user_arguments={"language": "fr"},
                                        )
    component.setup()
    dataframe = component.transform(dataframe=dataframe)

    # Then: dataframe should contain no rows anymore
    assert len(dataframe) == 0
