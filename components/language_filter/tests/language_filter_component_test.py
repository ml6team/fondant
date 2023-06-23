import pandas as pd
from components.language_filter.src.main import LanguageFilterComponent
from fondant.component_spec import ComponentSpec
from dask.dataframe import from_pandas


def test_run_component_test():
    """Test language filter component"""

    # Given: Dataframe with text in different languages
    data = [{"text": "Das hier ist ein Satz in deutscher Sprache"}, {"text": "This is a sentence in English"},
            {"text": "Dit is een zin in het Nederlands"}]
    df = pd.DataFrame(data)
    ddf = from_pandas(df, npartitions=1)

    # When: The language filter component proceed the dataframe
    # and filter out all entries which are not written in german
    spec = ComponentSpec.from_file("../fondant_component.yaml")

    component = LanguageFilterComponent(spec, input_manifest_path="./dummy_input_manifest.json",
                                        output_manifest_path="./dummy_input_manifest.json",
                                        metadata={},
                                        user_arguments={"language": "de"}
                                        )

    ddf = component.transform(dataframe=ddf, **component.user_arguments)

    # Then: dataframe only contains one german row
    df = ddf.compute()
    assert len(df) == 1
    assert df.loc[0]["text"] == "Das hier ist ein Satz in deutscher Sprache"
