"""Unit test for language filter component."""
import pandas as pd

from src.main import LanguageFilterComponent


def test_run_component_test():
    """Test language filter component."""
    # Given: Dataframe with text in different languages
    data = [
        {"text": "Das hier ist ein Satz in deutscher Sprache"},
        {"text": "This is a sentence in English"},
        {"text": "Dit is een zin in het Nederlands"},
    ]
    dataframe = pd.DataFrame(data)

    component = LanguageFilterComponent(
        language="de",
    )
    dataframe = component.transform(dataframe=dataframe)

    # Then: dataframe only contains one german row
    assert len(dataframe) == 1
    assert dataframe.loc[0]["text"] == "Das hier ist ein Satz in deutscher Sprache"


def test_run_component_test_filter_out_all():
    """Test language filter component."""
    # Given: Dataframe with text in different languages
    data = [
        {"text": "Das hier ist ein Satz in deutscher Sprache"},
        {"text": "This is a sentence in English"},
        {"text": "Dit is een zin in het Nederlands"},
    ]
    dataframe = pd.DataFrame(data)

    component = LanguageFilterComponent(
        language="fr",
    )
    dataframe = component.transform(dataframe=dataframe)

    # Then: dataframe should contain no rows anymore
    assert len(dataframe) == 0
