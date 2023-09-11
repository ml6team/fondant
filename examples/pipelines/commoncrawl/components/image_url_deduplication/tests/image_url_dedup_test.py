import pandas as pd

from src.main import ImageUrlDeduplication
import dask.dataframe as dd

test_data = {
    "image_url": {
        0: "https://eu.wikipedia.org/static/images/footer/wikimedia-button.png",
        1: "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Wappen_des_Saarlands.svg/75px-Wappen_des_Saarlands.svg.png",
        2: "https://newcanadianmedia.ca/wp-content/uploads/2023/03/unsplash-480x384.jpg",
        3: "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bb/Yugoslavia_1956-1990.svg/340px-Yugoslavia_1956-1990.svg.png",
        4: "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1f/Flag_of_Kosovo.svg/23px-Flag_of_Kosovo.svg.png",
        5: "https://upload.wikimedia.org/wikipedia/commons/thumb/0/06/Deutschland_Lage_des_Saarlandes.svg/285px-Deutschland_Lage_des_Saarlandes.svg.png",
        6: "https://globalvoices.org/wp-content/gv-static/img/tmpl/gv-logo-oneline-smallicon-600.png",
        7: "https://freesound.org/media/images/info.png",
        8: "https://newcanadianmedia.ca/wp-content/uploads/2022/12/Entrance-to-the-Asa-Cultural-museum-480x384.png",
        9: "https://newcanadianmedia.ca/wp-content/uploads/2022/12/GBV-Afghanistan-480x384.png",
    },
    "license_type": {
        0: "by-sa",
        1: "by-sa",
        2: "by",
        3: "by-sa",
        4: "by-sa",
        5: "by-sa",
        6: "by",
        7: "by",
        8: "by",
        9: "by",
    },
    "webpage_url": {
        0: "https://eu.wikipedia.org/wiki/Sarre",
        1: "https://eu.wikipedia.org/wiki/Sarre",
        2: "https://newcanadianmedia.ca/page/4/?molongui_byline=true&m_main_disabled=true&mca=molongui-disabled-link%2F",
        3: "https://hr.wikipedia.org/wiki/Demokratska_federativna_Jugoslavija",
        4: "https://hr.wikipedia.org/wiki/Demokratska_federativna_Jugoslavija",
        5: "https://eu.wikipedia.org/wiki/Sarre",
        6: "https://it.globalvoices.org/-/topics/human-rights/?m=201503",
        7: "https://freesound.org/people/tim.kahn/sounds/24244/",
        8: "https://newcanadianmedia.ca/page/4/?molongui_byline=true&m_main_disabled=true&mca=molongui-disabled-link%2F",
        9: "https://newcanadianmedia.ca/page/4/?molongui_byline=true&m_main_disabled=true&mca=molongui-disabled-link%2F",
    },
}


def test_image_url_deduplication():
    input_data = dd.from_pandas(pd.DataFrame(test_data), npartitions=2)

    # duplicate whole dataframe
    input_data = dd.concat([input_data, input_data], ignore_index=True)

    # Shuffle dataframe to test sorting
    input_data = input_data.sample(frac=1, random_state=42).reset_index(drop=True)

    input_data_len = len(input_data)
    input_data = input_data.rename(
        columns={"image_url": "image_image_url", "license_type": "image_license_type"}
    )
    component = ImageUrlDeduplication(spec=None)
    ddf = component.transform(input_data)

    # after dedup duplicates should be gone
    assert len(ddf) == input_data_len / 2  # nosec


def test_image_url_deduplication_check_license():
    input_data_by = dd.from_pandas(pd.DataFrame(test_data), npartitions=2)
    input_data_by["license_type"] = "by"

    input_data_by_nc_nd = dd.from_pandas(pd.DataFrame(test_data), npartitions=2)
    input_data_by_nc_nd["license_type"] = "by-nc-nd"

    input_data = dd.concat([input_data_by, input_data_by_nc_nd], ignore_index=True)
    input_data = input_data.rename(
        columns={"image_url": "image_image_url", "license_type": "image_license_type"}
    )

    component = ImageUrlDeduplication(spec=None)
    ddf = component.transform(input_data)

    # after dedup duplicates should be gone
    df = ddf.compute()
    assert (df["image_license_type"] == "by").any()  # nosec
