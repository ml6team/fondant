from dask import delayed
import requests


@delayed
def load_url(url):
    """
    Given a URL, this function fetches the content of the URL using requests
    library and returns the content as a string. If the request fails, it
    returns None.

    Args:
        url (str): The URL to fetch the content from

    Returns:
        str or None: The content of the URL as a string or None if the request
                     fails
    """
    headers = {
        'User-Agent':
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/56.0.2924.87 Safari/537.36'
    }
    try:
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        return str(resp.content)
    except requests.exceptions.RequestException:
        return None


def retrieve_images(df):
    """
    Given a Dask DataFrame containing a column "image_urls" with image URLs,
    this function fetches the content of each URL and adds a new column
    "images" containing the content as bytes.

    Args:
        df (Dask DataFrame): a Dask DataFrame containing a column "image_urls"
                             with image URLs

    Returns:
        df (Dask DataFrame): a Dask DataFrame with an additional column
                             "images", where each row contains the content of
                             the corresponding image URL in that row.
    """
    df["images"] = df.apply(
        lambda x: load_url(x.image_urls),
        axis=1,
        meta=('images', object)
    )
    return df
